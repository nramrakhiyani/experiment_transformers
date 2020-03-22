import os
import spacy
import keras
import pickle
import codecs
import numpy as np

from keras.models import Model
from keras.models import load_model
from keras_contrib.layers import CRF
from keras.utils import to_categorical
from keras.layers import Dense, Input, LSTM, TimeDistributed, Dropout, Bidirectional

def train_bert_based_entity_extraction(train_data_path, model_save_path, network_params, spacy_transformer_model):
	#Reading data
	sentences = []
	tokens = []
	labels = []
	all_labels = []
	max_sent_length = network_params['max_sentence_length']
	train_data = codecs.open(train_data_path, 'r', encoding = 'utf-8', errors = 'ignore')
	for line in train_data:
		line = line.strip()
		if(len(line) <= 0 and len(tokens) > 0):
			sentences.append((tokens, labels))
			#if(max_sent_length < len(tokens)):
			#	max_sent_length = len(tokens)
			tokens = []
			labels = []
			continue
		tokens.append(line.split('\t')[0])
		labels.append(line.split('\t')[3])
		if(line.split('\t')[3] not in all_labels):
			all_labels.append(line.split('\t')[3])
	train_data.close()

	#Preparing spacy model loaded with the BERT model
	print ('Loading spacy model - ' + spacy_transformer_model)
	nlp = spacy.load(spacy_transformer_model)
	print ('Loaded spacy transformer model')

	tag_index = {}
	index = 0
	for label in all_labels:
		tag_index[label] = index
		index += 1

	#Preparing X_train (using spacy_transformer's BERT models) and Y_train
	print ('Preparing X_train (using spacy_transformer\'s BERT models) and Y_train')
	X_train = np.empty((len(sentences), max_sent_length, network_params['input_dim']))
	Y_tr = np.empty((len(sentences), max_sent_length))
	for i in range(len(sentences)):
		if(i%50 == 0):
			print ('.', end = '')
		if(i%500 == 0):
			print ('')

		sentence = sentences[i]
		tokens = sentence[0]
		labels = sentence[1]
		sentence_str = ' '.join(tokens)

		#Getting BERT embeddings
		curr_doc = nlp(sentence_str)

		if(len(tokens) >= max_sent_length):
			limit = max_sent_length
		else:
			limit = len(tokens)

		for j in range(limit):
			X_train[i][j] = curr_doc._.trf_last_hidden_state[j+1]
			Y_tr[i][j] = tag_index[labels[j]]

		if(limit < max_sent_length):
			for j in range(limit, max_sent_length):
				X_train[i][j] = np.zeros(network_params['input_dim'])
				Y_tr[i][j] = tag_index['O']

	Y_train = [to_categorical(i, num_classes = len(tag_index)) for i in Y_tr]
	print ('Prepared X_train and Y_train')

	#Building Model - Start ================================================
	#Input Word Embedding
	word_emb_inp = Input(shape=(None, network_params['input_dim']))
	
	model1 = None
	if(network_params['dropout'] > 0):
		#Dropout
		dropout1 = Dropout(network_params['dropout'])(word_emb_inp)

		#Bidirectional LSTM layer (with Dropout)
		model1 = Bidirectional(LSTM(units = network_params['lstm_dim'], return_sequences = True))(dropout1)
	else:
		#Bidirectional LSTM layer (Direct without Dropout)
		model1 = Bidirectional(LSTM(units = network_params['lstm_dim'], return_sequences = True))(word_emb_inp)

	#TimeDistributed layer
	model2 = TimeDistributed(Dense(len(tag_index), activation = "relu"))(model1)

	#The output CRF layer
	crf = CRF(len(tag_index))
	out = crf(model2)
	#Building Model - End ==================================================

	#Model instantiation
	model = Model(word_emb_inp, out)

	#Model Compilation
	model.compile(optimizer = network_params['optimizer'], loss = crf.loss_function, metrics = [crf.accuracy])

	model.summary()

	model.fit(X_train, np.array(Y_train), batch_size = network_params['batch_size'], epochs = network_params['epochs'])

	#Model Writing
	model.save_weights(model_save_path)

	#Saving tag_index dict
	model_dir = os.path.dirname(model_save_path)
	
	tag_index_pickle_path = os.path.join(model_dir, 'tag_index.pickle')
	with open(tag_index_pickle_path, 'wb') as pickle_file_path:
		pickle.dump(tag_index, pickle_file_path)

	print ('Model trained and saved')

def test_bert_based_entity_extraction(test_data_path, model_path, test_predictions_path, network_params, spacy_transformer_model):
	#Reading data
	sentences = []
	tokens = []
	labels = []
	test_data = codecs.open(test_data_path, 'r', encoding = 'utf-8', errors = 'ignore')
	for line in test_data:
		line = line.strip()
		if(len(line) <= 0 and len(tokens) > 0):
			sentences.append((tokens, labels))
			tokens = []
			labels = []
			continue
		tokens.append(line.split('\t')[0])
		labels.append(line.split('\t')[3])
	test_data.close()

	#Preparing spacy model loaded with the BERT model
	print ('Loading spacy model - ' + spacy_transformer_model)
	nlp = spacy.load(spacy_transformer_model)

	model_dir = os.path.dirname(model_path)
	tag_index = pickle.load(os.path.join(model_dir, 'tag_index.pickle'))
	index_tag = {}
	for tag in tag_index:
		index_tag[tag_index[tag]] = tag

	#Building Model - Start ================================================
	#Input Word Embedding
	word_emb_inp = Input(shape=(None, network_params['input_dim']))
	
	model1 = None
	if(network_params['dropout'] > 0):
		#Dropout
		dropout1 = Dropout(network_params['dropout'])(word_emb_inp)

		#Bidirectional LSTM layer (with Dropout)
		model1 = Bidirectional(LSTM(units = network_params['lstm_dim'], return_sequences = True))(dropout1)
	else:
		#Bidirectional LSTM layer (Direct without Dropout)
		model1 = Bidirectional(LSTM(units = network_params['lstm_dim'], return_sequences = True))(word_emb_inp)

	#TimeDistributed layer
	model2 = TimeDistributed(Dense(len(tag_index), activation = "relu"))(model1)

	#The output CRF layer
	crf = CRF(len(tag_index))
	out = crf(model2)
	#Building Model - End ==================================================

	#Model instantiation
	model = Model(word_emb_inp, out)

	#Model Loading
	model.load_weights(model_file_path)

	#Testing on each example and collecting tags
	output_file = codecs.open(test_predictions_path, 'w', encoding = 'UTF-8')
	for i in range(len(sentences)):
		sentence = sentences[i]
		tokens = sentence[0]
		labels = sentence[1]
		sentence_str = ' '.join(tokens)

		#Getting BERT embeddings
		curr_doc = nlp(sentence_str)

		X_test = np.empty((1, len(tokens), network_params['input_dim']))
		for j in range(tokens):
			X_test[0][j] = curr_doc._.trf_last_hidden_state[j+1]

		p = model.predict([np.array([X_test])])
		p = np.argmax(p, axis = -1)
		for token, label, pred_tag in zip(tokens, labels, p[0]):
			output_file.write(token + '\t' + label + '\t' + index_tag[pred_tag] + '\n')
		output_file.write('\n')
	output_file.close()
	print ('Testing Complete.')