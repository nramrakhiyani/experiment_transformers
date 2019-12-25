import spacy
import keras
import numpy
import codecs
from keras.models import load_model
from keras.models import Model
from keras.layers import Dense, Input
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_recall_fscore_support

def train_bert_based_classification(train_data_path, model_save_path, dense_layer_required, network_params):
	#Reading data
	sentences = []
	labels = []
	train_data = codecs.open(train_data_path, 'r', encoding = 'utf-8', errors = 'ignore')
	for line in train_data:
		line = line.strip()
		if(len(line) <= 0):
			continue
		sentences.append(line.split('\t')[0])
		labels.append(line.split('\t')[1])
	train_data.close()

	#Preparing spacy model loaded with the BERT model
	print ('Loading spacy model')
	nlp = spacy.load("en_trf_bertbaseuncased_lg")

	#Preparing X_train (using spacy_transformer's BERT models) and Y_train
	X_train = numpy.zeros((len(sentences), network_params['input_dim']))
	Y_train = numpy.zeros((len(sentences), ))
	for i in range(len(sentences)):
		Y_train[i] = float(labels[i])
		curr_doc = nlp(sentences[i])
		X_train[i] = curr_doc._.trf_last_hidden_state[0]

	ohe = OneHotEncoder()
	Y_train = Y_train.reshape(-1,1)
	Y_train_onehot = ohe.fit_transform(Y_train).toarray()

	#Model Creation and Training
	input = Input(shape = (network_params['input_dim'],))

	output = None
	if(dense_layer_required.lower() == 'true'):
		hidden = Dense(300, activation = 'relu')(input)
		output = Dense(2, activation = 'softmax')(hidden)
	else:
		output = Dense(2, activation = 'softmax')(input)

	model = Model(inputs = input, outputs = output)
	model.compile(optimizer = network_params['optimizer'], loss = 'categorical_crossentropy')
	model.summary()

	model.fit(X_train, Y_train_onehot, epochs = network_params['epochs'], batch_size = network_params['batch_size'])

	#Model Writing
	model.save(model_save_path)
	print ('Model trained and saved')

def test_bert_based_classification(test_data_path, model_path, test_predictions_path, network_params):
	#Reading data
	sentences = []
	test_data = codecs.open(test_data_path, 'r', encoding = 'utf-8', errors = 'ignore')
	for line in test_data:
		line = line.strip()
		if(len(line) <= 0):
			continue
		sentences.append(line)
	test_data.close()

	#Preparing spacy model loaded with the BERT model
	print ('Loading spacy model')
	nlp = spacy.load("en_trf_bertbaseuncased_lg")

	#Preparing X_test (using spacy_transformer's BERT models)
	X_test = numpy.zeros((len(sentences), network_params['input_dim']))
	for i in range(len(sentences)):
		curr_doc = nlp(sentences[i])
		X_test[i] = curr_doc._.trf_last_hidden_state[0]

	#Model Loading
	model = load_model(model_path)

	#Model Predictions
	Y_pred = []
	for i in range(len(sentences)):
		prediction = model.predict([numpy.array([X_test[i]])])
		pred = numpy.argmax(prediction, axis = -1)
		Y_pred.append(pred[0])

	test_predictions_file = codecs.open(test_predictions_path, 'w', encoding = 'utf-8', errors = 'ignore')
	for k in range(len(Y_pred)):
		test_predictions_file.write(sentences[k] + '\t' + str(Y_pred[k]) + '\n')
	test_predictions_file.close()
	print ('Testing Complete.')

def evaluate_bert_based_classification(test_data_path, model_path, test_predictions_path, network_params):
	#Reading data
	sentences = []
	labels = []
	test_data = codecs.open(test_data_path, 'r', encoding = 'utf-8', errors = 'ignore')
	for line in test_data:
		line = line.strip()
		if(len(line) <= 0):
			continue
		sentences.append(line.split('\t')[0])
		labels.append(line.split('\t')[1])
	test_data.close()

	#Preparing spacy model loaded with the BERT model
	print ('Loading spacy model')
	nlp = spacy.load("en_trf_bertbaseuncased_lg")

		#Preparing X_test (using spacy_transformer's BERT models)
	X_test = numpy.zeros((len(sentences), network_params['input_dim']))
	Y_test = numpy.zeros((len(sentences), ))
	for i in range(len(sentences)):
		Y_test[i] = float(labels[i])
		curr_doc = nlp(sentences[i])
		X_test[i] = curr_doc._.trf_last_hidden_state[0]

	#Model Loading
	model = load_model(model_path)

	#Model Predictions
	Y_pred = []
	for i in range(len(sentences)):
		prediction = model.predict([numpy.array([X_test[i]])])
		pred = numpy.argmax(prediction, axis = -1)
		Y_pred.append(pred[0])
	
	test_predictions_file = codecs.open(test_predictions_path, 'w', encoding = 'utf-8', errors = 'ignore')
	for k in range(len(Y_pred)):
		test_predictions_file.write(sentences[k] + '\t' + str(labels[k]) + '\t' + str(Y_pred[k]) + '\n')
	test_predictions_file.close()
	
	print (confusion_matrix(Y_test, Y_pred))
	print (precision_recall_fscore_support(labels, Y_pred))
	print ('Evaluation Complete.')