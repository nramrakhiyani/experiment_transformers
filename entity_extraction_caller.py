import sys
from sentence_classification import *

mode = sys.argv[1] #train / test / evaluate
input_file_path = sys.argv[2] #if mode is train, this is training data path else test data path
model_file_path = sys.argv[3] #if mode is train, this is where the learnt model will be stored else path from where learnt model needs to be read

network_params = {}
network_params['input_dim'] = 768 #768 for BERT base
network_params['lstm_dim'] = 768
network_params['optimizer'] = 'adam' #adam, sgd and adadelta
network_params['epochs'] = 10
network_params['batch_size'] = 16
network_params['dropout'] = 0

if(mode == 'train'):
	spacy_transformer_model = sys.argv[5]
	train_bert_based_entity_extraction(input_file_path, model_file_path, network_params, spacy_transformer_model)

elif(mode == 'test'):
	prediction_file_path = sys.argv[4]
	spacy_transformer_model = sys.argv[5]
	test_bert_based_entity_extraction(input_file_path, model_file_path, prediction_file_path, network_params, spacy_transformer_model)