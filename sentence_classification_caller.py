import sys
from sentence_classification import *

mode = sys.argv[1] #train / test / evaluate
input_file_path = sys.argv[2] #if mode is train, this is training data path else test data path
model_file_path = sys.argv[3] #if mode is train, this is where the learnt model will be stored else path from where learnt model needs to be read

network_params = {}
network_params['input_dim'] = 768 #768 for BERT base
network_params['optimizer'] = 'adam' #adam, sgd and adadelta
network_params['epochs'] = 10
network_params['batch_size'] = 16

if(mode == 'train'):
	dense_layer_required = sys.argv[4] #true / false
	train_bert_based_classification(input_file_path, model_file_path, dense_layer_required, network_params)

elif(mode == 'test'):
	prediction_file_path = sys.argv[4]
	test_bert_based_classification(input_file_path, model_file_path, prediction_file_path)

elif(mode == 'evaluate'):
	prediction_file_path = sys.argv[4]
	evaluate_bert_based_classification(input_file_path, model_file_path, prediction_file_path)