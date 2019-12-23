import spacy
import keras
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Input
from sklearn.metrics import confusion_matrix
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
	nlp = spacy.load("en_trf_bertbaseuncased_lg")

	#Preparing X_train (using spacy_transformer's BERT models) and Y_train
	X_train = numpy.zeros((len(sentences), network_params['input_dim']))
	Y_train = numpy.zeros((len(sentences), ))
	for i in range(len(sentences)):
		Y_train[i] = float(labels[i])
		curr_doc = nlp(sentences[i])
		X_train[i] = curr_doc._.trf_last_hidden_state[0]

	#Model Creation and Training
	model = Sequential()
	model.add(Input(shape = (network_params['input_dim'],)))

	if(dense_layer_required.lower() == 'true'):
		model.add(Dense(300, activation = 'relu'))

	model.add(Dense(2, activation = 'softmax'))

	model.compile(optimizer = network_params['optimizer'], loss = 'categorical_crossentropy')
	model.print_summary()

	model.fit(X_train, Y_train, epochs = network_params['epochs'], batch_size = network_params['batch_size'])

	#Model Writing
	model.save(model_save_path)


def test_bert_based_classification(test_data_path, model_path, test_predictions_path):
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
	nlp = spacy.load("en_trf_bertbaseuncased_lg")

	#Preparing X_test (using spacy_transformer's BERT models)
	X_test = numpy.zeros((len(sentences), network_params['input_dim']))
	for i in range(len(sentences)):
		curr_doc = nlp(sentences[i])
		X_test[i] = curr_doc._.trf_last_hidden_state[0]

	#Model Loading
	model = load_model(model_path)

	#Model Predictions
	predictions = []
	for i in range(len(sentences)):
		prediction = model.predict([numpy.array([X_test[i]])])
		pred = np.argmax(prediction, axis = -1)
		predictions.append(pred[0])

	test_predictions_file = codecs.open(test_predictions_path, 'w', encoding = 'utf-8', errors = 'ignore')
	for k in range(len(predictions)):
		test_predictions_file.write(sentences[k] + '\t' + str(predictions[k]) + '\n')
	test_predictions_file.close()


def evaluate_bert_based_classification(test_data_path, model_path, test_predictions_path):
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
	nlp = spacy.load("en_trf_bertbaseuncased_lg")

		#Preparing X_test (using spacy_transformer's BERT models)
	X_test = numpy.zeros((len(sentences), network_params['input_dim']))
	for i in range(len(sentences)):
		curr_doc = nlp(sentences[i])
		X_test[i] = curr_doc._.trf_last_hidden_state[0]

	#Model Loading
	model = load_model(model_path)

	#Model Predictions
	predictions = []
	for i in range(len(sentences)):
		prediction = model.predict([numpy.array([X_test[i]])])
		pred = np.argmax(prediction, axis = -1)
		predictions.append(pred[0])
	
	test_predictions_file = codecs.open(test_predictions_path, 'w', encoding = 'utf-8', errors = 'ignore')
	for k in range(len(predictions)):
		test_predictions_file.write(sentences[k] + '\t' + str(labels[k]) + '\t' + str(predictions[k]) + '\n')
	test_predictions_file.close()
	
	print (confusion_matrix(labels, predictions)
	print (precision_recall_fscore_support(labels, predictions))