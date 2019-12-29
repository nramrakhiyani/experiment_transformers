# experiment_transformers
BERT/XLNet/similar based transformers are being employed widely to solve various NLP tasks. This repo contains a set of codes I use for my NLP tasks involving such transformer models.

Useful URLs
===========
https://colab.research.google.com/github/explosion/spacy-pytorch-transformers/blob/master/examples/Spacy_Transformers_Demo.ipynb#scrollTo=KRQOGjDdfXnN


Commands
========
1) For Training and saving the model

python sentence_classification_caller.py train <input_training_file> <path_for_saving_model> <dense_layer_required> <spacy_transformer_model>

<input_training_file>: Should be in the format "sentence<TAB>integer_label"
<dense_layer_required>: If a dense layer is required between the transformer embedding input layer and the softmax prediction layer. Available values are true and false.
<spacy_transformer_model>: The transformers model supported by spacy_transformers. Available values are en_trf_bertbaseuncased_lg, en_trf_xlnetbasecased_lg, en_trf_robertabase_lg, en_trf_distilbertbaseuncased_lg

2) For evaluation (if test data labels are available)

python sentence_classification_caller.py evaluate <input_test_file> <path_for_loading_saved_model> <test_predictions_path> <spacy_transformer_model>

<input_test_file>: Should be in the format "sentence<TAB>integer_label"
<spacy_transformer_model>: The transformers model supported by spacy_transformers. Available values are en_trf_bertbaseuncased_lg, en_trf_xlnetbasecased_lg, en_trf_robertabase_lg, en_trf_distilbertbaseuncased_lg


3) For testing (if test data labels are not available)

python sentence_classification_caller.py test <input_test_file> <path_for_loading_saved_model> <test_predictions_path> <spacy_transformer_model>

<input_test_file>: Should be in the format "sentence<TAB>integer_label"
<spacy_transformer_model>: The transformers model supported by spacy_transformers. Available values are en_trf_bertbaseuncased_lg, en_trf_xlnetbasecased_lg, en_trf_robertabase_lg, en_trf_distilbertbaseuncased_lg