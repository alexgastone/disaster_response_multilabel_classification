# Disaster Response Project

### Context:
In the face of a global pandemic and a majority of resources (including time) being rightly alotted to fighting it, I started thinking about working on a project that would be aimed at maximizing efficiency of strained responses to non-CovID related emergencies. There are thousands of messages that teams managing response efforts have to comb through in order to determine and assign the category of response required (ex. food, shelter..). 

Figure Eight provides a dataset of labeled multilingual disaster messages from social, direct, and news sources. These labels correspond to the categories of disaster responses.

#### The goal for this project is therefore to build a model that can accurately classify these messages. 

> Multilabel text classification task with imbalanced classes 

### Model:
I've trained a few different models of increasing complexity so far (Naive Bayes, deep models such as Bidirectional GRU or a fastText-like model)

* Best performance is by finetuning pretrained **BERT-base-uncased model** + Classifier (Dense output layer with sigmoid activation to predict probabilities) using the `ktrain` library
	* Training accuracy: 0.9423, validation accuracy: 0.9457
	* more flexible model is in the works, built using the `transformers` library from **HuggingFace** with tensorflow/keras

* Threshold moving for imbalanced classes: I've chosen to take the threshold that corresponds to the max geometric mean between the weighted ROC-AUC scores and the F1 scores for the validation set.

