# Disaster Response Project

### The goal:
In the face of a global pandemic and a majority of resources (including time) being alotted to fighting it, I started thinking about maximizing efficiency of responses to non-CovID related emergencies. These are emergencies that countries and teams directing responses efforts still have to face and triage through. 
> Figure Eight provides a dataset of labeled multilingual disaster messages from social, direct, and news sources. These labels correspond to the categories of disaster responses that best fit.

### The problem:
* Faced with a multilabel text classification task with imbalanced classes 

### Model:
* Best performance with pretrained **BERT-base-uncased model** (vs Bidirectional GRU & fastText-like model) + Classifier (Dense output layer with sigmoid activation to predict probabilities) using the `ktrain` library
	* more flexible model is in the works, built using the `transformers` library from **HuggingFace** with tensorflow/keras
* Training accuracy: 0.9423, validation accuracy: 0.9457
* Threshold (for classification from probabilities) so far is based on max of geometric mean of the weighted ROC-AUC score and F1 score between true and predicted labels of the validation set
