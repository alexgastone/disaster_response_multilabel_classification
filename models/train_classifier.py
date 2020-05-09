import sys
import argparse
import pandas as pd
from sqlalchemy import create_engine
import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import ktrain
from ktrain import text
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import roc_auc_score


def load_data(database_filepath, table_name):
    engine = create_engine('sqlite:///' + str(database_filepath))
    #table_name = str(database_filepath).split('.')[0]
    df = pd.read_sql_table(table_name, engine)

    label_cols = df.columns.tolist()[5:]
    category_labels = df[label_cols].values 

    df_train = df[df['split']=='train']
    df_val = df[df['split']=='validation']

    return df_train, df_val, label_cols

def preprocess_data(train, val, label_cols, args):
    """
    Tokenizes & pads text ('messages' column of pandas dataframe) using Bert Tokenizer
        note: padding transforms lists of integers (tokenized inputs) into a 2D np array
              of shape number of samples x max_len, padded with 0
    Splits training and validation according to 'split' column

    :param text: text data to tokenize and pad
    :param args: classifier arguments dictionary

    Returns array of token IDs (=0 for padded text) for x_train, y_train, x_test, and y_test
            and BERT preprocessor object from ktrain
    """
    (x_train, y_train), (x_test, y_test), preproc = text.texts_from_df(train,                           # training
                                                                      'message',                        # text col                                            
                                                                       label_cols,                      # label col
                                                                       val,                             # validation
                                                                       max_features=args['NUM_WORDS'],  # max features
                                                                       maxlen=args['MAX_LEN'],           # max len
                                                                       ngram_range=args['NGRAM'],       # n_gram 
                                                                       preprocess_mode='bert')          # model

    return x_train, y_train, x_test, y_test, preproc

def build_model(x_train, y_train, x_test, y_test, preproc):
    """
    Builds and initializes model

    :param x_train: preprocessed training dataset features (messages)
    :param y_train: preprocessed training dataset labels
    :param x_test: preprocessed testing dataset features
    :param y_test: preprocessed testing dataset labels
    :param preproc: preprocessor object

    Returns model and learner object
    """
    # instantiate model
    model = text.text_classifier('bert', (x_train, y_train), preproc=preproc, multilabel=True)
    
    # wrap model and data in learner object
    learner = ktrain.get_learner(model, train_data=(x_train, y_train), val_data=(x_test, y_test))

    return model, learner

class RocAucEvaluation(Callback):
  def __init__(self, validation_data=(), interval=1):
    super(Callback, self).__init__()

    self.interval = interval
    self.X_val, self.y_val = validation_data

  def on_epoch_end(self, epoch, logs={}):
    # evaluate at the end of epoch
    if epoch % self.interval == 0:
      y_pred = self.model.predict(self.X_val, verbose=0)
      score = roc_auc_score(self.y_val, y_pred, average='weighted')
      print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))

def train(x_train, y_train, x_test, y_test, learner, args, with_callback=False):
    """
    Runs training using triangular learning rate scheduler

    :param x_train: preprocessed training dataset features (messages)
    :param y_train: preprocessed training dataset labels
    :param x_test: preprocessed testing dataset features
    :param y_test: preprocessed testing dataset labels
    :param learner: learner object
    :param args: classifier arguments
    :param with_callback: whether to use RocAucEvaluation callback, prints average weighted
                          ROC-AUC score on validation set at the end of an epoch 
    """
    if with_callback:
        RocAuc = RocAucEvaluation(validation_data=(x_test, y_test), interval=1)
        learner.autofit(args['LEARN_RATE'], args['NUM_EPOCHS'], callbacks=[RocAuc])
    else:
        learner.autofit(args['LEARN_RATE'], args['NUM_EPOCHS'])


def save_model(learner, preproc, model_filepath):
    # Instantiate predictor object
    predictor = ktrain.get_predictor(learner.model, preproc=preproc)
    predictor.save(model_filepath)



def main():
    parser = argparse.ArgumentParser(description='Train our classifier with pretrained BERT')
    parser.add_argument('--database_filepath', type=str, required=True, default='../data/DisasterResponse_split.db', help='Please provide path to the Disaster Response database')
    parser.add_argument('--table_name', type=str, required=True, default='DisasterResponse_split', help='Please provide the name of the table in the Disaster Response database')
    parser.add_argument('--model_filepath', type=str, required=True, default='bert_finetuned_model', help='Please provide path to save trained model')

    args = parser.parse_args()
    database_filepath = args.database_filepath
    table_name = args.table_name
    model_filepath = args.model_filepath

    print('Loading data...\n    Database: {}\n    Table name: {}'.format(database_filepath, table_name))
    df_train, df_val, label_cols = load_data(database_filepath, table_name)

    args = {
        'NUM_WORDS' : 50000,
        'MAX_LEN' : 128,
        'NGRAM' : 1,
        'BATCH_SIZE' : 32,
        'NUM_EPOCHS' : 2,
        'LEARN_RATE' : 1e-5
    }

    print('BERT tokenization and preprocessing of messages...')
    x_train, y_train, x_test, y_test, preproc = preprocess_data(df_train, df_val, label_cols, args)

    print('Building BERT model...')
    _, learner = build_model(x_train, y_train, x_test, y_test, preproc)

    print('Training the classifier...')
    train(x_train, y_train, x_test, y_test, learner, args, with_callback=False)

    print('Saving model...\n    Model: {}'.format(model_filepath))
    save_model(learner, preproc, model_filepath)

    print('Trained model saved! Preprocessing required for the model is saved at {}.preproc'.format(model_filepath))


if __name__ == '__main__':
    main()