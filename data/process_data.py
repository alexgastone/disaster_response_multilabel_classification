import sys
import pandas as pd 
from sqlalchemy import create_engine


def load_data(training_filepath, validation_filepath):
    """
    Loads and concatenates training and validation data. 
    Training/validation split are preserved in 'split' column

    :param training_filepath: path to training dataset csv
    :param validation_filepath: path to validation dataset csv

    returns df: pandas dataframe of training and validation data
    """
    training = pd.read_csv(training_filepath)
    validation = pd.read_csv(validation_filepath)
    df = pd.concat([training, validation], ignore_index=True)
    return df


def clean_data(df):
    """
    Preprocesses data - cleans categories and drops duplicates

    :param df: pandas dataset to clean

    returns df: cleaned dataset
    """
    # drop the following columns, no samples for them in dataset
    df.drop(['PII', 'child_alone', 'offer'], axis=1, inplace=True)

    # if sum of labels is equal to 0, then uncategorized evaluates to 1
    category_labels = df.columns[5:]
    df['sum'] = df[category_labels].sum(axis=1)
    df['uncategorized'] = df['sum'].apply(lambda x: 1 if x==0 else 0)

    # drop sum
    df.drop(['sum'], axis=1, inplace=True)

    # drop duplicates
    df.drop_duplicates(inplace=True, ignore_index=True)

    return df

def save_data(df, database_filepath):
    """
    Saves pandas dataframe as SQL table in database

    :param df: dataframe to save
    :param database_filepath: name of database and table
    """
    engine = create_engine('sqlite:///' + str(database_filepath))

    # save as SQL table
    df.to_sql('DisasterResponse', engine, index=False)


def main():
    if len(sys.argv) == 4:

        # training and validation are available through FigureEight's open data
        training_filepath, validation_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    Training data: {}\n    Validation data: {}'
              .format(training_filepath, validation_filepath))
        df = load_data(training_filepath, validation_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    Database: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of training and validation '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python3.7 process_data.py '\
              'disaster_response_messages_training.csv '\
              'disaster_reponse_messages_validatiom=n.csv DisasterResponse.db')


if __name__ == '__main__':
    main()