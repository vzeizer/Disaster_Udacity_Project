import sys
import sqlalchemy
from sqlalchemy import create_engine
import numpy as np
import pandas as pd

def load_data(messages_filepath, categories_filepath):
    """
    This function loads the data.
    Inputs: - messages_filepath: the csv containing the messages
            - categories_filepath: the csv cantaining the categories
    Outputs: 
            -df: a dataframe which is a merge of the messages and categories
            -categories: categories dataframe when data is read
            -messages: messages dataframe when data is read
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.join(categories,lsuffix=' ')
    df=df.drop(columns=['id'],axis=1)
    df=df.rename({'id ':"id"},axis='columns')
    
    return df, categories, messages

def clean_data(df,categories):
    """
    This function cleans the dataframes df and categories
    input: df, categories
    outputs:- categories: dataframe after being worked conveniently to represent the categories
            - df: dataframe after being conveniently cleaned
    
    """
    categories =categories['categories'].str.split(";", expand=True) 
    category_colnames = [categories.iloc[0][xx].split('-')[0] for xx in range(categories.shape[1])]
    categories.columns = category_colnames
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].transform(lambda x: x[-1]).values
    
    # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    for i in range(len(categories['related'])):
        if(categories['related'][i]==2):
            categories['related'][i]=0
        else:
            pass


    df.drop(['categories'],axis=1,inplace=True)

    df = df.join(categories)
    
    df.drop_duplicates(inplace=True)
    
    return df, categories
    
    
def save_data(df, database_filename):
    """
    This function saves the dataframe into a database
    inputs: df, database_filnename: the name you want to save the database
    outputs: just saves in sql your dataframe to be used after
    """
    engine = create_engine('sqlite:///{0}'.format(database_filename))
    df.to_sql('InsertTableName', engine, index=False,if_exists='replace')

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df,categories,messages = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df,categories = clean_data(df,categories)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()