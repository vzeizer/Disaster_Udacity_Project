import sys

# import libraries

import nltk
nltk.download(['punkt', 'wordnet','stopwords'])

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.model_selection import GridSearchCV
#from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import make_scorer
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sqlalchemy import create_engine

from sklearn.multioutput import MultiOutputClassifier

import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import classification_report



stop_words = set(stopwords.words('english'))


def load_data(database_filepath):
    '''
    This function loads the data
    input: database_filepath: name of the sql database
    outputs:- X: the features
            - Y: the target
            - category_names: the classes the model will predict
    '''
#    engine = create_engine('sqlite:///InsertDatabaseName.db')
    engine = create_engine('sqlite:///{0}'.format(database_filepath))
    df=pd.read_sql_table("InsertTableName",con=engine)
    df=df.drop(columns=['original'], axis=1)
    X = df[['message']]
    Y = df.drop(columns=['id','message','genre'],axis=1)
    category_names =Y.columns
    
    return X, Y, category_names 


def tokenize(text):
    """
    This function basically does tokenization
    iinputs: text
    outputs: clean_tokens, text after normalization, stop-words removal and lemmatization
    """
    
    tokens = word_tokenize(text)
    tokens=[word.lower() for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words] 

    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens    


def build_model(X_train,Y_train):
    """
    This function builds a machine learning model
    inputs: X_train, Y_train: features and target of the training data
    outputs: cv: model you built though pipeline
    """
    
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
    
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
    'vect__ngram_range': ((1, 1), (1, 2)),
    'vect__max_df': (0.5, 1.0),
    }

    cv = GridSearchCV(pipeline, param_grid=parameters,n_jobs=-1)
    
#    return pipeline
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    This function prints the classification report in the test set.
    inputs: -model: model that was trained
            -X_test, Y_test: features and targets of the test set
            - category_names: the category names
    outputs: just prints the classification report in the test set
    """
    m = MultiLabelBinarizer().fit(Y_test)
    X_test=np.reshape(X_test,(-1,2))
    X_test=X_test.flatten()
    y_pred = model.predict(X_test)
#    print(classification_report(m.transform(Y_test),m.transform(y_pred),target_names=category_names))
    print(classification_report(Y_test,y_pred,target_names=category_names))


def save_model(model, model_filepath):
    """
    This function saves a model in pickle
    inputs: -model: model that was trained
            -model_filepath: file path and name of the model you want to save
    outputs:- just saves the model in .pkl
    """
    
    try:
        joblib.dump(model.best_estimator_, model_filepath)
    except:
        joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
#        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        X_train, X_test, Y_train, Y_test = train_test_split(X.values, Y.values, test_size=0.2)
        
        print('Building model...')
        model = build_model(X_train,Y_train)
        
        print('Training model...')
        X_train=np.reshape(X_train,(-1,2))
    
        X_train=X_train.flatten()
        
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()