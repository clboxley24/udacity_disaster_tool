# import libraries
import pandas as pd
from sqlalchemy import create_engine
import os
import sys
import sqlite3
import pickle
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(database_filepath):
    """
    Description:
        Loads data from database
    
    Parameters:
        database_filepath (string): Pathway to SQLite database file
    
    Returns:
        (DataFrame) X: Independent variables
        (DataFrame) Y: Dependent variables
        (List) categories: Y column names
    """

    con = sqlite3.connect(database_filepath)
    df = pd.read_sql_query("SELECT * from surveys", con)

    df.columns
                           
    X = df['message']
    Y = df.iloc[:,4:]

    # get names of the categories
    categories = Y.columns

    return(X, Y, categories)


def tokenize(text):

    """
    Description:
        Tokenizes messages
    
    Parameters:
       text (string): messages
    
    Returns:
        (DataFrame) clean_tokens: array of tokenized messages
    """

    # Extract the word tokens from messages
    tokens = nltk.word_tokenize(text)
    
    # Lemmanitizer 
    lemmatizer = nltk.WordNetLemmatizer()

    # List of clean tokens
    clean_tokens = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]

    return(clean_tokens)


def build_model():
    """
    Description:
        Builds nlp pipeline
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=42)))
    ])

    return(pipeline)

def evaluate_model(pipeline, X_test, Y_test, categories):
    """
    Description:
        Evaluates models performance in predicting message categories
    
    Parameters:
        pipeline (Classification): stored classification model
        X_test (string): Independent variables
        Y_test (string): Dependent variables
        categories (list): Stores message category labels
    """
    Y_pred = pipeline.predict(X_test)

    for i in range(0, len(categories)):
        print("Category:", categories[i],"\n", classification_report(Y_test.iloc[:, i].values, Y_pred[:, i]))

def save_model(model, model_filepath):
    """
    Description:
        Saves model to pickle file
    
    Parameters:
        model (Classification): stored model
        model_filepath (string): Pathway to pickle file
    """
    
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, categories = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, categories)

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