import sys
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import  GridSearchCV
import numpy as np
from sklearn.metrics import confusion_matrix
import pickle

def load_data(database_filepath):
    
    """
    Load and merge datasets
    input:
         database name
    outputs:
        X: messages 
        y: everything esle
        category names.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql("SELECT * FROM TableName", engine)
    X = df['message']
    Y = df.drop(['message', 'genre', 'id', 'original'], axis = 1)
    return X,Y,Y.columns


def tokenize(text):
    
    '''
    input:
        text = Messages to tokenize
    output:
        clean_tokeni = tokenized messaged.
    '''
    
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    tokens = word_tokenize(text)
    stop_words = stopwords.words("english")
    stemmed = [PorterStemmer().stem(w) for w in words]
    
    return stemmed


def build_model():
    
    """
     Function: build a classifier
     Return: classification model
    """
    pipeline = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf',MultiOutputClassifier(RandomForestClassifier()))])
    parameters = { 'vect__max_df': (0.75, 1.0),
                'clf__estimator__n_estimators': [10, 25],
                'clf__estimator__min_samples_split': [2, 6]
              }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=10, n_jobs=4)
    
    return cv
    
def performance(model, X_test, Y_test):
    
    """
    Function: Evaluate the model
    Args:
    model: the classification model
    X_test: test messages
    Y_test: test target
    """

    y_predicted = model.predict(X_test)
    for i, col in enumerate(Y_test):
        print(col)
        print(classification_report(Y_test[col], y_predicted[:, i]))
    
def evaluate_model(model, X_test, Y_test, category_names):
    """
    Function: Evaluate the model
    Args:
    model: the classification model
    X_test: test messages
    Y_test: test target
    """
    y_predicted = model.predict(X_test)
    for i, col in enumerate(Y_test):
        print(col)
        print(classification_report(Y_test[col], y_predicted[:, i]))
   


def save_model(model, model_filepath):
    """ Saving model's best_estimator
    """
    pickle.dump(model, open(f'{model_filepath}', 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
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