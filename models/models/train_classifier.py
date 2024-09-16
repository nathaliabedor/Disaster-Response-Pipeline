import sys
import pandas as pd
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report
import pickle

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
    """
    Loads the data from a SQLite database.

    Parameters:
    -----------
    database_filepath : str
        Path to the SQLite database file.

    Returns:
    --------
    X : DataFrame
        Features dataset (messages).
    Y : DataFrame
        Target dataset (categories).
    category_names : list
        List of category names for classification.
    """
    # Create engine to connect to the database

    engine = create_engine(f'sqlite:///{database_filepath}')
    
    # Load the data from the table
    df = pd.read_sql_table('DisasterResponse', engine)
    
    # Define features and target variables
    X = df['message']
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])
    category_names = Y.columns
    
    return X, Y, category_names


def tokenize(text):
    """
    Tokenizes and processes text data.

    Parameters:
    -----------
    text : str
        Input text to process.
    
    Returns:
    --------
    tokens : list
        List of processed and tokenized words.
    """
    # Normalize text: remove punctuation and convert to lowercase
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords and lemmatize words
    tokens = [WordNetLemmatizer().lemmatize(word) for word in tokens if word not in stopwords.words('english')]
    
    return tokens

def build_model():
    """
    Builds a machine learning model pipeline with GridSearchCV.

    Returns:
    --------
    cv : GridSearchCV
        A GridSearchCV object with the pipeline and parameters.
    """
    # Define the pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # Define parameters for grid search
    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2)],
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__min_samples_split': [2, 4]
    }

    # Set up GridSearchCV
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3, n_jobs=-1)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates the model's performance on the test data.

    Parameters:
    -----------
    model : GridSearchCV
        The trained machine learning model.
    X_test : DataFrame
        Test features (messages).
    Y_test : DataFrame
        Test target variables (categories).
    category_names : list
        List of category names.
    """
    # Predict on the test data
    Y_pred = model.predict(X_test)

    # Loop through each category and print the classification report
    for i, category in enumerate(category_names):
        print(f'Category: {category}')
        print(classification_report(Y_test.iloc[:, i], Y_pred[:, i]))


def save_model(model, model_filepath):
    """
    Saves the model to a pickle file.

    Parameters:
    -----------
    model : GridSearchCV
        The trained machine learning model.
    model_filepath : str
        The filepath where the model will be saved.
    """
    # Save the model to a pickle file
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)



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