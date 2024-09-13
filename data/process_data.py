import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Loads the message and category data from CSV files.

    Parameters:
    -----------
    messages_filepath : str
        Path to the CSV file containing the messages.
    categories_filepath : str
        Path to the CSV file containing the categories.
    
    Returns:
    --------
    df : DataFrame
        DataFrame containing the merged messages and categories.
    """
    # Load datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # Merge the datasets
    df = pd.merge(messages, categories, on='id')
    
    return df

def clean_data(df):
    """
    Cleans the dataframe by transforming the category columns into separate columns
    and converting values to binary (0 or 1). Drops rows where 'related' equals 2.

    Parameters:
    -----------
    df : DataFrame
        The merged DataFrame with messages and categories.
    
    Returns:
    --------
    df : DataFrame
        Cleaned DataFrame with separate category columns and no duplicates.
    """
    # Split the 'categories' column into separate columns
    categories = df['categories'].str.split(';', expand=True)
    
    # Use the first row to extract column names
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].str[-1]  # Take the last character
        categories[column] = categories[column].astype(int)  # Convert to integer

    # Drop rows where 'related' equals 2
    categories = categories[categories['related'] != 2]

    # Drop the original 'categories' column and concatenate the new 'categories' dataframe
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)

    # Remove duplicates
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    """
    Saves the cleaned dataframe into an SQLite database.

    Parameters:
    -----------
    df : DataFrame
        The cleaned DataFrame to be saved.
    database_filename : str
        The name of the SQLite database file.
    """
    # Create SQLite engine
    engine = create_engine(f'sqlite:///{database_filename}')
    
    # Save the dataframe to the database
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')



def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
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