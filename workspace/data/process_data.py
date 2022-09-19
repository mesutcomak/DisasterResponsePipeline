import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    
    '''
    args:   messages
            categories of the messages
    return  merged dataframe
    
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    return pd.merge(messages, categories, on="id")

def converting(input_cell):
    '''
    args: cell
    return cleaned and labeled cell with 0 or 1
    '''
    a=int(input_cell[-1])
    if a>=1:
        a=1
    else:
        a=0
    return a

def clean_data(df):
    """
    Clean dataframe by removing duplicates & converting categories
    
    Args:
    df: merged content of messages
       
    Returns:
    df: dataframe.  cleaned version of input dataframe
    """
    categories = df ['categories'].str.split (pat = ';', expand = True)
    row = categories.iloc[0]
    category_colnames = row.transform(lambda x: x[:-2]).tolist()
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str [-1]

        categories[column] = categories[column].apply(lambda x:converting(x))

        # convert column from string to numeric
        categories[column] = categories[column].astype('int')

    df.drop('categories',inplace=True, axis = 1)
    df = pd.concat([df, categories],axis = 1)
    print(df.duplicated().sum())
    df.drop_duplicates(inplace = True)
    print(df.duplicated().sum())
    print(df.head())
    return df


def save_data(df, database_filename):
    
    """Save processed data to a SQLite database
    
    Arguments:
        df -- Pandas Dataframe
        database_filename  -- Table name
    """
    
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('TableName', engine, index=False,if_exists='replace')
     


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