import pandas as pd
import string

def load_and_clean_data():

    df = pd.read_csv('headlines.csv')
    df.drop(df.tail(390000).index, inplace=True)
    df = df[['CATEGORY','TITLE']]
    df = df[pd.notnull(df['TITLE'])]
    print(df['TITLE'].head(5))
    df.columns = ['CATEGORY', 'TITLE']
    df.TITLE = df.TITLE.apply(lambda x: x.lower())
    df.TITLE = df.TITLE.apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    df.TITLE = df.TITLE.apply(lambda x: x.translate(str.maketrans('', '', '1234567890')))
    df['category_id'] = df['CATEGORY'].factorize()[0]


    return df
