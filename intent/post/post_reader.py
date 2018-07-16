import pandas as pd


def read_post_info(path):
    df = pd.read_csv(filepath_or_buffer=path, sep=';', encoding='cp1251')
    columns = set(list(df.keys()))
    keep_columns = {'Id of post', 'text'}
    for col in columns-keep_columns:
        df.drop(col, axis=1, inplace=True)
    return df


def get_posts_col(df):
    return df['Id of post']


def get_text_col(df):
    return df['text']
