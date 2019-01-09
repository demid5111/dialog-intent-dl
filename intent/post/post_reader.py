import pandas as pd
import os

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


def substitute_cyrillic():
    dialogs_list = []
    data_path = "../feature_and_vector_seq"
    file_list = os.listdir(data_path)
    for file_name in file_list[0:]:
        file_and_path = os.path.join(data_path, file_name)
        try:
            dialogs_list.append(pd.read_hdf(file_and_path)['Intent analysis'].values)
        except Exception as e:
            print('file_and_path',file_and_path)
            ss=['-Шарова', '-шарова',
                '-Набокова', '-Ким',
                '-Плотникова', '-Тимофеева']
            su=['-Sharova', '-Sharova',
                '-Nabokova', '-Kim',
                '-Plotnikova', '-Timofeeva']
            for i, si in enumerate(ss):
                pos = file_and_path.find(si)
                if pos > -1:
                    new_file_and_path = file_and_path[:pos]+su[i]+file_and_path[pos+len(si):]
                    os.rename(file_and_path, new_file_and_path)
    #                 print('new_file_and_path',new_file_and_path)

    len(dialogs_list), len(file_list)