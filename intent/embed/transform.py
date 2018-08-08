"""
Implementation inspired by: https://github.com/ideis/news-cleaning/blob/master/news-cleaning.ipynb
"""
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import numpy as np


tokenizer = RegexpTokenizer(r'\w+')


def clean_text(text, remove_stopwords=True):
    """
    Removes stop words from the given text
    :param text: string with the text
    :param remove_stopwords: flag to enable removal of stop words
    :return: set of tokens without any stop words
    """
    text = text.lower()
    tokens = tokenizer.tokenize(text)

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words('russian'))
        tokens = [w for w in tokens if w not in stops]

    return tokens


def create_average_vec(doc, vec_dim, ft_model):
    average = np.zeros((vec_dim,), dtype='float32')
    num_words = 0.
    for word in doc:
        if word in ft_model.wv.vocab:
            average = np.add(average, ft_model[word])
            num_words += 1.
    if num_words != 0.:
        average = np.divide(average, num_words)
    return average


def doc2vec(document, vec_size=300, ft_model=None):
    """
    Translates the given string to the embedding vector corresponding to it
    :param document: string with the text
    :param vec_size: size of the embedding vector
    :param ft_model: model containing all word embeddings
    :return: embedding vector
    """
    processed_text = clean_text(document)
    return create_average_vec(processed_text, vec_dim=vec_size, ft_model=ft_model)
