'''
Zoey Chappell and Jonah Whitney
Based on https://github.com/AdvAttack/BU-SPO/blob/main/bigram.py
NOTE: this is currently directly copied from the link. Will edit later. 

'''

import os
import re
import pandas as pd
from config import config
import numpy as np
from sklearn.model_selection import train_test_split
# Was keras.utils.np_utils
from keras.utils import to_categorical

'''
This function, grabbed from Yang et al.
is used to remove HTML tags from the text data. 
It uses a regular expression to identify and 
remove any content within angle brackets, 
which are typically used for HTML tags.

Parameters: 
    - text: (STR) Input string to strip HTML tags.
Returns: 
    - (STR) The input string with HTML tags removed.
'''
def rm_tags(text):
    re_tag = re.compile(r'<[^>]+>')
    return re_tag.sub('', text)

'''original function'''
def read_imdb_files(filetype):
    """
    filetype: 'train' or 'test'
    """

    # [0,1] means positive，[1,0] means negative
    # One hot labels
    all_labels = []
    for _ in range(12500):
        all_labels.append([0, 1])
    for _ in range(12500):
        all_labels.append([1, 0])

    all_texts = []
    file_list = []
    path = r'./data_set/aclImdb/'
    pos_path = path + filetype + '/pos/'
    for file in os.listdir(pos_path):
        file_list.append(pos_path + file)
    neg_path = path + filetype + '/neg/'
    for file in os.listdir(neg_path):
        file_list.append(neg_path + file)
    for file_name in file_list:
        with open(file_name, 'r', encoding='utf-8') as f:
            all_texts.append(rm_tags(" ".join(f.readlines())))
    return all_texts, all_labels
'''
Updated function to read the IMDB dataset from a CSV file instead of individual text files.

Inputs: 
    - N/a
Outputs:
    - all_text: A list of all text reviews from the IMDB dataset.
    - all_labels: A numpy array of one-hot encoded labels 
        corresponding to the reviews.
'''
def updated_read_imdb_files():
    path = 'IMDB_Dataset.csv'
    # Use pandas to read the CSV file
    df = pd.read_csv(path)
    all_text = []
    # Grab all the text reviews
    for text in df['review']:
        all_text.append(rm_tags(text))

    # Replicates the one hot encoding pattern from the original
    labels_map = {'positive': [0, 1], 'negative': [1, 0]}

    # Grab all the labels: 
    all_labels = np.array([labels_map[sentiment] for sentiment in df['sentiment']], dtype='float32')

    return all_text, all_labels


def split_imdb_files(test_size = 0.5, random_state = 42):
    print('Processing IMDB dataset')

    texts, labels = updated_read_imdb_files()
    # Split into test and train
    # test_size = 0.5 mirrors original split
    # Random state to reproduce results
    # stratify=labels.argmax(axis=1) to keep pos/neg balanced
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=labels.argmax(axis=1)
    )

    return train_texts, train_labels, test_texts, test_labels

if __name__ == '__main__':
    train_texts, train_labels, test_texts, test_labels = split_imdb_files()
    for n in range(5):
        print(f"Train text {n}: {train_texts[n]}")
        print(f"Train label {n}: {train_labels[n]}")
        print(f"Test text {n}: {test_texts[n]}")
        print(f"Test label {n}: {test_labels[n]}")