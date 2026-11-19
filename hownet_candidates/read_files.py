'''
Zoey Chappell and Jonah Whitney
Based on https://github.com/AdvAttack/BU-SPO
'''
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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

'''
Updated function to read the IMDB dataset from a CSV file instead of individual text files.

Inputs: 
    - N/a
Outputs:
    - all_text: A list of all text reviews from the IMDB dataset.
    - all_labels: A numpy array of one-hot encoded labels 
        corresponding to the reviews.
'''
_split = None

def read_imdb_files(filetype):
    global _split
    if _split is None:
        df = pd.read_csv('IMDB_Dataset.csv')
        texts = df['review'].astype(str).str.replace('<br />', ' ', regex=False).values
        labels = (df['sentiment'] == 'positive').astype(int).values
        _split = train_test_split(texts, labels, test_size=0.5, random_state=42, stratify=labels)
    x_tr, x_te, y_tr, y_te = _split
    if filetype == 'train':
        return list(x_tr), np.eye(2, dtype='float32')[y_tr]
    return list(x_te), np.eye(2, dtype='float32')[y_te]

'''
Updated function. Splits the IMDB dataset into training and testing. 

Parameters: 
    - test_size: Default = 0.5.The proportion of the dataset to include in the test split.
    - random_state: Default = 42. Reproducible results
    
Returns:
    - train_texts: A list of training text reviews.
    - train_labels: A numpy array of one-hot encoded labels for the training reviews.
    - test_texts: A list of testing text reviews.
    - test_labels: A numpy array of one-hot encoded labels for the testing reviews.
'''
def split_imdb_files():
    print('Processing IMDB dataset')
    train_texts, train_labels = read_imdb_files('train')
    test_texts, test_labels = read_imdb_files('test')
    return train_texts, train_labels, test_texts, test_labels

if __name__ == '__main__':
    train_texts, train_labels, test_texts, test_labels = split_imdb_files()
    for n in range(5):
        print(f"Train text {n}: {train_texts[n]}")
        print(f"Train label {n}: {train_labels[n]}")
        print(f"Test text {n}: {test_texts[n]}")
        print(f"Test label {n}: {test_labels[n]}")