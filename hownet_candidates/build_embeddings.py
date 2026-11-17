'''
Zoey Chappell and Jonah Whitney
Based on Yang et al. 
Completed the necessary changes. 
'''
import numpy as np
# commented out
# from tf_keras.preprocessing.text import Tokenizer
# commented out
# from keras.preprocessing.sequence import pad_sequences
import pickle

import os
#import nltk
# commented out
# import re
# commented out
# from collections import Counter


import data_utils_utf8
import glove_utils
# Changed to the csv file path
IMDB_PATH = 'IMDB_Dataset.csv'
MAX_VOCAB_SIZE = 50000
GLOVE_PATH = 'hownet_candidates/glove.840B.300d/glove.840B.300d.txt'

if not os.path.exists('aux_files'):
	os.mkdir('aux_files')
imdb_dataset = data_utils_utf8.IMDBDataset(path=IMDB_PATH, max_vocab_size=MAX_VOCAB_SIZE)

# save the dataset
with open(('aux_files/dataset_%d.pkl' %(MAX_VOCAB_SIZE)), 'wb') as f:
    pickle.dump(imdb_dataset, f)

# create the glove embeddings matrix (used by the classification model)
glove_model = glove_utils.loadGloveModel(GLOVE_PATH)
glove_embeddings, _ = glove_utils.create_embeddings_matrix(glove_model, imdb_dataset.dict, imdb_dataset.full_dict)
# save the glove_embeddings matrix
np.save('aux_files/embeddings_glove_%d.npy' %(MAX_VOCAB_SIZE), glove_embeddings)




print('All done')
