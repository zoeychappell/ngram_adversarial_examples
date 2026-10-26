'''
Zoey Chappell and Jonah Whitney
Based on https://github.com/AdvAttack/BU-SPO/blob/main/neural_networks.py

'''
# Used to gather the IMDB dataset
import kagglehub
from kagglehub import KaggleDatasetAdapter
# Used by Yang et al. to create the Bi-directional LSTM
from config import config
from keras.layers import Embedding, LSTM, Bidirectional, TextVectorization
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import pandas as pd
from sklearn.model_selection import train_test_split

"""
    Note from Zoey: Current accuracy is 0.8567
"""

'''
Grab the IMDB database

Parameters: 
    - None
Returns: 
    - A DataFrame containing the IMDB dataset.
'''
def load_database():
    return pd.read_csv("IMDB Dataset.csv")

'''
Parameters: 
    - df: The DataFrame containing the dataset.
    - dataset: The name of the dataset (e.g., 'imdb').
Returns: 
    - x_train: Training data features.
    - y_train: Training data labels.
    - x_test: Testing data features.
    - y_test: Testing data labels.
    - vectorizer: The TextVectorization instance used for preprocessing.
'''
def prepare_data(df, dataset):
    # Grab the configuration details from config.py
    # Based on Yang et al.
    max_len = config.word_max_len[dataset]
    num_classes = config.num_classes[dataset]
    num_words = config.num_words[dataset]

    # Clean text
    texts = df["review"].astype(str).str.replace("<br />", " ", regex=False).values

    # Build labels
    # astype makes True/False into 1/0, so positive = 1
    labels = (df["sentiment"] == "positive").astype(int).values

    # Split into test and train
    # Split = 50% for training and testing
    x_train_txt, x_test_txt, y_train, y_test = train_test_split(
        texts, labels, test_size=0.5, random_state=42, stratify=labels
    )

    # Fit the vocabulary on training text only 
    vectorizer = TextVectorization(max_tokens=num_words, output_sequence_length=max_len)
    vectorizer.adapt(x_train_txt)

    # Vectorize
    x_train = vectorizer(x_train_txt).numpy()
    x_test = vectorizer(x_test_txt).numpy()

    # One hot encode labels
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)

    # Return
    return x_train, y_train, x_test, y_test, vectorizer

'''
Define the Bidirectional LSTM model.
Parameters: 
    - dataset: The name of the dataset (e.g., 'imdb').
Returns:
    - A compiled Keras Sequential model with a Bidirectional LSTM architecture.
'''
def bilstm(dataset):
    '''
    Note from Zoey: Yang et al. 
     Bidirectional LSTM (Bi-LSTM) consists of a
128-dimension word embedding layer, a bidirectional layer
that wraps 64 LSTM units, a dropout of 0.3, and a fully connected layer for classification
    '''
    # Grab the configuration details from config.py. This was provided by Yang et al. 
    # max_len = config.word_max_len[dataset] # input_length is deprecated in Keras Embedding
    num_classes = config.num_classes[dataset]
    loss = config.loss[dataset]
    activation = config.activation[dataset]
    embedding_dims = config.bdLSTM_embedding_dims[dataset] # = 128
    num_words = config.num_words[dataset]

    # Define Hyperparameters based on Yang et al. 
    lstm_units = 64
    dropout = 0.3 

    print('Build word_bdlstm model...')
    model = Sequential()

    model.add(Embedding(num_words, embedding_dims))
    model.add(Bidirectional(LSTM(lstm_units)))  # 64 / LSTM-2:128 / LSTM-3: 32
    model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation=activation))

    # try using different optimizers and different optimizer configs
    model.compile('adam', loss, metrics=['accuracy'])

    return model


def main():
    # 1. Set the database to imdb. 
    dataset = 'imdb'
    # 2. Load the IMDB dataset into a DataFrame.
    df = load_database()
    # 3. Prepare the data for the model.
    x_train, y_train, x_test, y_test, vectorizer = prepare_data(df, dataset)
    # 4. Build the Bidirectional LSTM model.
    model = bilstm(dataset)
    # 5. Print the model summary.
    model.summary()

    # 6. Train the model.
    print("\033[1;32mTraining the Bidirectional LSTM model...\033[0m")
    model.fit(
        x_train, y_train,
        batch_size=config.bdLSTM_batch_size[dataset], # = 32
        epochs=config.bdLSTM_epochs[dataset], # = 6
        validation_split=0.1,
    )

    # 7. Evaluate the model on the test set
    print("\033[1;32mEvaluating the Bidirectional LSTM model on the test set...\033[0m")
    loss, acc = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {acc:.4f}")

    # 8. Save the trained model
    model.save("bilstm_imdb.keras")


if __name__ == '__main__':
    main()
