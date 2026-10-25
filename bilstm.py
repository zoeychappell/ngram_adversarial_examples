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

'''IMDB database'''
def load_database():
    return pd.read_csv("IMDB Dataset.csv")
'''
Parameters: 
    - df: The DataFrame containing the dataset.
    - dataset: The name of the dataset (e.g., 'imdb').
Returns: 
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
def bilstm(dataset):
    max_len = config.word_max_len[dataset]
    num_classes = config.num_classes[dataset]
    loss = config.loss[dataset]
    activation = config.activation[dataset]
    embedding_dims = config.bdLSTM_embedding_dims[dataset]
    num_words = config.num_words[dataset]

    print('Build word_bdlstm model...')
    model = Sequential()

    model.add(Embedding(num_words, embedding_dims, input_length=max_len))
    model.add(Bidirectional(LSTM(64)))  # 64 / LSTM-2:128 / LSTM-3: 32
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation=activation))

    # try using different optimizers and different optimizer configs
    model.compile('adam', loss, metrics=['accuracy'])

    return model
def main():
    dataset = 'imdb'
    df = load_database()
    x_train, y_train, x_test, y_test, vectorizer = prepare_data(df, dataset)

    model = bilstm(dataset)
    model.summary()

    model.fit(
        x_train, y_train,
        batch_size=config.bdLSTM_batch_size[dataset],
        epochs=config.bdLSTM_epochs[dataset],
        validation_split=0.1,
    )

    # Evaluate the model on the test set
    loss, acc = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {acc:.4f}")

    model.save("bilstm_imdb.keras")


if __name__ == '__main__':
    main()
