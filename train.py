import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import numpy as np
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.model_selection import train_test_split
import json
import os

BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 3
EMBEDDING_DIM = 32
TRAIN_X = 'vectorized_data.npy'
TRAIN_Y = 'labels.npy'
WORD_2_VEC = 'word_2_vec.json'
NUM_OUTPUT = 3

if __name__ == '__main__':
    if not os.path.exists(TRAIN_X) or not os.path.exists(TRAIN_Y) or not os.path.exists(WORD_2_VEC):
        print('Data is not preprocessed yet. Please run preprocess.py before training the model.')
        exit()
    
    X = np.load(TRAIN_X)
    y = np.load(TRAIN_Y)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)

    word_2_vec = {}
    with open(WORD_2_VEC, 'r') as f:
        word_2_vec = json.load(f)

    max_int_index = max(word_2_vec.values()) + 1

    mirrored_strategy = tf.distribute.MirroredStrategy()

    # with mirrored_strategy.scope():
    model = Sequential()
    model.add(Embedding(input_dim=max_int_index, output_dim=EMBEDDING_DIM, input_length=X.shape[-1]))
    model.add(LSTM(100))
    model.add(Dense(NUM_OUTPUT, activation='softmax'))

    print(model.summary())

    model.compile(optimizer='adam', loss=CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val, y_val))
    model.save('./model')

