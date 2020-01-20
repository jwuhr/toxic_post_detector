import sys, os, re, csv, codecs
import numpy as np
import pandas as pd

from keras import initializers, regularizers, constraints, optimizers, layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, CuDNNLSTM, Embedding, SpatialDropout1D, Dropout, Activation
from keras.layers import Bidirectional, Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, concatenate
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

TRAIN_DATA = 'train.csv'         #Training dataset
TEST_DATA = 'test.csv'           #Testing dataset
SUB = 'sample_submission.csv'    #Submission file for Kaggle
WORD_VEC = 'glove.840B.300d.txt' #Pre-trained Stanford Global Vectors for Word Representation (GloVe)

embed_size = 300  #300d pretrained vectors
features = 150000 #Num unique words to add to encode
maxlen = 200  #Max words in comment

#Read each CSV file
train = pd.read_csv(TRAIN_DATA)
test = pd.read_csv(TEST_DATA)
submission = pd.read_csv(SUB) #File to sumbit

#Data preprocessing
list_train = train["comment_text"].fillna("_na_").values #Non-tokenized training data
classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[classes].values #Target data
list_test = test["comment_text"].fillna("_na_").values #Non-tokenized testing data

#Tokenize training and testing data using Keras
tok = Tokenizer(num_words=features)
tok.fit_on_texts(list(list_train))

#Final tokenized training and testing data
tokenized_train = tok.texts_to_sequences(list_train)
tokenized_test = tok.texts_to_sequences(list_test)

#Pad vectors with 0s for sentences shorter than maxlen
train_x = pad_sequences(tokenized_train, maxlen=maxlen) #Input training data
test_x = pad_sequences(tokenized_test, maxlen=maxlen) #Input testing data

#Read word vectors into dictionary
def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')

embeddings_index = dict(get_coefs(*o.strip().split(" ")) for o in open(WORD_VEC))

#Embedding matrix
word_index = tok.word_index
total_words = min(features, len(word_index))

#Creating empty matric for embedding
embedding_matrix = np.zeros((total_words, embed_size))

#Filling in embedding matrix with vectors
for word, i in word_index.items():
    if i >= features:
        continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

#Layers of convoluted neural network
inp = Input(shape=(maxlen,))

#Bi-directional long short-term memory, and convolution layer
x = Embedding(features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
x = SpatialDropout1D(0.2)(x)
x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
x = Conv1D(64, kernel_size=3, padding="valid", kernel_initializer="glorot_uniform", activation="relu")(x)

#Pooling layer
avg_pool = GlobalAveragePooling1D()(x)
max_pool = GlobalMaxPooling1D()(x)

#Classifying layers
x = concatenate([avg_pool, max_pool])
x = Dense(128, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(6, activation="sigmoid")(x)

#Building model
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
highest_acc_model = 'models/model.h5'
#Saving model after every epoch, 5 total
checkpoint = ModelCheckpoint(highest_acc_model, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')

#Fitting model, only 5 epochs for the sake of time
model.fit(train_x, y, batch_size=512, epochs=5, callbacks=[early_stop, checkpoint], validation_split=0.1)

#Load new model
model = load_model(highest_acc_model)
final = model.predict(test_x, batch_size=512, verbose=1)
submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = final

#Write final submission file
submission.to_csv('final/submission.csv', index=False)