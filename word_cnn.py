import pandas as pd
import numpy as np
import sklearn as sk
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, pyll
from google.colab import drive


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, Activation, Flatten, Dense
from keras.layers import Conv1D, MaxPooling1D, Dropout
from keras.models import Model

from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import numpy as np
import time

from keras_preprocessing.text import Tokenizer
from keras.preprocessing.sequence  import pad_sequences

import tensorflow as tf
from keras.layers import *
from keras.models import Model
from keras.backend import clear_session
from hyperopt import Trials, STATUS_OK, tpe
from keras.callbacks import EarlyStopping, History
from keras.backend import clear_session

word_cnn_space = {
    'conv_1_filter_number' : hp.choice('conv_1_filter_number', [300, 600, 900, 1800]),
    'conv_2_filter_number' : hp.choice('conv_2_filter_number', [300, 600, 900, 1800]),
    'conv_3_filter_number' : hp.choice('conv_3_filter_number', [300, 600, 900, 1800]),
    'dense_size' : hp.choice('dense_size', [100, 300, 600, 1000]),
    'extra_dense' : hp.choice('extra_dense', [True, False]),
    'extra_dense_size' : hp.choice('extra_dense_size', [100, 300, 600]),
    'dropout_rate' : hp.uniform('dropout_rate', 0.0, 0.5),
    'optimizer': hp.choice('optimizer',['Adam', 'Adagrad', 'Adadelta', 'Nadam'])
}




def data_prep(sentences_train, sentences_test, maxlen=50, max_features=20000):

  tokenizer = Tokenizer(num_words = max_features)
  tokenizer.fit_on_texts(sentences_train)
  word_index = tokenizer.word_index

  X_train = tokenizer.texts_to_sequences(sentences_train)
  X_test = tokenizer.texts_to_sequences(sentences_test)

  vocab_size = len(tokenizer.word_index) +1
  X_train = pad_sequences (X_train, padding = 'post', maxlen=maxlen)
  X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
  
  return X_train, X_test , maxlen, max_features 

def return_model(params, input_size, max_features):
  embedding_size=300
  #input layer
  inputs = Input(shape=(input_size,), dtype='int32')
  #embedding layer
  #embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=input_size)
  embedding_layer = Embedding(input_dim=max_features, output_dim=embedding_size, input_length=input_size)
  #reshape
  reshape = Reshape((input_size, embedding_size))
  reshape_out = Flatten()
  #Filters
  conv_1 = Conv1D(params['conv_1_filter_number'], 3, activation='relu')
  conv_2 = Conv1D(params['conv_2_filter_number'], 5, activation='relu')
  conv_3 = Conv1D(params['conv_3_filter_number'], 7, activation='relu')

  #maxpools
  max_1 = MaxPool1D(input_size - 3 + 1)
  max_2 = MaxPool1D(input_size - 5 + 1)
  max_3 = MaxPool1D(input_size - 7 + 1)

  #Fully connected Layer[
  dense_1 = Dense(units=params['dense_size'], activation='sigmoid')
  dense = Dense(units=18, activation='softmax')

  #forward_pass
  output_embedding = embedding_layer(inputs)
  reshape_out = reshape(output_embedding)
  output_c1 = conv_1(reshape_out)
  output_c2 = conv_2(reshape_out)
  output_c3 = conv_3(reshape_out)

  output_max_1 = max_1(output_c1)
  output_max_2 = max_2(output_c2)
  output_max_3 = max_3(output_c3)

  concatenated_tensor = Concatenate()([output_max_1,output_max_2, output_max_3])
  flat = Flatten()(concatenated_tensor)

  dropout_out = Dropout(params['dropout_rate'])(flat)
  #possible extra layer
  if(params['extra_dense']):
    dropout_out= Dense(
        units=params['extra_dense_size'],
        activation='sigmoid')(dropout_out)


  output = dense(dense_1(dropout_out))

  model= Model(inputs=inputs, outputs=output)
  return model       