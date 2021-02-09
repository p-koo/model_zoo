from tensorflow import keras
import numpy as np
import sys

sys.path.append('..')
from tfomics.layers import MultiHeadAttention, RevCompConv1D


def model(input_shape, num_labels, filters=32, dims=64, num_heads=12, num_layers=4, 
          pool_size=20, num_units=512, activation='relu', bn=False, rc=True):

  # l2 regularization
  l2 = keras.regularizers.l2(1e-6)

  if bn:
    use_bias = False
  else:
    use_bias = True

  # input layer
  inputs = keras.layers.Input(shape=input_shape)

  # layer 1 - convolution
  if rc:  
    nn = keras.layers.RevCompConv1D(filters=filters, kernel_size=19, use_bias=use_bias, padding='same',
                                    kernel_regularizer=l2, concat=True)(inputs)      
  else:
    nn = keras.layers.Conv1D(filters=filters, kernel_size=19, use_bias=use_bias, padding='same',
                             kernel_regularizer=l2)(inputs)            
  if bn:
    nn = keras.layers.BatchNormalization()(nn)
  nn = keras.layers.Activation(activation)(nn)
  nn = keras.layers.MaxPool1D(pool_size=pool_size)(nn)
  nn = keras.layers.Dropout(0.1)(nn)

  forward_layer = keras.layers.LSTM(dims//2, return_sequences=True)
  backward_layer = keras.layers.LSTM(dims//2, activation='relu', return_sequences=True, go_backwards=True)
  nn = keras.layers.Bidirectional(forward_layer, backward_layer=backward_layer)(nn)
  nn = keras.layers.Dropout(0.1)(nn)

  #nn = keras.layers.Add()([nn, nn2])
  nn = keras.layers.LayerNormalization(epsilon=1e-6)(nn)
  nn2,_ = MultiHeadAttention(d_model=dims, num_heads=num_heads)(nn, nn, nn)
  nn2 = keras.layers.Dropout(0.1)(nn2)
  nn = keras.layers.Add()([nn, nn2])
  nn = keras.layers.LayerNormalization(epsilon=1e-6)(nn)

  # layer 3 - Fully-connected 
  nn = keras.layers.Flatten()(nn)
  nn = keras.layers.Dense(num_units, activation=None, use_bias=False)(nn)      
  nn = keras.layers.BatchNormalization()(nn)
  nn = keras.layers.Activation('relu')(nn)
  nn = keras.layers.Dropout(0.5)(nn)

  #nn = keras.layers.Dense(num_units, activation=None, use_bias=False)(nn)      
  #nn = keras.layers.BatchNormalization()(nn)
  #nn = keras.layers.Activation('relu')(nn)
  #nn = keras.layers.Dropout(0.5)(nn)
  
  # Output layer
  logits = keras.layers.Dense(num_labels, activation='linear', use_bias=True)(nn)
  outputs = keras.layers.Activation('sigmoid')(logits)

  # create keras model
  return keras.Model(inputs=inputs, outputs=outputs)


