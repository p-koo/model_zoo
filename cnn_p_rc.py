from tensorflow import keras
import sys
sys.path.append('..')
from tfomics.layers import RevCompConv1D#, RevCompMaxPool


def model(input_shape, num_labels, activation='relu', units=[32, 128, 512], pool_size=[25, 4], 
          dropout=[0.1, 0.1, 0.5], bn=[False, True, True], l2=None):
  
  L, A = input_shape

  # l2 regularization
  if l2 is not None:
    l2 = keras.regularizers.l2(l2)

  use_bias = []
  for status in bn:
    if status:
      use_bias.append(True)
    else:
      use_bias.append(False)

  # input layer
  inputs = keras.layers.Input(shape=(L,A))

  # layer 1 - convolution     
  fwd, rev = RevCompConv1D(filters=units[0], kernel_size=19, use_bias=use_bias[0], padding='same',
                           kernel_regularizer=l2, concat=False)(inputs)        

  if bn[0]:
    fwd = keras.layers.BatchNormalization()(fwd)
    rev = keras.layers.BatchNormalization()(rev)

  fwd = keras.layers.Activation(activation)(fwd)
  fwd = keras.layers.MaxPool1D(pool_size=pool_size[0])(fwd)
  fwd = keras.layers.Dropout(dropout[0])(fwd)

  rev = keras.layers.Activation(activation)(rev)
  rev = keras.layers.MaxPool1D(pool_size=pool_size)(rev)
  rev = keras.layers.Dropout(dropout[0])(rev)


  # layer 2 - convolution
  fwd, rev = RevCompConv1D(filters=units[1], kernel_size=7, use_bias=use_bias[1], padding='same',
                           kernel_regularizer=l2, concat=False)(fwd, rev)       
  if bn[1]:
    fwd = keras.layers.BatchNormalization()(fwd)
    rev = keras.layers.BatchNormalization()(rev)
  fwd = keras.layers.Activation('relu')(fwd)
  rev = keras.layers.Activation('relu')(rev)

  # nn = RevCompMaxPool()(fwd, rev, reverse=False)
  nn = keras.layers.MaxPool1D(pool_size=pool_size[1])(nn)
  nn = keras.layers.Dropout(dropout[1])(nn)

  # layer 3 - Fully-connected 
  nn = keras.layers.Flatten()(nn)
  nn = keras.layers.Dense(units[2],
                          activation=None,
                          use_bias=use_bias[2],
                          kernel_regularizer=l2, 
                          )(nn)      
  if bn[2]:
    nn = keras.layers.BatchNormalization()(nn)
  nn = keras.layers.Activation('relu')(nn)
  nn = keras.layers.Dropout(dropout[2])(nn)

  # Output layer
  logits = keras.layers.Dense(num_labels, activation='linear', use_bias=True)(nn)
  outputs = keras.layers.Activation('sigmoid')(logits)

  # create keras model
  return keras.Model(inputs=inputs, outputs=outputs)

