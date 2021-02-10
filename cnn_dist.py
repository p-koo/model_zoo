from tensorflow import keras

def model(input_shape, num_labels, activation='relu', 
          units=[24, 32, 48, 64, 96], dropout=[0.1, 0.2, 0.3, 0.4, 0.5], 
          bn=[True, True, True, True, True], l2=None):

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
  inputs = keras.layers.Input(shape=(200,4))

  # layer 1
  nn = keras.layers.Conv1D(filters=units[0],
                           kernel_size=19,
                           strides=1,
                           activation=None,
                           use_bias=use_bias[0],
                           padding='same',
                           kernel_regularizer=l2, 
                           )(inputs)
  if bn[0]:
    nn = keras.layers.BatchNormalization()(nn)
  nn = keras.layers.Activation(activation)(nn)
  nn = keras.layers.Dropout(dropout[0])(nn)

  nn = keras.layers.Conv1D(filters=units[1],
                           kernel_size=7,
                           strides=1,
                           activation=None,
                           use_bias=use_bias[1],
                           padding='same',
                           kernel_regularizer=l2, 
                           )(nn)
  if bn[1]:
    nn = keras.layers.BatchNormalization()(nn)
  nn = keras.layers.Activation('relu')(nn)
  nn = keras.layers.Dropout(dropout[1])(nn)
  nn = keras.layers.MaxPool1D(pool_size=4)(nn)

  # layer 2
  nn = keras.layers.Conv1D(filters=units[2],
                           kernel_size=5,
                           strides=1,
                           activation=None,
                           use_bias=use_bias[2],
                           padding='same',
                           kernel_regularizer=l2, 
                           )(nn)
  if bn[2]:
    nn = keras.layers.BatchNormalization()(nn)
  nn = keras.layers.Activation('relu')(nn)
  nn = keras.layers.MaxPool1D(pool_size=4)(nn)
  nn = keras.layers.Dropout(dropout[2])(nn)

  # layer 3
  nn = keras.layers.Conv1D(filters=units[3],
                           kernel_size=5,
                           strides=1,
                           activation=None,
                           use_bias=use_bias[3],
                           padding='same',
                           kernel_regularizer=l2, 
                           )(nn)
  if bn[3]:
    nn = keras.layers.BatchNormalization()(nn)
  nn = keras.layers.Activation('relu')(nn)
  nn = keras.layers.MaxPool1D(pool_size=4)(nn)
  nn = keras.layers.Dropout(dropout[3])(nn)

  # layer 4 - Fully-connected 
  nn = keras.layers.Flatten()(nn)
  nn = keras.layers.Dense(units[4],
                          activation=None,
                          use_bias=use_bias[4],
                          kernel_regularizer=l2, 
                          )(nn)      
  if bn[4]:
    nn = keras.layers.BatchNormalization()(nn)
  nn = keras.layers.Activation('relu')(nn)
  nn = keras.layers.Dropout(dropout[4])(nn)

  # Output layer 
  logits = keras.layers.Dense(num_labels, activation='linear', use_bias=True)(nn)
  outputs = keras.layers.Activation('sigmoid')(logits)
      
  # compile model
  return keras.Model(inputs=inputs, outputs=outputs)
