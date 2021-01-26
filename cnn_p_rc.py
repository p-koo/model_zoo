from tensorflow import keras

sys.path.append('..')
from tfomics.layers import RevCompConv1D, RevCompMaxPool


def model(input_shape, num_labels, activation='relu', pool_size=25, units=[32, 128, 512], 
          dropout=[0.1, 0.1, 0.5], l2=None, bn=False):
  
  L, A = input_shape
  pool_size2 = L // pool_size

  # l2 regularization
  if l2 is not None:
    l2 = keras.regularizers.l2(l2)

  # input layer
  inputs = keras.layers.Input(shape=(L,A))

  # layer 1 - convolution
  if bn:
    use_bias = False
  else:
    use_bias = True
     
  nn, nn2 = RevCompConv1D(filters=dims, kernel_size=19, use_bias=use_bias, padding='same',
                           kernel_regularizer=l2, concat=False)(inputs)        

  if bn:
    nn = keras.layers.BatchNormalization()(nn)
    nn2 = keras.layers.BatchNormalization()(nn2)
  nn = keras.layers.Activation(activation)(nn)
  nn = keras.layers.MaxPool1D(pool_size=pool_size)(nn)
  nn = keras.layers.Dropout(dropout[0])(nn)

  nn2 = keras.layers.Activation(activation)(nn2)
  nn2 = keras.layers.MaxPool1D(pool_size=pool_size)(nn2)
  nn2 = keras.layers.Dropout(dropout[0])(nn2)


  # layer 2 - convolution
  nn, nn2 = RevCompConv1D(filters=units[1], kernel_size=7, use_bias=False, padding='same',
                           kernel_regularizer=l2, concat=False)(nn, nn2)       
  nn = keras.layers.BatchNormalization()(nn)
  nn = keras.layers.Activation('relu')(nn)
  nn2 = keras.layers.BatchNormalization()(nn2)
  nn2 = keras.layers.Activation('relu')(nn2)
  nn = RevCompMaxPool()(nn, nn2)
  nn = keras.layers.MaxPool1D(pool_size=pool_size2)(nn)
  nn = keras.layers.Dropout(dropout[1])(nn)

  # layer 3 - Fully-connected 
  nn = keras.layers.Flatten()(nn)
  nn = keras.layers.Dense(units[2],
                          activation=None,
                          use_bias=False,
                          kernel_regularizer=l2, 
                          )(nn)      
  nn = keras.layers.BatchNormalization()(nn)
  nn = keras.layers.Activation('relu')(nn)
  nn = keras.layers.Dropout(dropout[2])(nn)

  # Output layer
  logits = keras.layers.Dense(num_labels, activation='linear', use_bias=True)(nn)
  outputs = keras.layers.Activation('sigmoid')(logits)

  # create keras model
  model = keras.Model(inputs=inputs, outputs=outputs)


  return model

