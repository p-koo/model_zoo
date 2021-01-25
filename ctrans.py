import tensorflow as tf
from tensorflow import keras
import numpy as np


def model(input_shape, num_labels, dims, num_heads=12, num_layers=4, pool_size=5, 
          num_units=1024, activation='relu', bn=False):

  # l2 regularization
  l2 = keras.regularizers.l2(1e-6)

  # input layer
  inputs = keras.layers.Input(shape=input_shape)

  # layer 1 - convolution
  nn = keras.layers.Conv1D(filters=dims, kernel_size=19, use_bias=True, padding='same',
                           kernel_regularizer=l2)(inputs)        
  if bn:
    nn = keras.layers.BatchNormalization()(nn)
  nn = keras.layers.Activation('exponential')(nn)
  nn = keras.layers.MaxPool1D(pool_size=pool_size)(nn)
  nn = keras.layers.Dropout(0.1)(nn)

  forward_layer = keras.layers.LSTM(dims//2, return_sequences=True)
  backward_layer = keras.layers.LSTM(dims//2, activation='relu', return_sequences=True, go_backwards=True)
  nn2 = keras.layers.Bidirectional(forward_layer, backward_layer=backward_layer)(nn)
  nn = keras.layers.Dropout(0.1)(nn2)
  #nn = keras.layers.Add()([nn, nn2])
  nn = keras.layers.LayerNormalization(epsilon=1e-6)(nn)

  for i in range(num_layers):
    nn2,_ = MultiHeadAttention(d_model=dims, num_heads=num_heads)(nn, nn, nn)
    nn2 = keras.layers.Dropout(0.1)(nn2)
    nn = tf.keras.layers.Add()([nn, nn2])
    nn = keras.layers.LayerNormalization(epsilon=1e-6)(nn)

    nn2 = keras.layers.Dense(16, activation='relu')(nn)
    nn2 = keras.layers.Dense(dims)(nn2)
    nn2 = keras.layers.Dropout(0.1)(nn2)
    nn = tf.keras.layers.Add()([nn, nn2])
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
  model = keras.Model(inputs=inputs, outputs=outputs)


  # set up optimizer and metrics
  auroc = keras.metrics.AUC(curve='ROC', name='auroc')
  aupr = keras.metrics.AUC(curve='PR', name='aupr')
  optimizer = keras.optimizers.Adam(learning_rate=0.0003)
  loss = keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.0)

  model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=[auroc, aupr])

  return model


@tf.function
def scaled_dot_product_attention(q, k, v):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead) 
  but it must be broadcastable for addition.
  
  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.
    
  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
  
  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model
    
    assert d_model % self.num_heads == 0
    
    self.depth = d_model // self.num_heads
    
    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)
    
    self.dense = tf.keras.layers.Dense(d_model)
        
  def split_heads(self, x, batch_size, seq_len):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, seq_len, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])
    
  def call(self, v, k, q):
    batch_size = tf.shape(q)[0]
    seq_len = tf.constant(q.shape[1])

    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)
    
    q = self.split_heads(q, batch_size, seq_len)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size, seq_len)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size, seq_len)  # (batch_size, num_heads, seq_len_v, depth)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v)
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention, 
                                  (batch_size, seq_len, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

    return output, attention_weights
