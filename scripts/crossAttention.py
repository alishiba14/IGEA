from keras.layers import *
import tensorflow as tf
from keras import backend as K

class CrossAttention(tf.keras.Model):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(CrossAttention, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.Wq = self.add_weight(name='Wq', 
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.Wk = self.add_weight(name='Wk', 
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.Wv = self.add_weight(name='Wv', 
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(CrossAttention, self).build(input_shape)
        
    def call(self, inputs):
        q, k = inputs
        
        # Compute the attention weights and context vectors
        q_proj = K.dot(q, self.Wq)  # project the query
        k_proj = K.dot(k, self.Wk)  # project the key
        v_proj = K.dot(k, self.Wv)  # project the value
        
        attn = Dot(axes=-1)([q_proj, k_proj])
        attn_weights = Softmax()(attn)
        
        context = Dot(axes=1)([attn_weights, v_proj])
        
        # Concatenate the query and context vectors
        output = Concatenate()([q, context])
        return output
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][-1] + self.output_dim)
