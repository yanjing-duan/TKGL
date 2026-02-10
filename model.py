import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Layer, Dense, Dropout, LayerNormalization, ReLU

import math
import time
import numpy as np
import matplotlib.pyplot as plt



def gelu(x):
    return 0.5 * x * (1.0 + tf.math.erf(x / tf.sqrt(2.)))


def scaled_dot_product_attention(q, k, v, mask,adjoin_matrix):
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

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    if adjoin_matrix is not None:
        scaled_attention_logits += adjoin_matrix

        # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
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

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask,adjoin_matrix):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask,adjoin_matrix)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation=gelu),  # (batch_size, seq_len, dff)tf.keras.layers.LeakyReLU(0.01)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask,adjoin_matrix):
        attn_output, attention_weights = self.mha(x, x, x, mask,adjoin_matrix)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2,attention_weights


class Encoder(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        # self.pos_encoding = positional_encoding(maximum_position_encoding,
        #                                         self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask,adjoin_matrix):
        seq_len = tf.shape(x)[1]
        adjoin_matrix = adjoin_matrix[:,tf.newaxis,:,:]
        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x,attention_weights = self.enc_layers[i](x, training, mask,adjoin_matrix)
        return x  # (batch_size, input_seq_len, d_model)

class Encoder_test(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Encoder_test, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        # self.pos_encoding = positional_encoding(maximum_position_encoding,
        #                                         self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask,adjoin_matrix):
        seq_len = tf.shape(x)[1]
        adjoin_matrix = adjoin_matrix[:,tf.newaxis,:,:]
        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)
        attention_weights_list = []
        xs = []

        for i in range(self.num_layers):
            x,attention_weights = self.enc_layers[i](x, training, mask,adjoin_matrix)
            attention_weights_list.append(attention_weights)
            xs.append(x)

        return x,attention_weights_list,xs

class BertModel_test(tf.keras.Model):
    def __init__(self,num_layers = 6,d_model = 256,dff = 512,num_heads = 8,vocab_size = 17,dropout_rate = 0.1):
        super(BertModel_test, self).__init__()
        self.encoder = Encoder_test(num_layers=num_layers,d_model=d_model,
                        num_heads=num_heads,dff=dff,input_vocab_size=vocab_size,maximum_position_encoding=200,rate=dropout_rate)
        self.fc1 = tf.keras.layers.Dense(d_model, activation=gelu)
        self.layernorm = tf.keras.layers.LayerNormalization(-1)
        self.fc2 = tf.keras.layers.Dense(vocab_size)
    def call(self,x,adjoin_matrix,mask,training=False):
        x,att,xs = self.encoder(x,training=training,mask=mask,adjoin_matrix=adjoin_matrix)
        x = self.fc1(x)
        x = self.layernorm(x)
        x = self.fc2(x)
        return x,att,xs


class BertModel(tf.keras.Model):
    def __init__(self,num_layers = 6,d_model = 256,dff = 512,num_heads = 8,vocab_size = 17,dropout_rate = 0.1):
        super(BertModel, self).__init__()
        self.encoder = Encoder(num_layers=num_layers,d_model=d_model,
                        num_heads=num_heads,dff=dff,input_vocab_size=vocab_size,maximum_position_encoding=200,rate=dropout_rate)
        self.fc1 = tf.keras.layers.Dense(d_model, activation=gelu)
        self.layernorm = tf.keras.layers.LayerNormalization(-1)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

    def call(self,x,adjoin_matrix,mask,training=False):
        x = self.encoder(x,training=training,mask=mask,adjoin_matrix=adjoin_matrix)
        x = self.fc1(x)
        x = self.layernorm(x)
        x = self.fc2(x)
        return x


class PredictModel(tf.keras.Model):
    def __init__(self,num_layers = 6,d_model = 256,dff = 512,num_heads = 8,vocab_size =17,dropout_rate = 0.1,dense_dropout=0.1):
        super(PredictModel, self).__init__()
        self.encoder = Encoder(num_layers=num_layers,d_model=d_model,
                        num_heads=num_heads,dff=dff,input_vocab_size=vocab_size,maximum_position_encoding=200,rate=dropout_rate)

        self.fc1 = tf.keras.layers.Dense(256,activation=tf.keras.layers.LeakyReLU(0.1))
        self.dropout = tf.keras.layers.Dropout(dense_dropout)
        self.fc2 = tf.keras.layers.Dense(1)

    def call(self,x,adjoin_matrix,mask,training=False):
        x = self.encoder(x,training=training,mask=mask,adjoin_matrix=adjoin_matrix)
        x = x[:,0,:]
        x = self.fc1(x)
        x = self.dropout(x,training=training)
        x = self.fc2(x)
        return x


class Global_Attention(Layer):
    def __init__(self):
        super(Global_Attention, self).__init__()

    def call(self, x):
        attention_weights = tf.nn.softmax(x, axis=-1)
        aggregated_output = x * attention_weights
        return aggregated_output


class WeightFusion(Layer):
    def __init__(self, feat_views, feat_dim, bias: bool = True, **kwargs) -> None:
        super(WeightFusion, self).__init__(**kwargs)
        self.feat_views = feat_views
        self.feat_dim = feat_dim
        self.bias = bias
        
        self.weight = self.add_weight(
            name="weight",
            shape=(1, 1, feat_views),
            initializer=tf.keras.initializers.VarianceScaling(
                scale=1./3.,
                mode='fan_in',
                distribution='uniform'
            ),
            trainable=True
        )
        
        if self.bias:
            fan_in = feat_views
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            self.bias_var = self.add_weight(
                name="bias",
                shape=(feat_dim,),
                initializer=tf.keras.initializers.RandomUniform(minval=-bound, maxval=bound),
                trainable=True
            )
        else:
            self.bias_var = None

    def call(self, inputs) -> tf.Tensor:
        x = tf.stack(inputs, axis=-1)
        weighted = x * self.weight
        result = tf.reduce_sum(weighted, axis=-1)
        
        if self.bias_var is not None:
            result += self.bias_var
        
        return result


class PredictModelFusionDA(tf.keras.Model):
    def __init__(self, num_layers=6, d_model=256, dff=512, num_heads=8, vocab_size=17,
                 dropout_rate=0.1, dense_dropout=0.1, KE_dim=None, SA_dim=None, hidden_dim=None,
                 num_hidden_layers=None, feat_views=3):
        super(PredictModelFusionDA, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff, input_vocab_size=vocab_size,
                               maximum_position_encoding=200, rate=dropout_rate)

        self.KE_attention = Dense(KE_dim)
        self.KE_extra_fc = tf.keras.Sequential([
            Dense(hidden_dim),
            ReLU(),
            Dropout(dense_dropout),
        ])

        self.SA_attention = Dense(SA_dim)
        self.SA_extra_fc = tf.keras.Sequential([
            Dense(hidden_dim),
            ReLU(),
            Dropout(dense_dropout)
        ])

        self.graph_fc = tf.keras.Sequential([
            Dense(hidden_dim),
            ReLU(),
            Dropout(dense_dropout)
        ])

        self.task_fc = [Dense(hidden_dim) for _ in range(self.num_hidden_layers)]
        self.task_fc.append(Dense(1))
        self.task_dropouts = [
            Dropout(dense_dropout) for _ in range(self.num_hidden_layers)
        ]

        self.pool = Global_Attention()
        self.fusion = WeightFusion(feat_views, hidden_dim)
        self.Layernorm = tf.keras.Sequential([
            Dense(hidden_dim),
            LayerNormalization(),
            ReLU()
        ])


    def call(self, x, adjoin_matrix, mask, x_KE, x_SA, training=False):
        ck = []
        KE_attn_weights = tf.sigmoid(self.KE_attention(x_KE))
        SA_attn_weights = tf.sigmoid(self.SA_attention(x_SA))

        x_weighted_KE = x_KE * KE_attn_weights
        x_weighted_SA = x_SA * SA_attn_weights

        x_weighted_KE = self.KE_extra_fc(x_weighted_KE, training=training)
        x_weighted_SA = self.SA_extra_fc(x_weighted_SA, training=training)

        x_graph = self.encoder(x, training=training, mask=mask, adjoin_matrix=adjoin_matrix)
        x_graph = x_graph[:, 0, :]
        x_graph = self.graph_fc(x_graph, training=training)

        graph_pool = self.pool(x_graph)
        KE_pool = self.pool(x_weighted_KE)
        SA_pool = self.pool(x_weighted_SA)

        ck.append(self.Layernorm(graph_pool))
        ck.append(self.Layernorm(KE_pool))
        ck.append(self.Layernorm(SA_pool))
        molecule_emb = self.fusion(ck)
        task_hidden = molecule_emb
        for j in range(self.num_hidden_layers):
            task_hidden = tf.nn.relu(self.task_fc[j](task_hidden))
            task_hidden = self.task_dropouts[j](task_hidden, training=training)

        task_output = self.task_fc[-1](task_hidden)

        return task_output, KE_attn_weights, SA_attn_weights, ck, molecule_emb






