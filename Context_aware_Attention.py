import tensorflow as tf
import numpy as np

def attention(inputs, attention_size, time_major=False, return_alphas=False):

    if isinstance(inputs, tuple):
        inputs = tf.concat(inputs, 2)

    if time_major:
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])


    # Y = tf.constant(np.random.randn(batch_size, seq_len, dim), tf.float32)
    
    # # [batch_size x dim]            -- h_N
    # h = tf.constant(np.random.randn(batch_size, dim), tf.float32)

    # initializer = tf.random_uniform_initializer()
    # W = tf.get_variable("weights_Y", [dim, dim], initializer=initializer)
    # w = tf.get_variable("weights_w", [dim], initializer=initializer)

    # # [batch_size x seq_len x dim]  -- tanh(W^{Y}Y)
    # M = tf.tanh(tf.einsum("aij,jk->aik", Y, W))
    # # [batch_size x seq_len]        -- softmax(Y w^T)
    # a = tf.nn.softmax(tf.einsum("aij,j->ai", M, w))
    # # [batch_size x dim]            -- Ya^T
    # r = tf.einsum("aij,ai->aj", Y, a)
    
    sequence_length = inputs.shape[1].value # the length of sequences processed in the antecedent RNN layer
    hidden_size = inputs.shape[2].value # hidden size of the RNN layer

    # Attention mechanism
    W_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1]))
    vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
    exps = tf.reshape(tf.exp(vu), [-1, sequence_length])
    alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])

    # Output of Bi-RNN is reduced with attention vector
    output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, sequence_length, 1]), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas