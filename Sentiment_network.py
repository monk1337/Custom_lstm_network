import tensorflow as tf
import numpy as np
import Context_aware_Attention
from tensorflow.contrib import rnn



class SentimentNetwork(object):

    
    def __init__(self):


        tf.reset_default_graph()


        # placeholders
        sentences = tf.placeholder(tf.float32, [None,35,356], name='sentence')
        
        
        labels    = tf.placeholder(tf.int32, [None, ], name='label')
        mode      = tf.placeholder(tf.int32, (), name='mode')
        
        
        self.placeholder = {
                'sentence' : sentences,
                'label'    : labels,
                
                'mode'     : mode
                }
        
        
        DropoutWrapper = tf.nn.rnn_cell.DropoutWrapper

        
        

        # drop out
        dropout = tf.cond(
                tf.equal(mode, 0), # If
                lambda : 0.5, # True
                lambda : 0. # False
                )
        
        #first convo layer
        with tf.name_scope("convo_first"):


            weight_f = tf.get_variable(
                                        name='convo_first_weight', 
                                        shape=[3,356,100],
                                        initializer=tf.random_uniform_initializer(-0.01, 0.01), 
                                        dtype=tf.float32
                                      )


            bias_f = tf.get_variable(

                                        name='convo_first_bias', 
                                        shape=[100], 
                                        initializer=tf.random_uniform_initializer(-0.01, 0.01),
                                        dtype=tf.float32
                                    )


            output_f = tf.nn.conv1d(sentences, weight_f, stride=1, padding="SAME")
            h_f = tf.nn.relu(tf.nn.bias_add(output_f, bias_f), name="relu")




        with tf.name_scope("convo_second"):

            weight_s = tf.get_variable(
                                        name='convo_second_weight', 
                                        shape=[4,100,100],
                                        initializer=tf.random_uniform_initializer(-0.01, 0.01),
                                        dtype=tf.float32
                                      )


            bias_s = tf.get_variable(
                                        name='convo_second_bias', 
                                        shape=[100], 
                                        initializer=tf.random_uniform_initializer(-0.01, 0.01),
                                        dtype=tf.float32
                                    )

            output_s = tf.nn.conv1d(h_f, weight_s, stride=1, padding="SAME")
            h_s = tf.nn.relu(tf.nn.bias_add(output_s, bias_s), name="relu")



        with tf.name_scope("convo_third"):

            weight_t = tf.get_variable(
                                        name='convo_third_weight', 
                                        shape=[5,100,100],
                                        initializer=tf.random_uniform_initializer(-0.01, 0.01), 
                                        dtype=tf.float32
                                      )


            bias_t = tf.get_variable(
                                        name='convo_third_bias', 
                                        shape=[100], 
                                        initializer=tf.random_uniform_initializer(-0.01, 0.01),
                                        dtype=tf.float32
                                    )

            output_t = tf.nn.conv1d(h_s, weight_t, stride=1, padding="SAME")
            h_t = tf.nn.relu(tf.nn.bias_add(output_t, bias_t), name="relu")

    
            # with tf.Session() as sess:
            #     sess.run(tf.global_variables_initializer())
            #     print(sess.run(h_t,feed_dict={input_x:dummpy_data}).shape)
                
            # pooled = tf.nn.pool(h_t,[1, 30, 1, 1],padding='VALID',
            #                         name="pool")
            
        
        #max_pool from all three convolutional layers
        B = tf.nn.pool(h_t, [5], 'MAX', 'SAME', strides = [5])

        #Applying attention_layer
        final_output = Context_aware_Attention.attention(
                                                            inputs=B,
                                                            attention_size=300,
                                                            time_major=False, 
                                                            return_alphas=False
                                                        )


        # attention_size = 100
        # sequence_length = inputs.shape[1].value # the length of sequences processed
        # hidden_size = inputs.shape[2].value # features_maps
        # print(sequence_length,hidden_size)

        # # Attention mechanism
        # W_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
        # b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        # u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

        # v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1]))
        # vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
        # exps = tf.reshape(tf.exp(vu), [-1, sequence_length])
        # alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])

        # output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, sequence_length, 1]), 1)
        # final_output = Context_aware_Attention.attention(B,100,time_major=False, return_alphas=False)
        # final_output = tf.reshape(B,[tf.shape(sentences)[0],-1])


 
        
        
        # #num_cell = 12
        # #ex : # 24 x 1
        # attention_size = tf.get_variable(name='attention_size',
        #                                  shape=[2*300,1],
        #                                  dtype=tf.float32,
        #                                  initializer=tf.random_uniform_initializer(-0.01,0.01))
        # # bias 1
        # bias          = tf.get_variable(name='bia_s',shape=[1],
        #                                 dtype=tf.float32,
        #                                 initializer=tf.random_uniform_initializer(-0.01,0.01))
        
        
        # #projection without activation 
        # #ex : 120x24 matmul 24x 1 ==> 120x1
        # attention_projection = tf.add(tf.matmul(input_reshape,attention_size),bias)
        
        
        # #reshape . 120x1 ==> 12x10x1 (shape of input )
        # output_reshape = tf.reshape(attention_projection,[tf.shape(sentences)[0],tf.shape(sentences)[1],-1])
        
        # #softmax over logits 12x10x1
        # attention_output = tf.nn.softmax(output_reshape,dim=1)
        
        
        # #reshape as input 12x10
        # attention_visualize = tf.reshape(attention_output,
        #                                  [tf.shape(sentences)[0],
        #                                   tf.shape(sentences)[1]],
        #                                  name='Plot')
        
        
        # # 12x10x1 multiply 12x10x24  == > 12x10x24
        # attention_projection_output = tf.multiply(attention_output,transpose)
        
        # #reduce across time 120x10x24 ==> 12x24
        # Final_output = tf.reduce_sum(attention_projection_output,1)

#         self.output = {

#             'loss':  B,
#             'accuracy':  final_output

#         }


        # # state_output = tf.concat([state_c[0].c, state_h[0].c], axis=-1)
        
        # # #check

        #        # will return [batch_size, output_state_size]
        weights = tf.get_variable(name='weights',
                                  shape=[100, 3],
                                  dtype=tf.float32,
                                  initializer=tf.random_uniform_initializer(-0.01, 0.01))

        bias = tf.get_variable(name='bias',
                               shape=[3],
                               dtype=tf.float32,
                               initializer=tf.random_uniform_initializer(-0.01, 0.01))

        logits = tf.add(tf.matmul(final_output , weights),
                        bias, name='network_output')

        # self.check_shapes={'embedding':embedding_lookup,'model':model,'tr':transpose,'atten':attention_output,'al':alphas,'outa':mat_out}

        probability_distribution = tf.nn.softmax(logits, name='netout')

        prediction = tf.argmax(probability_distribution, axis=-1)

        # cross entropy
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=labels
        )
        # loss
        loss = tf.reduce_mean(ce)

        # accuracy calculation

        accuracy_calculation = tf.reduce_mean(tf.cast(
            tf.equal(
                tf.cast(prediction, tf.int32), labels),
            tf.float32))

        self.output = {

            'loss': loss,
            'prob': probability_distribution,
            'pred': prediction,
            'logits': logits,
            'accuracy': accuracy_calculation

        }

        self.train = tf.train.AdamOptimizer().minimize(loss)

        
        
        
        
        
        


        # with tf.variable_scope('rnn_cell') as scope:
        #     cell = DropoutWrapper(
        #             tf.nn.rnn_cell.LSTMCell(hdim), 
        #             output_keep_prob=1. - dropout
        #             )

        # with tf.variable_scope('encoder') as scope:
        #     outputs, final_state = tf.nn.dynamic_rnn(
        #             cell = cell,
        #             inputs = emb_sentence,
        #             sequence_length = tf.count_nonzero(sentences, axis=-1),
        #             dtype=tf.float32
        #             )

        # logits = tf.contrib.layers.fully_connected(final_state.c, num_labels)

        # self.out = {
        #         'prob' : tf.nn.softmax(logits),
        #         'pred' : tf.argmax(tf.nn.softmax(logits), axis=-1),
        #         'loss' : tf.reduce_mean(
        #             tf.nn.sparse_softmax_cross_entropy_with_logits(
        #                 logits=logits,
        #                 labels=labels
        #                 ))
        #             }

        # self.out['accuracy'] = tf.cast(tf.equal(
        #     tf.cast(self.out['pred'], tf.int32), 
        #     labels), tf.float32)

        # self.train_op = tf.train.AdamOptimizer().minimize(self.out['loss'])


# def rand_execution(netw):
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         return sess.run(netw.out, feed_dict = {
#             netw.placeholders['sentence'] : np.random.randint(0, 100, [8, 10]),
#             netw.placeholders['pos'] : np.random.randint(0, 10, [8, 10]),
#             netw.placeholders['sentiment'] : np.random.randint(0, 10, [8, 10]),
#             netw.placeholders['mode']    : 0,
#             netw.placeholders['label']    : np.random.randint(0, 4, [8, ])
#             })['impro'].shape


# if __name__ == '__main__':

#     netw = SentimentNetwork(w,p,s)
#     print(rand_execution(netw))


# w = np.random.uniform(0,1,[4330,300])
# p = np.random.uniform(0,1,[16,50])
# s = np.random.uniform(0,1,[4330,6])