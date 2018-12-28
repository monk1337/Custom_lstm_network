import tensorflow as tf
import numpy as np



class SentimentNetwork(object):

    
    def __init__(self,word_embedding_load,pos_embedding_load,sentiment_embedding_load):

        tf.reset_default_graph()

        # placeholders
        sentences = tf.placeholder(tf.int32, [None, None], name='sentence')
        pos       = tf.placeholder(tf.int32, [None, None], name='pos')
        sentiment = tf.placeholder(tf.int32, [None,None] , name ='sentiment')
        
        
        labels    = tf.placeholder(tf.int32, [None, ], name='label')
        mode      = tf.placeholder(tf.int32, (), name='mode')
        
        
        self.placeholders = {
                'sentence' : sentences,
                'pos'      : pos,
                'sentiment': sentiment,
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

        # word embedding
        word_embedding = tf.get_variable(
                                         name="Word_embedding", 
                                         shape=[4330,300],
                                         initializer=tf.constant_initializer(
                                             np.array(word_embedding_load)), 
                                         trainable=False
                                        )
        # add UNK and PAD
        wemb = tf.concat([ tf.zeros([2, 300]), word_embedding], axis=0)
        
        
        #Embedding_lookup
        word_embedding_lookup = tf.nn.embedding_lookup(wemb, sentences)


        
        
        
        
        #pos_embedding
        pos_embedding = tf.get_variable(
                                         name="Pos_embedding", 
                                         shape=[16,50],
                                         initializer=tf.constant_initializer(
                                             np.array(pos_embedding_load)), 
                                         trainable=False
                                        )
        #pos_embedding_lookup
        pos_embedding_lookup = tf.nn.embedding_lookup(pos_embedding,  pos)
        
        
        #sentiment_embedding
        sentiment_embedding = tf.get_variable(
                                         name="Sentiment_embedding", 
                                         shape=[4330,6],
                                         initializer=tf.constant_initializer(
                                             np.array(sentiment_embedding_load)), 
                                         trainable=False
                                        )
        #sentiment_embedding_lookup
        sentiment_embedding_lookup = tf.nn.embedding_lookup(sentiment_embedding,  sentiment)
        
        
        improved_vecot             = tf.concat(
                                               values=[
                                                       word_embedding_lookup, 
                                                       pos_embedding_lookup, 
                                                       sentiment_embedding_lookup], 
                                               axis=-1)
        
        #check
        
        self.out = {
                     'out1': word_embedding_lookup , 
                     'out2': pos_embedding_lookup ,
                     'out3' : sentiment_embedding_lookup , 
                     'impro':improved_vecot
                   }
        
        
        
        
        
        


#         with tf.variable_scope('rnn_cell') as scope:
#             cell = DropoutWrapper(
#                     tf.nn.rnn_cell.LSTMCell(hdim), 
#                     output_keep_prob=1. - dropout
#                     )

#         with tf.variable_scope('encoder') as scope:
#             outputs, final_state = tf.nn.dynamic_rnn(
#                     cell = cell,
#                     inputs = emb_sentence,
#                     sequence_length = tf.count_nonzero(sentences, axis=-1),
#                     dtype=tf.float32
#                     )

#         logits = tf.contrib.layers.fully_connected(final_state.c, num_labels)

#         self.out = {
#                 'prob' : tf.nn.softmax(logits),
#                 'pred' : tf.argmax(tf.nn.softmax(logits), axis=-1),
#                 'loss' : tf.reduce_mean(
#                     tf.nn.sparse_softmax_cross_entropy_with_logits(
#                         logits=logits,
#                         labels=labels
#                         ))
#                     }

#         self.out['accuracy'] = tf.cast(tf.equal(
#             tf.cast(self.out['pred'], tf.int32), 
#             labels), tf.float32)

#         self.train_op = tf.train.AdamOptimizer().minimize(self.out['loss'])


def rand_execution(netw):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        return sess.run(netw.out, feed_dict = {
            netw.placeholders['sentence'] : np.random.randint(0, 100, [8, 10]),
            netw.placeholders['pos'] : np.random.randint(0, 10, [8, 10]),
            netw.placeholders['sentiment'] : np.random.randint(0, 10, [8, 10]),
            netw.placeholders['mode']    : 0,
            netw.placeholders['label']    : np.random.randint(0, 4, [8, ])
            })['impro'].shape


if __name__ == '__main__':

    netw = SentimentNetwork(w,p,s)
    print(rand_execution(netw))


w = np.random.uniform(0,1,[4330,300])
p = np.random.uniform(0,1,[16,50])
s = np.random.uniform(0,1,[4330,6])
