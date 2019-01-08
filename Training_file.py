# coding: utf-8

#training file for LSTM rnn
import random
import time
import shutil
from tqdm import tqdm
import os 
import numpy as np
import pickle as pk
import tensorflow as tf
import Sentiment_network

epoch = 50

with open('Improved_vector_pad.pkl','rb') as f:
    improved_vector = pk.load(f)
    
with open('labels_pad.pkl','rb') as f:
    labels = pk.load(f)
    
print(len(improved_vector))
print(len(labels))
    
train_data = int(len(improved_vector) * 0.85)
sorted_train_data = improved_vector[:train_data]  # split train_data
labels_sorted_train = labels[:train_data]


sorted_test_data = improved_vector[train_data:]  # split test data
labels_sorted_test = labels[train_data:]

def evaluate_(model, batch_size=50):

    """
    Checking  the test accuracy on testing data set

    :param model:
       current lstm model

    :param batch_size:
       batch size for test data set

    :return:
       mean accuracy of test data set

    :raise:
       if input shape is different from placeholder shape:
          ValueError: Cannot feed value of shape

    """
    sess = tf.get_default_session()

    # batch_data = test_data_in
    # batch_labesl = train_labels

    iteration = len(sorted_test_data) // batch_size

    accuracy = []

    for i in range(iteration):
        batch_data_js = sorted_test_data[i * batch_size:(i + 1) * batch_size]
        labelss       = labels_sorted_test[i * batch_size:(i + 1) * batch_size]
        

        network_out = sess.run(model.output, feed_dict={model.placeholder['sentence']: batch_data_js,
                                                        model.placeholder['label']: labelss,
                                                        model.placeholder['mode']: 1})
        

        accuracy.append(network_out['accuracy'])
    return np.mean(np.array(accuracy))


def train_model(model, batch_size=50):
    """

    :param model:
          current lstm model

    :param batch_size:
          batch size for training

    :print:
          epoch
          iteration
          training_loss
          accuracy

    """
    saver = tf.train.Saver()
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        iteration = len(sorted_train_data) // batch_size
        print("iteration", iteration)
        time.sleep(5)

        for i in range(epoch):

            for j in range(iteration):
                batch_data_j = np.array(sorted_train_data[j * batch_size:(j + 1) * batch_size])
                labels       = np.array(labels_sorted_train[j * batch_size:(j + 1) * batch_size])
                
                print(batch_data_j.shape)
                print(labels.shape)

                
        
                network_out = sess.run(model.output,
                                          feed_dict={model.placeholder['sentence']: batch_data_j,
                                                     model.placeholder['label']: labels,
                                                     model.placeholder['mode']: 0})

                

                print({'epoch': i,
                       'iteration': j,
                       'training_accuracy': network_out['accuracy'],
                       'training_losss': network_out['loss']
                       })
                
                if j % 100 == 0:
                    with open('iterres.txt', 'a') as f:
                        f.write(str({'epoch': i, 'test_accuracy': evaluate_(model, batch_size=150), 'iteration': j}) + '\n')



#             os.system('mkdir ' + str(i) + 'epoch' + str(j))
#             saver.save(sess, '.' + str(i) + 'epoch' + str(j) + '/' + str(i))

#             print({'epoch': i, 'test_accuracy': evaluate_(model)})
#             with open('epochandresult', 'a') as f:
#                 f.write(str({'epoch': i, 'test_accuracy': evaluate_(model)}) + '\n')
            


if __name__ == "__main__":

    model = Sentiment_network.SentimentNetwork()

    train_model(model)
    
    
    
#     def __init__(self, dropout_value_, 
#                  word_vocab_size_, 
#                  word_embedding_dim_, 
#                  forget_bias_, 
#                  rnn_num_units,
#                  labels_nos,char_vocab_size_, 
#                  char_embedding_dim_):