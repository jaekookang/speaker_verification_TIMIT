'''
Train

2018-07-07
'''
import ipdb as pdb
import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import utils
from datafeeder import gen_batch
from hparams import hparams as hp


class Graph:
    def __init__(self, training=True):

        # Set up Graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Define Data feeding
            # x: mel-spectrogram, float32 (batch_size, T, num_mels)
            # y: speaker ID (index), int32 (batch_size, 1)
            if training:
                self.y, self.x, num_batch = gen_batch()
            else:
                self.x = tf.placeholder(
                    tf.float32, shape=(None, None, hp.num_mels))
                self.y = tf.placeholder(tf.int32, shape=(None, 1))

            # Stacked LSTM
            with tf.name_scope('Stacked_LSTM'):
                cell = tf.contrib.rnn.MultiRNNCell(
                    [tf.contrib.rnn.BasicLSTMCell(hp.num_hids)
                     for _ in range(hp.num_LSTM_layers)])
                # time_major=False: (batch_size x time x dim)
                # outputs: (batch_size x time x num_hids)
                # states: (LSTMState(batch_size x num_hids),
                #         LSTMState(batch_size x num_hids),
                #         LSTMState(batch_size x num_hids))
                outputs, states = tf.nn.dynamic_rnn(
                    cell, self.x, time_major=False, dtype=tf.float32)
            last_output = outputs[:, -1, :]  # (batch_size x num_hids)

            # Last projection layer
            with tf.name_scope('Output_layer'):
                projected = tf.layers.dense(
                    last_output, hp.embed_size, activation=None)  # linear ??
                l2_normalized = tf.nn.l2_normalize(
                    projected, axis=1, name='L2_normalized')

            # Scoring
            with tf.name_scope('Scoring'):
                pass
        pdb.set_trace()

        # Define Loss

        pass


if __name__ == '__main__':
    # Setting
    starttime = utils.get_time()
    save_dir = os.path.join(hp.model_dir, 'model')
    utils.safe_mkdir(save_dir)
    logger = utils.set_logger(save_dir)

    G = Graph()
    print('Graph built')
    with G.graph.as_default():
        with tf.Session() as sess:
            print('haha')

    print('Finished')
