'''
Training

2018-07-14
'''

import ipdb as pdb
import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from datafeeder import gen_batch
from hparams import hparams as hp
from hparams import debug_hparams
from utils import *


class Graph:

    def __init__(self, is_training=True):
        # Set up a graph
        self.graph = tf.Graph()
        with self.graph.as_default():

            # Define data feeding
            #  x: mel spectrogram, (batch_size, time, num_mels)
            #  y: speaker ID (index), (batch_size,)
            with tf.name_scope('Data'):
                self.x = tf.placeholder(
                    tf.float32, (None, None, hp.num_mels), 'x')
                self.y = tf.placeholder(tf.int32, (None,), 'y')
                self.global_step = tf.Variable(
                    0, name='global_step', trainable=False)

            # Stacked LSTM
            with tf.name_scope('Stacked_LSTM'):
                cell = tf.contrib.rnn.MultiRNNCell(
                    [tf.contrib.rnn.BasicLSTMCell(hp.num_hids)
                     for _ in range(hp.num_LSTM_layers)])
                outputs, states = tf.nn.dynamic_rnn(
                    cell, self.x, time_major=False, dtype=tf.float32)
                # outputs: (batch_size, time, num_hids)
                # states: a stacked list of LSTMState(batch_size, num_hids)
                last_output = outputs[:, -1, :]  # (batch_size, num_hids)

            # Last projection layer
            with tf.name_scope('Projection_Layer'):  # <- 'd'-vector
                projected = tf.layers.dense(  # (batch_size, embed_size)
                    last_output, hp.embed_size, activation=None)
                self.projected_norm = tf.nn.l2_normalize(  # (batch_size, embed_size)
                    projected, axis=1, name='L2_normalized')

            # Scoring
            final_dense = True
            with tf.name_scope('Scoring'):
                # Make embedding matrix
                indices = (tf.range(hp.batch_utt) +
                           tf.range(0, hp.batch_size, hp.batch_utt)[:, tf.newaxis])[..., tf.newaxis]
                # (batch_spkr, batch_utt, embed_size); (24, 5, 256)
                self.embed = tf.gather_nd(
                    self.projected_norm, indices, 'embeddings')
                # Compute speaker-wise averaged centroid,
                # (batch_spkr, embed_size) -> (embed_size, batch_spkr)
                self.centroid = tf.transpose(tf.reduce_mean(
                    self.embed, axis=1), name='centroid')
                # (batch_spkr*batch_utt, batch_spkr)
                self.cos = tf.matmul(self.projected_norm, self.centroid)
                self.norm = tf.matmul(tf.norm(self.projected_norm,
                                              axis=1, keepdims=True),
                                      tf.norm(self.centroid,
                                              axis=0, keepdims=True), name='norm')
                self.cos /= self.norm  # normalize
                self.S = tf.layers.dense(
                    self.cos, hp.batch_spkr, activation=None, name='similarity_matrix')
                _, self.spkr_idx = tf.unique(self.y)
                # (batch_size, batch_spkr)
                self.y_out = tf.one_hot(
                    self.spkr_idx, depth=hp.batch_spkr, name='y_one_hot')
            # Loss
            with tf.name_scope('Loss'):
                # _loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                #     labels=self.y_out, logits=self.S)
                _loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.spkr_idx, logits=self.S)
                self.loss = tf.reduce_mean(_loss, name='loss')

            # Train op
            with tf.name_scope('Train_Op'):
                self.optimizer = tf.train.AdamOptimizer(
                    learning_rate=hp.learning_rate)
                # Gradient clipping
                self.grads = self.optimizer.compute_gradients(self.loss)
                self.clipped = []
                for grad, var in self.grads:
                    grad = tf.clip_by_norm(grad, hp.grad_clip)
                    self.clipped.append((grad, var))
                self.train_op = self.optimizer.apply_gradients(
                    self.clipped, global_step=self.global_step)

            # Summary
            with tf.name_scope('Summary'):
                tf.summary.scalar('Loss', self.loss)
                tf.summary.image('Similarity_matrix',
                                 tf.reshape(self.S, (1, -1, hp.batch_spkr, 1)))
                tf.summary.image('Onehot_matrix', tf.reshape(
                    self.y_out, (1, -1, hp.batch_spkr, 1)))
                tf.summary.image('Centroid', tf.reshape(
                    self.centroid, (1, -1, hp.batch_spkr, 1)))
                tf.summary.image('Norm', tf.reshape(
                    self.norm, (1, -1, hp.batch_spkr, 1)))
                tf.summary.histogram('Speakers', self.y)
                tf.summary.histogram('Similarity_matrix', self.S)
                self.summary_op = tf.summary.merge_all()


if __name__ == '__main__':
    begtime = get_time()
    save_dir = os.path.join(hp.model_dir, f'model_{begtime}')
    safe_rmdir(save_dir)
    safe_mkdir(save_dir)

    g = Graph()
    print('Graph built')
    with g.graph.as_default():
        saver = tf.train.Saver(max_to_keep=5)
        with tf.Session(graph=g.graph) as sess:
            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter(save_dir, g.graph)
            # config = projector.ProjectorConfig()
            # # Add embedding on TensorBoard
            # embedding = config.embeddings.add()
            # embedding.tensor_name = g.centroid.name
            # # embedding.metadata_path = meta_file
            # projector.visualize_embeddings(writer, config)

            total_loss = 0.0
            for i in range(hp.max_epoch):
                print(f'Epoch:{i}/{hp.max_epoch}', flush=True)
                gen = gen_batch()
                batch_loss = 0.0
                for b in tqdm(range(hp.num_batch), total=hp.num_batch, unit='b'):
                    batch_x, batch_y = next(gen)
                    gs, loss, summary, spkrs, _ = sess.run(
                        [g.global_step, g.loss, g.summary_op, g.y, g.train_op],
                        {g.x: batch_x, g.y: batch_y})
                    writer.add_summary(summary, global_step=gs)
                    batch_loss += loss

                    # Get speaker meta file
                    idx2spkr = np.load('data/idx2spkr.npy').item()
                    spkr_id = list(set([idx2spkr[i] for i in spkrs]))
                    meta_file = write_spkr_meta(spkr_id, save_dir)

                total_loss += batch_loss/hp.num_batch
                print(f'total_loss: {total_loss}')

                if (i+1) % hp.n_test == 0:
                    # Save model
                    saver.save(sess, os.path.join(
                        save_dir, 'checkpoints'), global_step=gs)
    print('Finished')
