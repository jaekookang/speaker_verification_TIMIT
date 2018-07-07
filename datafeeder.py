'''
Datafeeder

2018-07-06

ref:
- https://www.github.com/kyubyong/deepvoice3
'''
import ipdb as pdb
import os
import re
import glob
import random
import numpy as np
import tensorflow as tf
import pandas as pd
from hparams import hparams as hp


def load_meta_data(meta_data):
    '''Load speaker meta data'''
    return pd.read_table(meta_data, sep=',')


def load_data(training=True):
    '''Load data

    Returns:
      spkr_mel: a dictionary for mel files (values) given speaker (key)
      spkr2idx: a dictionary for speaker index
      idx2spkr: a dictionary for finding speaker from index
    '''
    # Make data lists with ID and data directory
    spkr_list, mel_list = [], []

    # Get Mel directories
    if training:
        mel_list = sorted(glob.glob(os.path.join(
            hp.train_dir, '**', 'mel', '*.npy')))
    else:
        mel_list = sorted(glob.glob(os.path.join(
            hp.test_dir, '**', 'mel', '*.npy')))

    # Get speaker IDs
    for m in mel_list:
        spkr_list.append(m.split('/')[-3])

    # Load dictionary
    spkr2idx = np.load(os.path.join(hp.data_dir, 'spkr2idx.npy')).item()
    idx2spkr = np.load(os.path.join(hp.data_dir, 'idx2spkr.npy')).item()
    return spkr_list, mel_list, spkr2idx, idx2spkr


def gen_batch():
    '''Load data and prepare queue'''
    with tf.device('/cpu:0'):
        # Load data
        spkr_list, mel_list, spkr2idx, idx2spkr = load_data()

        # Get number of batches
        num_batch = len(mel_list) // hp.batch_spkr

        # Create Queues
        spkr, mel = tf.train.slice_input_producer(
            [spkr_list, mel_list], shuffle=True)

        # Slicing
        def slice_mel(x):
            x = np.load(x.decode('utf-8'))
            length = random.randint(hp.length[0], hp.length[1])  # e.g 140 ms
            width = (length - hp.frame_width) // hp.frame_shift + 1  # e.g 12
            beg_frame = random.randint(1, x.shape[0] - width)
            return x[beg_frame:beg_frame+width, :]

        # Decoding
        mel = tf.py_func(slice_mel, [mel], tf.float32)  # (None, num_mels)

        mel = tf.reshape(mel, (-1, hp.num_mels))
        spkr = tf.reshape(spkr, (-1,))

        # Create batch queues
        spkrs, mels = tf.train.batch([spkr, mel],
                                     num_threads=10,
                                     batch_size=hp.batch_size,
                                     capacity=hp.batch_size*32,
                                     dynamic_pad=True)
    return spkrs, mels, num_batch


def find_elements(pattern, my_list):
    '''Find elements in a list'''
    elements = []
    index = []
    for i, l in enumerate(my_list):
        if re.search(pattern, l):
            elements.append(my_list[i])
            index.append(i)
    return index, elements


if __name__ == '__main__':
    # Test this code
    spkrs, mels, num_batch = gen_batch()

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for _ in range(10):
            spkr_out, mel_out = sess.run([spkrs, mels])

        coord.request_stop()
        coord.join(threads)
