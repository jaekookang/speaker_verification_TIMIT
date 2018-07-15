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
      spkr_idx: a list of speaker index
      mel_list: a list of mel files
      spkr2mel: a dictionary for mel files (values) given speaker (key)
      spkr2idx: a dictionary for speaker index
      idx2spkr: a dictionary for finding speaker from index

    TODO: make it simple
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
    spkr2mel = {s: [] for s in spkr_list}

    # Get spkr2mel dictionary
    for s, m in zip(spkr_list, mel_list):
        spkr2mel[s].append(m)

    assert spkr_list != []
    assert mel_list != []

    # Organize speaker ID with mel files
    # To balanace each mini-batch with batch_spkr and batch_utt,
    # data were appended sequentially as below:
    spkr_set = list(set(spkr_list))
    spkr_out, mel_out = [], []
    for sidx in range(0, len(spkr_set), hp.batch_spkr):
        try:
            slist = spkr_set[sidx:sidx+hp.batch_spkr]
        except:
            slist = spkr_set[sidx:]
        for s in slist:
            rnd_samp = list(np.random.choice(spkr2mel[s], hp.batch_utt))
            mel_out += rnd_samp
            spkr_out += [s]*hp.batch_utt

    # Load dictionary
    spkr2idx = np.load(os.path.join(hp.data_dir, 'spkr2idx.npy')).item()
    idx2spkr = np.load(os.path.join(hp.data_dir, 'idx2spkr.npy')).item()
    spkr_idx = [spkr2idx[s] for s in spkr_out]
    return spkr_idx, mel_out, spkr2mel, spkr2idx, idx2spkr


def gen_batch():
    '''Load data and prepare queue'''
    with tf.device('/cpu:0'):
        # Load data
        spkr_list, mel_list, spkr2mel, spkr2idx, idx2spkr = load_data()
        pdb.set_trace()
        # Get number of batches
        spkr_set = spkr2mel.keys()
        num_batch = len(spkr_set) // hp.batch_spkr

        # Create Queues
        spkr, mel = tf.train.slice_input_producer(
            [spkr_list, mel_list], shuffle=False)  # shuffle=False !!

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
                                     capacity=num_batch*10,
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


def save_plot(mel_data):
    '''Save Mel outputs as png'''
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 10))
    # (num_mels x time)
    ax.imshow(mel_data.T, aspect='auto', origin='bottom')
    plt.savefig('out.png', format='png')


if __name__ == '__main__':
    # Test this code
    spkrs, mels, num_batch = gen_batch()

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for _ in range(10):
            spkr_out, mel_out = sess.run([spkrs, mels])
            uq_spkr, uq_cnt = np.unique(spkr_out, return_counts=True)
            print(f'Total utts: {len(spkr_out)}')
            print(f'Unique spkr: {len(uq_spkr)}')
            print(f'Utts per spkr: {uq_cnt}\n')
            pdb.set_trace()
        save_plot(mel_out[0])
        coord.request_stop()
        coord.join(threads)
