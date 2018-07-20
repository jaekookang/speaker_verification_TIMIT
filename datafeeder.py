'''
Datafeeder

2018-07-06

TODO:
[] Fix batch length difference
[] Add batch to output file w/ speaker info

ref:
- https://www.github.com/kyubyong/deepvoice3
'''
import ipdb as pdb
import os
import glob
import random
import numpy as np
import tensorflow as tf
import pandas as pd
import textgrid
from tqdm import tqdm

from hparams import hparams as hp
from utils import safe_mkdir, safe_rmdir, find_elements


def load_meta_data(meta_data):
    '''Load speaker meta data'''
    return pd.read_table(meta_data, sep=',')


def load_data(is_training=True):
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
    mel_train = sorted(glob.glob(os.path.join(
        hp.train_dir, '**', 'mel', '*.npy')))
    mel_test = sorted(glob.glob(os.path.join(
        hp.test_dir, '**', 'mel', '*.npy')))
    mel_all = mel_train + mel_test

    # Get speaker IDs
    for m in mel_all:
        spkr_list.append(m.split('/')[-3])
    spkr_uq = list(set(spkr_list))
    spkr2idx = {s: i for i, s in enumerate(spkr_uq)}
    idx2spkr = {i: s for i, s in enumerate(spkr_uq)}
    # Save
    np.save(os.path.join(hp.data_dir, 'spkr2idx.npy'), spkr2idx)
    np.save(os.path.join(hp.data_dir, 'idx2spkr.npy'), idx2spkr)

    # Make balanced dataset
    if is_training:
        mel_list = mel_train
    else:
        mel_list = mel_test
    mel_bal = []  # a list of mel files
    spkr_bal = []  # a list of spkr (idx)
    for i in range(hp.num_batch):
        spkr_samp = random.sample(spkr_uq, hp.batch_spkr)
        for s in spkr_samp:
            _, _mel = find_elements(s, mel_list)
            mel_samp = random.choices(_mel, k=hp.batch_utt)
            mel_bal += mel_samp
            spkr_bal += [spkr2idx[s]]*hp.batch_utt
    return mel_bal, spkr_bal


def slice_mel(x, length):
    '''Slice mel data'''
    x = np.load(x)
    beg_frame = random.randint(1, x.shape[0] - length)
    return x[beg_frame:beg_frame+length, :]  # (time, num_mels)


def gen_batch(is_training=True):
    '''Load data and prepare queue'''
    with tf.device('/cpu:0'):
        # Load data
        mel_list, spkr_list = load_data(is_training)

        # For each mini-batch
        i = 0
        for b in range(hp.num_batch):
            # Randomly choose frame number (140<= len <=180)
            length = random.randint(hp.length[0], hp.length[1])
            # Initiate (batch_size, length, num_mels)
            mel_batch = np.zeros((hp.batch_size, length, hp.num_mels))
            spkr_batch = []  # (batch_size)
            # Stack data
            m = 0
            while 1:
                mel_batch[m] = slice_mel(mel_list[i], length)
                spkr_batch.append(spkr_list[i])
                i += 1
                m += 1
                if i % hp.batch_size == 0:
                    break
            yield mel_batch, spkr_batch


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
    # batch_x, batch_y, batch_x_txt, num_batch = gen_batch()
    gen = gen_batch()
    for _ in range(hp.num_batch):
        x, y = next(gen)
        pdb.set_trace()
