'''
Plot vectors

2018-07-21
'''

import matplotlib.pyplot as plt
import ipdb as pdb
from time import strftime, gmtime
import os
import re
import sys
import shutil
import random
import glob
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from make_spkr_dict import hp


def timenow():
    '''Get current time'''
    return strftime("%Y-%m-%d_%H_%M_%S", gmtime())


def safe_mkdir(path):
    ''' Create a directory if there isn't one already. '''
    try:
        os.mkdir(path)
    except OSError:
        pass


def safe_rmdir(path):
    try:
        if os.path.exists(path):
            shutil.rmtree(path)
    except OSError:
        pass


def make_tsv(save_dir, spkr_keys, spkr_id=None):
    '''Make tsv meta file

    Make meta.tsv (default)
    Make meta_{spkr}.tsv (optional)
    '''
    header = S.columns[:3].tolist() + ['Phone'] + ['Context']

    # For each speaker
    with open(os.path.join(save_dir, 'meta.tsv'), 'w') as f:
        f.write('\t'.join(header) + '\n')
        for s in spkr_keys:
            data = S.loc[S.ID == s, ['ID', 'Sex', 'DR']].values[0]
            data = [str(d) for d in data]
            # For each vowel
            for v in vowels:
                # For each data point
                for r in range(sdict[s][v].shape[0]):
                    ctx = cdict[s][v][r]
                    f.write('\t'.join(data + [v] + [ctx]) + '\n')
    # Only speakers in spkr_id
    if spkr_id is not None:
        for s in spkr_id:
            with open(os.path.join(save_dir, f'meta_{s}.tsv'), 'w') as f:
                f.write('\t'.join(header) + '\n')
                data = S.loc[S.ID == s, ['ID', 'Sex', 'DR']].values[0]
                data = [str(d) for d in data]
                # For each vowel
                for v in vowels:
                    # For each data point
                    for r in range(sdict[s][v].shape[0]):
                        ctx = cdict[s][v][r]
                        f.write('\t'.join(data + [v] + [ctx]) + '\n')
    print('tsv written')


if __name__ == '__main__':
    # Get suffix for LOG_DIR
    try:
        suffix = sys.argv[1]
    except IndexError:
        print('\n >> Provide suffix for log directory! << \n')
        print(' eg. python plot_embedding.py allSpeakers')
        raise

    # Load data
    TMT_DIR = '../data/TMT'
    # SDICT_DIR = 'spkr_sdict.npy'
    # CDICT_DIR = 'spkr_cdict.npy'
    SDICT_DIR = 'spkr_sdict_vowel_stop_fric_afric.npy'
    CDICT_DIR = 'spkr_cdict_vowel_stop_fric_afric.npy'
    LOG_DIR = f'vis_{suffix}'
    META_DIR = os.path.join(LOG_DIR, 'meta')
    sdict = np.load(SDICT_DIR).item()
    cdict = np.load(CDICT_DIR).item()
    S = pd.read_table('../data/spkr_info.txt', sep=',', na_filter=False)
    # vowels = ['iy', 'aa', 'uh', 's', 'z', 'sh', 'f']
    vowels = ['iy', 'ae', 'aa', 'uh',
              'b', 'd', 'g', 'p', 't', 'k',
              's', 'z', 'sh', 'dh', 'f', 'v',
              'jh', 'ch']
    NUM_DIM = [3, 6, 12, 24, 36]
    spkr_num = 630  # total: 630
    spkr_keys = random.sample([*sdict], spkr_num)

    safe_rmdir(LOG_DIR)
    safe_mkdir(LOG_DIR)
    safe_mkdir(META_DIR)

    # Combine all data
    _x = np.array([], dtype=np.float32).reshape(0, hp.n_mfcc)
    sdict_cent = {s: np.array([], dtype=np.float32).reshape(0, hp.n_mfcc)
                  for s in spkr_keys}
    meta_spkr_vowel = []
    for s in spkr_keys:
        for v in vowels:
            _x = np.vstack([_x, sdict[s][v]])
            sdict_cent[s] = np.vstack(
                [sdict_cent[s],
                 sdict[s][v] - np.mean(sdict[s][v], axis=0, keepdims=True)])
            # Add meta lines
            for _ in range(sdict[s][v].shape[0]):
                meta_spkr_vowel.append([s, v])

    # Center data
    x = _x - np.mean(_x, axis=0, keepdims=True)
    # Save centered data
    np.save(os.path.join(LOG_DIR, 'x_data.npy'), x)
    # Save meta data (speaker, vowel)
    np.save(os.path.join(LOG_DIR, 'x_meta.npy'), meta_spkr_vowel)

    # # Prepare audio data
    # wavs = sorted(glob.glob(os.path.join(TMT_DIR, '**', '**', '**', '*.WAV')))

    # Select num dimension & make tf.Variable
    var1, var2 = [], []
    for d in NUM_DIM:
        var1.append(tf.Variable(x[:, :d], name=f'Vowel_vectors_ndim_{d}'))
    for s in sdict_cent.keys():
        var2.append(tf.Variable(sdict_cent[s][:, :3], name=f'{s}'))

    # Write meta file
    make_tsv(META_DIR, spkr_keys)
    make_tsv(META_DIR, spkr_keys, spkr_id=sdict_cent.keys())

    # Set up summary
    # for wav in wavs[:100]:
    #     fid = wav.split('/')[-2][1:]
    #     y, sr = librosa.load(wav, sr=16000)
    #     tf.summary.audio(f'{fid}', y.reshape((1, -1)),
    #                      sample_rate=16000, max_outputs=1, family=fid)
    summary_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter(LOG_DIR)

    # Set up configuration
    config = projector.ProjectorConfig()
    for v in var1:  # for all speakers
        embedding = config.embeddings.add()
        embedding.tensor_name = v.name
        embedding.metadata_path = 'meta/meta.tsv'
    for v in var2:  # for individual speakers
        embedding = config.embeddings.add()
        embedding.tensor_name = v.name
        _name = re.sub('\:[0-9]', '', v.name)
        embedding.metadata_path = f"meta/meta_{_name}.tsv"
    projector.visualize_embeddings(writer, config)

    # Save the data
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    # summary = sess.run(summary_op)
    # writer.add_summary(summary)
    saver.save(sess, os.path.join(LOG_DIR, 'data.ckpt'), 1)

    print('done')

    '''
    Refs:
    - http://www.pinchofintelligence.com/simple-introduction-to-tensorboard-embedding-visualisation/
    '''
