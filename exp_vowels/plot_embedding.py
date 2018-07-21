'''
Plot vectors

2018-07-21
'''

import matplotlib.pyplot as plt
import ipdb as pdb
from time import strftime, gmtime
import os
import sys
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector


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


def make_tsv(save_dir):
    '''Make tsv meta file'''
    header = S.columns[:3].tolist() + ['Vowel', 'Speaker']
    with open(os.path.join(save_dir, 'meta.tsv'), 'w') as f:
        f.write('\t'.join(header) + '\n')
        # For each speaker
        for s in sdict.keys():
            data = S.loc[S.ID == s, ['ID', 'Sex', 'DR']].values[0]
            data = [str(d) for d in data]
            # For each vowel
            for v in vowels:
                # For each data point
                for r in range(sdict[s][v].shape[0]):
                    f.write('\t'.join(data + [v] + [data[0]]) + '\n')
    print('tsv written')


if __name__ == '__main__':
    # Get suffix for LOG_DIR
    suffix = sys.argv[1]

    # Load data
    SDICT_DIR = 'spkr_dict.npy'
    LOG_DIR = f'vis_{suffix}'
    sdict = np.load(SDICT_DIR).item()
    S = pd.read_table('../data/spkr_info.txt', sep=',', na_filter=False)
    vowels = ['iy', 'ae', 'aa']
    NUM_DIM = [3, 6, 12, 24, 36]

    safe_rmdir(LOG_DIR)
    safe_mkdir(LOG_DIR)

    # Combine all data
    _x = np.array([], dtype=np.float32).reshape(0, 40)
    for s in sdict.keys():
        for v in vowels:
            _x = np.vstack([_x, sdict[s][v]])
    # Center data
    x = _x - np.mean(_x, axis=0, keepdims=True)

    # Select num dimension & make tf.Variable
    var = []
    for d in NUM_DIM:
        var.append(tf.Variable(x[:, :d], name=f'Vowel_vectors_ndim_{d}'))

    # Write meta file
    make_tsv(LOG_DIR)

    # Set up summary
    summary = tf.summary.FileWriter(LOG_DIR)

    # Set up configuration
    config = projector.ProjectorConfig()
    for v in var:
        embedding = config.embeddings.add()
        embedding.tensor_name = v.name
        embedding.metadata_path = 'meta.tsv'
    projector.visualize_embeddings(summary, config)

    # Save the data
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(LOG_DIR, 'data.ckpt'), 1)

    print('done')

    '''
    Refs:
    - http://www.pinchofintelligence.com/simple-introduction-to-tensorboard-embedding-visualisation/
    '''
