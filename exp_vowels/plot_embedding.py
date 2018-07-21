'''
Plot vectors

2018-07-21
'''

import matplotlib.pyplot as plt
import ipdb as pdb
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector


def make_tsv(save_dir):
    '''Make tsv meta file'''
    header = S.columns[:3].tolist() + ['Vowel']
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
                    f.write('\t'.join(data + [v]) + '\n')
    print('tsv written')


if __name__ == '__main__':
    # Load data
    SDICT_DIR = 'spkr_dict.npy'
    LOG_DIR = 'visualization'
    sdict = np.load(SDICT_DIR).item()
    S = pd.read_table('../data/spkr_info.txt', sep=',', na_filter=False)
    vowels = ['iy', 'ae', 'aa']

    # Centralize data per speaker & combine all
    x = np.array([], dtype=np.float32).reshape(0, 40)
    for s in sdict.keys():
        for v in vowels:
            centered = sdict[s][v] - \
                np.mean(sdict[s][v], axis=1, keepdims=True)
            sdict[s][v] = centered
            x = np.vstack([x, centered])

    # Write meta file
    make_tsv('visualization')

    # Set up data
    embedding_var = tf.Variable(x, name='Vowel_vectors_by_speakers')
    summary = tf.summary.FileWriter(LOG_DIR)

    # Set up configuration
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
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
