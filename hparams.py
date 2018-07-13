'''
Hyper-parameters

2018-07-06

ref:
- https://github.com/keithito/tacotron
'''

import tensorflow as tf

hparams = tf.contrib.training.HParams(
    # Audio
    num_mels=40,
    num_freq=1025,
    sample_rate=16000,  # TIMIT sampling rate
    frame_width=25,  # ms
    frame_shift=10,  # ms
    preemphasis=.97,

    # Model
    num_hids=768,
    num_LSTM_layers=3,
    embed_size=256,
    batch_utt=10,  # number of utterances per speaker within a mini-batch
    batch_spkr=24,  # number of speakers within a mini-batch
    # batch_size will be batch_utt*batch_spkr; 24*10=640
    length=(140, 180),  # (lower bound, upper bound) in ms
    initial_learning_rate=0.01,
    decrease_learning_rate=10000,  # (in paper) x30M steps
    grad_clip=3,
    frad_scale=0.5,

    # Eval
    max_iters=100000,

    # Data
    data_dir='./data',
    timit_dir='./data/TMT',  # TIMIT
    train_dir='./data/train',  # new
    test_dir='./data/test',  # new
    model_dir='./model',
    meta_dir='./data/SPKRINFO_rev.txt'
)

# Add batch_size
hparams.add_hparam(
    'batch_size', hparams.batch_utt*hparams.batch_spkr)


def debug_hparams():
    '''Print hypterparameters
    '''
    vals = hparams.values()
    hp = [f' {name}: {vals[name]} ' for name in sorted(vals)]
    return 'Hyperparameters\n + ----\n'+'\n'.join(hp)
