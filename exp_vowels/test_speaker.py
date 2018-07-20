'''
Exp1. Test within-speaker variability

2018-07-20
'''
import ipdb as pdb
import os
import textgrid
import glob
import numpy as np
import pandas as pd
import librosa

# Get directories
TMT_DIR = '../data/TMT'
TRAIN_DIR = '../data/train'
TEST_DIR = '../data/test'
SPKR_INFO = '../data/spkr_info.txt'
S = pd.read_table('SPKR_INFO', sep=',')


class hp:
    # Params
    def __init__(self):
        self.num_mels = 40
        self.num_freq = 1025
        self.sample_rate = 16000
        self.frame_width = 25
        self.frame_shift = 10
        self.preemphasis = 0.97


def get_spectrogram(wav_file, phn_file, segments):
    '''Returns normalized log mel-filterbank energies
    based on the specified segments

    Returns:
      mel: a list of mel spectrogram
      time_vec: a list of time vector
    '''
    # Load wav file
    _y, sr = librosa.load(wav_file, sr=hp.sample_rate)

    # Get non-silence utterance interval
    if phn_file is not None:
        begS, endS = get_utterance_interval(phn_file)
        y = _y[begS:endS]
    else:
        y, _ = librosa.effects.trim(_y)

    # Preemphasis
    y = np.append(y[0], y[1:] - hp.preemphasis * y[:-1])  # ??

    # stft
    linear = librosa.stft(y=y,
                          n_fft=hp.num_freq,
                          hop_length=hp.frame_shift,
                          win_length=hp.frame_width)
    mag = np.abs(linear)  # (1+num_freq//2, time)

    # Mel spectrogram
    mel_basis = librosa.filters.mel(  # (num_mels, 1+num_freq//2)
        hp.sample_rate, hp.num_freq, hp.num_mels)
    mel = np.dot(mel_basis, mag)  # (num_mels, time)

    # Spectrogram length
    length = np.ones_like(mel[0, :]).astype(np.int32)

    # to decibel
    mel = librosa.amplitude_to_db(mel)
    # mag = librosa.amplitude_to_db(mag) # for later purpose

    # Normalize
    mel = mel.T.astype(np.float32)  # (time, num_mels)
    # mag = mag.T.astype(np.float32)  # (time, 1+num_freq//2) # for later purpose

    return mel, time_vec


# Make speaker dictionary
#  - normalize vector length


# Save


# Connect to tensorboard projector
