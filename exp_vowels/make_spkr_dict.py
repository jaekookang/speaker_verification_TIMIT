'''
Exp1. Test within-speaker variability

2018-07-20
'''
import ipdb as pdb
import os
import re
import textgrid
import glob
import numpy as np
import pandas as pd
import librosa


class hparams:
    # Params
    def __init__(self):
        self.num_mels = 40
        self.num_freq = 1025
        self.sample_rate = 16000
        self.frame_width = 25
        self.frame_shift = 10
        self.preemphasis = 0.97


def find_elements(pattern, my_list):
    '''Find elements in a list'''
    elements = []
    index = []

    for i, l in enumerate(my_list):
        if re.search(pattern, l):
            elements.append(my_list[i])
            index.append(i)
    return index, elements


def read_phn(phn_file):
    '''Read PHN file
    Returns:
      label: a list of labels
      time: a 2-d numpy array (sample location)
    '''
    with open(phn_file, 'r') as f:
        lines = f.readlines()
        time = np.zeros((len(lines), 2), dtype=np.int32)
        label = []
        for i, l in enumerate(lines):
            _b, _e, _l = l.strip().split()
            time[i] = int(_b), int(_e)
            label.append(_l)
    return label, time


def get_spectrogram(wav_file, phn_file, segment):
    '''Returns normalized log mel-filterbank energies
    based on the specified segment

    Returns:
      mel: a list of mel spectrogram
      time_vec: a list of time vector
    '''
    # Load wav file
    _y, sr = librosa.load(wav_file, sr=hp.sample_rate)

    # Load phn file
    labels, samples = read_phn(phn_file)

    # Iterate over provided segments
    mels = np.array([], dtype=np.float32).reshape(0, hp.num_mels)
    idx, _ = find_elements(segment, labels)
    if len(idx) > 0:
        for i in idx:
            # Extract sample
            begT, endT = samples[i, 0], samples[i, 1]
            y, _ = librosa.effects.trim(_y[begT:endT])
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
            # to decibel
            mel = librosa.amplitude_to_db(mel)
            # mag = librosa.amplitude_to_db(mag) # for later purpose
            mel = mel.T.astype(np.float32)  # (time, num_mels)
            # Slice sample (at mid point; TODO: add frame choice)
            mel_slice = mel[mel.shape[0] // 2, :]
            # Add mel spectrogram
            mels = np.vstack([mels, mel_slice])
        return mels
    else:
        return None


if __name__ == '__main__':
    # Get directories
    TMT_DIR = '/Volumes/Transcend/_DataArchive/TMT'
    TRAIN_DIR = os.path.join(TMT_DIR, 'TRAIN')
    TEST_DIR = os.path.join(TMT_DIR, 'TEST')
    SPKR_INFO = '../data/spkr_info.txt'
    S = pd.read_table(SPKR_INFO, sep=',', na_filter=False)

    # Get files
    _trains = glob.glob(os.path.join(
        TRAIN_DIR, 'DR[0-9]', '[FM]*[0-9]', '*.wav'))
    _tests = glob.glob(os.path.join(
        TEST_DIR, 'DR[0-9]', '[FM]*[0-9]', '*.wav'))
    wavs = sorted(_trains) + sorted(_tests)

    # Get parameters
    hp = hparams()

    # Speaker list
    spkrs = S.ID.unique().tolist()  # eg. JMI0
    vowels = ['iy', 'ae', 'aa']

    # Make speaker dictionary
    #  - normalize vector length
    init = np.array([], dtype=np.float32).reshape(0, 40)
    sdict = {s: {v: init for v in vowels} for s in spkrs}
    for i, wav in enumerate(wavs):
        # eg. [FM] + JMI0
        spkr_id = re.search('DR[0-9]/(\w+\d)/', wav).group(1)
        phn = re.sub('wav', 'PHN', wav)
        for v in vowels:
            _mel = get_spectrogram(wav, phn, v)
            if _mel is not None:
                _data = sdict[spkr_id[1:]][v]
                sdict[spkr_id[1:]][v] = np.vstack([_data, _mel])
        if i % 100 == 0:
            print(f'{i+1}/{len(wavs)}')

    # Save
    np.save('spkr_dict.npy', sdict)
    print('Finished')
