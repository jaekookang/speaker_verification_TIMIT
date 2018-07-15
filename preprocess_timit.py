'''
Preprocess TIMIT dataset

2018-07-06

Generates .npy files for each .wav file
and saves them under each speaker folder

Original train/test data will be combined together to make
new train/test dataset. Speakers in the train data should also
exist in test data. That's why the entire dataset was combined
and then re-divided.

ref:
- https://www.github.com/kyubyong/deepvoice3
'''
import ipdb as pdb
import os
import re
import glob
import tqdm
import random
import numpy as np
import librosa

from utils import safe_mkdir, safe_rmdir, find_elements
from hparams import hparams as hp


def get_utterance_interval(phn_file):
    '''Returns nth sample at beginning and ending excluding 'h#'
    Arguments:
      phn_file: a .PHN file (full path)

    Returns:
      begS: nth sample after the first h#
      endS: nth sample before the last h#
    '''
    with open(phn_file, 'r') as f:
        lines = f.readlines()
    if re.search('h#', lines[0]):
        begS = int(lines[0].split()[1])  # ['0', '3050', 'h#']
    else:
        begS = 0
    if re.search('h#', lines[-1]):
        endS = int(lines[-1].split()[0])  # ['44586', '46720', 'h#']
    else:
        endS = int(lines[-1].split()[1])
    return begS, endS


def get_spectrogram(wav_file, phn_file=None):
    '''Returns normalized log mel-filterbank energies with length
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
    mag = librosa.amplitude_to_db(mag)

    # Normalize
    mel = mel.T.astype(np.float32)  # (time, num_mels)
    mag = mag.T.astype(np.float32)  # (time, 1+num_freq//2)
    return mel, length, mag


def plot_mel_specgram(mel, fid, save_dir=None):
    '''Plot mel spectrogram'''
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(mel.T, aspect='auto', origin='upper')
    ax.set_title(fid)
    ax.set_xlabel('frames')
    ax.set_ylabel('Mel coefficients')
    ax.set_xticks([])
    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, f'{fid}.png'))
        print(os.path.join(save_dir, f'{fid}.png'))
    plt.close()


def create_spkr_folder(data_dir, spkr_list):
    '''Make speaker folder
    e.g., ./data/train/FAEM0
    '''
    for s in spkr_list:
        folder = os.path.join(data_dir, s)
        safe_mkdir(folder)


def divide_train_test():
    '''Divide train test data

    Returns:
      spkr_list, train_list, test_list

    Dictionaries (spkr2idx.npy, idx2spkr.npy) will be created
    under ./data
    '''
    # Get data list
    train_utts = 9  # total 10 utterances per speaker
    data_list = sorted(glob.glob(os.path.join(
        hp.timit_dir, 'T[RE]*', 'DR[0-9]', '[FM]*', '*.WAV')))
    spkr_list = [d.split('/')[-2] for d in data_list]
    spkr_list = list(set(spkr_list))

    # Save dictionaries
    spkr2idx = {s: i for i, s in enumerate(spkr_list)}
    idx2spkr = {i: s for i, s in enumerate(spkr_list)}
    np.save(os.path.join(hp.data_dir, 'spkr2idx.npy'), spkr2idx)
    np.save(os.path.join(hp.data_dir, 'idx2spkr.npy'), idx2spkr)

    # Assign train/test data
    train_list, test_list = [], []
    for s in spkr_list:
        files = find_elements(s, data_list)
        random.shuffle(files)
        train_list += files[:train_utts]
        test_list += files[train_utts:]
    print(f'Train utterances: {len(train_list)}')
    print(f'Test utterances: {len(test_list)}')
    return spkr_list, train_list, test_list


if __name__ == '__main__':
    # Divide train/test
    spkr_list, train_list, test_list = divide_train_test()

    # Create speaker folders
    safe_rmdir(hp.train_dir)
    safe_rmdir(hp.test_dir)
    safe_mkdir(hp.train_dir)
    safe_mkdir(hp.test_dir)
    create_spkr_folder(hp.train_dir, spkr_list)
    create_spkr_folder(hp.test_dir, spkr_list)

    print('Convert train data')
    for f in tqdm.tqdm(train_list):
        path, fid = os.path.split(f)
        mel_dir = os.path.join(hp.train_dir, re.sub('.*/', '', path), 'mel')
        len_dir = os.path.join(hp.train_dir, re.sub('.*/', '', path), 'len')
        safe_mkdir(mel_dir)
        safe_mkdir(len_dir)
        phn_file = os.path.join(path, fid.replace('.WAV', '.PHN'))
        mel, nframe, _ = get_spectrogram(f, phn_file)  # (num_mels, time)
        # Save
        np.save(os.path.join(mel_dir, fid.replace('.WAV', '.npy')), mel)
        np.save(os.path.join(len_dir, fid.replace('.WAV', '.npy')), nframe)
        # plot_mel_specgram(mel, f, hp.data_dir)

    print('Convert test data')
    for f in tqdm.tqdm(test_list):
        path, fid = os.path.split(f)
        mel_dir = os.path.join(hp.test_dir, re.sub('.*/', '', path), 'mel')
        len_dir = os.path.join(hp.test_dir, re.sub('.*/', '', path), 'len')
        safe_mkdir(mel_dir)
        safe_mkdir(len_dir)
        phn_file = os.path.join(path, fid.replace('.WAV', '.PHN'))
        mel, nframe, _ = get_spectrogram(f, phn_file)  # (num_mels, time)
        # Save
        np.save(os.path.join(mel_dir, fid.replace('.WAV', '.npy')), mel)
        np.save(os.path.join(len_dir, fid.replace('.WAV', '.npy')), nframe)
