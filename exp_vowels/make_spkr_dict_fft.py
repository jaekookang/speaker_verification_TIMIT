'''
Exp1. Test within-speaker variability

2018-07-20
2018-08-09 make fft (512-d) and fft (40-d)

'''
import ipdb as pdb
import os
import sys
import re
import textgrid
import glob
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from multiprocessing import Pool
from speech_features import SpeechFeatures
from sklearn.decomposition import PCA


class hparams:
    # Params
    def __init__(self):
        self.mels = 40
        self.num_freq = 1025  # =nfft
        self.sample_rate = 16000
        self.win_size = 0.02
        self.win_step = 0.01
        self.preemphasis = 0.97
        self.ndct = 36  # =numcep, num mfcc coeffs
        self.nfilt = 40


# Get parameters
hp = hparams()


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


def get_spectrogram(wav_file, phn_file, segment, time='center'):
    '''Returns normalized log mel-filterbank energies
    based on the specified segment (eg. 'iy' as string)
    mels_amp, mels_db, mags_amp, mags_db, mfcc_all

    Returns (example):
      mels_amp: np.array of mel amplitude
      mels_db: amplitude_to_db(mels_amp)
      mags_amp: np.array of FFT amplitude
      mags_db: amplitude_to_db(mags_amp)
      mfcc_all: np.array of mfcc 
      ctx: phone labels (triphone); eg. ['h#_sh_iy', ...]
    '''
    # Initialize SpeechFeatures
    S = SpeechFeatures(wav_file, hp.win_size, hp.win_step, hp.num_freq,
                       hp.nfilt, hp.ndct, win_fun=np.hamming, pre_emp=hp.preemphasis)
    # # Load wav file
    # _y, sr = librosa.load(wav_file, sr=hp.sample_rate)

    # Load phn file
    labels, samples = read_phn(phn_file)

    # Iterate over provided segments
    fft_all = np.array([], dtype=np.float32).reshape(
        0, hp.num_freq//2+1)  # (,513)
    idx, _ = find_elements(segment, labels)
    if len(idx) > 0:
        ctx = []
        for i in idx:
            # Audio processing
            # Extract sample
            begT, endT = samples[i, 0], samples[i, 1]
            sig_part = S.sig[begT:endT]
            # Get magnitude spectrogram (db, =log spectrogram)
            _, powspec, _ = S.get_fft(sig_part)
            # Slice sample (at mid point; TODO: add frame choice)
            if time == 'center':
                fft = powspec[powspec.shape[0]//2, :]
            elif time == 'all':
                fft = powspec
            else:
                raise Exception(f'time={time} is not supported yet')
            # Add mel spectrogram
            fft_all = np.vstack([fft_all, fft])
            # Get context
            if (i-1) < 0:
                pre = '#'
            else:
                pre = labels[i-1]
            if (i+1) > len(labels):
                post = '#'
            else:
                post = labels[i+1]
            for _ in range(fft.shape[0]):
                ctx.append('_'.join([pre, labels[i], post]))
    else:
        return None
    return fft_all, ctx


def get_pca(data, n_comp):
    '''Reduce the dimension of the data into n_comp
    (n x k) --> (n x n_comp)
    '''
    pca = PCA(n_components=n_comp)
    return pca.fit_transform(data)


if __name__ == '__main__':
    # Get directories
    if sys.platform == 'darwin':
        TMT_DIR = '/Volumes/Transcend/_DataArchive/TMT'
    else:
        TMT_DIR = '../data/TMT'
    TRAIN_DIR = os.path.join(TMT_DIR, 'TRAIN')
    TEST_DIR = os.path.join(TMT_DIR, 'TEST')
    SPKR_INFO = '../data/spkr_info.txt'
    S = pd.read_table(SPKR_INFO, sep=',', na_filter=False)

    # Get files
    if sys.platform == 'darwin':
        wav_ext = '*.wav'
    elif sys.platform == 'linux':
        wav_ext = '*.WAV'
    else:
        raise Exception(
            f'OS should be either darwin of linux, not {sys.platform}')
    wavs = sorted(glob.glob(
        os.path.join(TMT_DIR, '**', '**', '**', wav_ext)))
    assert len(wavs) > 0

    # Speaker list
    spkrs = S.ID.unique().tolist()  # eg. JMI0
    # phones = ['iy', 'ae', 'aa', 'uh',
    #           'b', 'd', 'g', 'p', 't', 'k',
    #           's', 'z', 'sh', 'dh', 'f', 'v',
    #           'jh', 'ch']
    phones = ['iy', 'aa', 'uh', 's', 'f']

    # Make speaker dictionary
    init_fft = np.array([], dtype=np.float32).reshape(0, hp.num_freq//2+1)
    init_fftpca = np.array([], dtype=np.float32).reshape(0, hp.nfilt)
    # fft dictionary
    fft_d = {s: {v: init_fft for v in phones} for s in spkrs}
    fftall = np.array([], dtype=np.float32).reshape(0, hp.num_freq//2+1)
    # mfcc dictionary
    fftpca_d = {s: {v: init_fftpca for v in phones} for s in spkrs}
    # context dictionary
    cdict = {s: {v: [] for v in phones} for s in spkrs}

    for i, wav in enumerate(tqdm(wavs)):
        # eg. [FM] + JMI0
        spkr_id = re.search('DR[0-9]/(\w+\d)/', wav).group(1)
        phn = re.sub('wav|WAV', 'PHN', wav)
        spkrs = []
        phns = []
        for v in phones:
            out = get_spectrogram(wav, phn, v, time='all')
            if out is not None:
                fft, ctx = out
                if fft.shape[0] != len(ctx):
                    raise Exception(
                        f'fft({fft.shape[0]}) and ctx({len(ctx)}) have different lengths')
                # fft dictionary
                _data = fft_d[spkr_id[1:]][v]
                fft_d[spkr_id[1:]][v] = np.vstack([_data, fft])
                fftall = np.vstack([fftall, fft])
                # context dictionary
                cdict[spkr_id[1:]][v] += ctx
                # save lists of speaker and phones
                spkrs += [spkr_id[1:]]*fft.shape[0]
                phns += [v]*fft.shape[0]
        if (i+1) % 100 == 0:
            print(f'{i+1}/{len(wavs)}')

    # Redistribute data into dictionary
    fftpca = get_pca(fftall, n_comp=hp.nfilt)
    for s, v in tqdm(zip(spkrs, phns)):
        fftpca_d[s][v] = np.vstack([fftpca_d[s][v], fftpca[0]])
    # Save
    np.save('spkr_sdict_fft.npy', fft_d)
    np.save('spkr_sdict_fftpca.npy', fftpca_d)
    np.save('spkr_cdict_fft.npy', cdict)
    print('Finished')
