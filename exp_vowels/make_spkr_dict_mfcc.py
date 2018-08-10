'''
Exp1. Test within-speaker variability

2018-07-20
2018-08-01
- Feature extraction method was changed to my code
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


class hparams:
    # Params
    def __init__(self):
        self.num_mels = 40
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

    Returns:
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
    logspec_all = np.array([], dtype=np.float32).reshape(
        0, 1 + hp.num_freq // 2)  # (,513)
    fftdct_all = np.array([], dtype=np.float32).reshape(
        0, hp.ndct)  # (,36)
    mfcc_all = np.array([], dtype=np.float32).reshape(0, hp.ndct)  # (,36)
    idx, _ = find_elements(segment, labels)
    if len(idx) > 0:
        # Get context
        ctx = []
        for i in idx:
            if (i-1) < 0:
                pre = '#'
            else:
                pre = labels[i-1]
            if (i+1) > len(labels):
                post = '#'
            else:
                post = labels[i+1]
            ctx.append('_'.join([pre, labels[i], post]))

        # Audio processing
        for i in idx:
            # Extract sample
            begT, endT = samples[i, 0], samples[i, 1]
            sig_part = S.sig[begT:endT]
            # Get magnitude spectrogram (db, =log spectrogram)
            magspec, powspec, logspec = S.get_fft(sig_part)
            # Get dct of logspec
            fftdct = S.apply_dct(logspec)
            # Get MFCCs
            mfcc = S.get_mfcc(powspec)
            # Slice sample (at mid point; TODO: add frame choice)
            if time == 'center':
                logspec = logspec[logspec.shape[0]//2, :]
                fftdct = fftdct[fftdct.shape[0]//2, :]
                mfcc = mfcc[mfcc.shape[0]//2, :]
            elif time == 'all':
                pass
            else:
                raise Exception(f'time={time} is not supported yet')
            # Add mel spectrogram
            logspec_all = np.vstack([logspec_all, logspec])
            fftdct_all = np.vstack([fftdct_all, fftdct])
            mfcc_all = np.vstack([mfcc_all, mfcc])
    else:
        return None
    return (logspec_all, fftdct_all, mfcc_all, ctx)


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
    init_spec = np.array([], dtype=np.float32).reshape(0, 1 + hp.num_freq//2)
    init_dct = np.array([], dtype=np.float32).reshape(0, hp.ndct)
    # logspec dictionary
    logspec_d = {s: {v: init_spec for v in phones} for s in spkrs}
    # fftdct dictionary
    fftdct_d = {s: {v: init_dct for v in phones} for s in spkrs}
    # mfcc dictionary
    mfcc_d = {s: {v: init_dct for v in phones} for s in spkrs}
    # context dictionary
    cdict = {s: {v: [] for v in phones} for s in spkrs}

    for i, wav in enumerate(tqdm(wavs)):
        # eg. [FM] + JMI0
        spkr_id = re.search('DR[0-9]/(\w+\d)/', wav).group(1)
        phn = re.sub('wav|WAV', 'PHN', wav)
        for v in phones:
            out = get_spectrogram(wav, phn, v, time='all')
            if out is not None:
                logspec, fftdct, mfcc, ctx = out
                # logspec dictionary
                _data = logspec_d[spkr_id[1:]][v]
                logspec_d[spkr_id[1:]][v] = np.vstack([_data, logspec])
                # fftdct dictionary
                _data = fftdct_d[spkr_id[1:]][v]
                fftdct_d[spkr_id[1:]][v] = np.vstack([_data, fftdct])
                # mfcc dictionary
                _data = mfcc_d[spkr_id[1:]][v]
                mfcc_d[spkr_id[1:]][v] = np.vstack([_data, mfcc])
                # context dictionary
                cdict[spkr_id[1:]][v] += ctx
        if (i+1) % 100 == 0:
            print(f'{i+1}/{len(wavs)}')

    # Save
    np.save('spkr_sdict_logspec_all.npy', logspec_d)
    np.save('spkr_sdict_fftdct_all.npy', fftdct_d)
    np.save('spkr_sdict_mfcc_all.npy', mfcc_d)
    np.save('spkr_cdict_all.npy', cdict)
    print('Finished')
