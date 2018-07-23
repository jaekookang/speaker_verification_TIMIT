'''
Plot Mel spectrogram of original data

2018-07-22
'''

import ipdb as pdb
import os
import re
import sys
sys.path.append('../')
from preprocess_timit import plot_mel_specgram
from make_spkr_dict import get_spectrogram, find_elements, hparams
import matplotlib.pyplot as plt
import glob
import numpy as np
import pandas as pd
import librosa
import librosa.display as display

if __name__ == '__main__':
    # Load speaker info
    TMT_DIR = '/Volumes/Transcend/_DataArchive/TMT'
    SDICT_DIR = 'spkr_dict.npy'
    SAVE_DIR = 'spectrogram'
    sdict = np.load(SDICT_DIR).item()
    S = pd.read_table('../data/spkr_info.txt', sep=',', na_filter=False)
    spkrs = [*sdict]
    spkr = spkrs[0]  # sample one speaker
    vowels = ['iy', 'ae', 'aa']
    vowel = vowels[0]  # choose vowel
    wavs = sorted(glob.glob(os.path.join(TMT_DIR, '**', '**', '**', '*.wav')))
    hp = hparams()

    # Plot mel spectrogram
    _, spkr_wavs = find_elements(spkr, wavs)
    mels_amp_all = np.array([], dtype=np.float32).reshape(0, hp.num_mels)
    mels_db_all = np.array([], dtype=np.float32).reshape(0, hp.num_mels)
    mags_amp_all = np.array([], dtype=np.float32).reshape(
        0, 1 + hp.num_freq // 2)
    mags_db_all = np.array([], dtype=np.float32).reshape(
        0, 1 + hp.num_freq // 2)
    mfcc_all = np.array([], dtype=np.float32).reshape(0, hp.n_mfcc)
    for s in spkr_wavs:
        phn_file = re.sub('.wav', '.PHN', s)
        if get_spectrogram(s, phn_file, segment=vowel, time='all') is not None:
            mels_amp, mels_db, mags_amp, mags_db, mfcc = get_spectrogram(
                s, phn_file, segment=vowel, time='all')
            mels_amp_all = np.vstack([mels_amp_all, mels_amp])
            mels_db_all = np.vstack([mels_db_all, mels_db])
            mags_amp_all = np.vstack([mags_amp_all, mags_amp])
            mags_db_all = np.vstack([mags_db_all, mags_db])
            mfcc_all = np.vstack([mfcc_all, mfcc])

    # # mel (amplitude)
    # plot_mel_specgram(
    #     mels_amp_all, f'Speaker:{spkr}, Vowel:{vowel}', use_agg=False)

    # mel (db)
    plot_mel_specgram(
        mels_db_all, f'Speaker={spkr}_Vowel={vowel}_(Mel_filterbanks_in_db)',
        use_agg=False, save_dir=SAVE_DIR)
    plt.close()
    # # mag (amplitude)
    # plot_mel_specgram(
    #     mags_amp_all, f'Speaker:{spkr}, Vowel:{vowel}', use_agg=False)

    # mel (db)
    plot_mel_specgram(
        mags_db_all, f'Speaker={spkr}_Vowel={vowel}_(FFT_amplitudes_in_db)',
        use_agg=False, save_dir=SAVE_DIR)
    plt.close()
    # mfcc
    plot_mel_specgram(
        mfcc_all, f'Speaker={spkr}_Vowel={vowel}_(MFCC)',
        use_agg=False, save_dir=SAVE_DIR)
    plt.close()

    # display.specshow(mfcc_all.T, x_axis='time')
    # plt.show()
