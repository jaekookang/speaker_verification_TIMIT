'''
Get phone frequency and durational distribution in TIMIT

2018-07-19
'''

import ipdb as pdb
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
import seaborn as sns

# Get directories
TRAIN_DIR = '../TRAIN'
TEST_DIR = '../TEST'
PHONE_DIR = '../info/phone_code.csv'
P = pd.read_csv(PHONE_DIR)

# Get .PHN files
_train = sorted(glob.glob(os.path.join(  # 4620
    TRAIN_DIR, 'DR[0-9]', '[FM]*[0-9]', '*.PHN')))
_test = sorted(glob.glob(os.path.join(  # 1680
    TEST_DIR, 'DR[0-9]', '[FM]*[0-9]', '*.PHN')))


def read_phn(phn_file):
    '''Read .PHN file'''
    beg_time, end_time, phones = [], [], []
    with open(phn_file, 'r') as f:
        lines = f.readlines()
        for l in lines:
            begT, endT, phone = l.strip().split()
            beg_time.append(int(begT))
            end_time.append(int(endT))
            phones.append(phone)
    return beg_time, end_time, phones


# Iterate over .PHN files
duration, phones = [], []
for i, p in enumerate(_train + _test):
    begT, endT, phone = read_phn(p)
    duration += [e - b for e, b in zip(endT, begT)]
    phones += phone

# Get frequency & duration
uq_phones = P.Symbol.tolist()
CNT = {p: 0 for p in uq_phones}
DUR = {p: 0 for p in uq_phones}
for p, d in zip(phones, duration):
    CNT[p] += 1
    DUR[p] += d

# Average duration
for p in uq_phones:
    DUR[p] /= CNT[p]
    DUR[p] /= 16000 / 1000  # sec -> msec

Count = pd.DataFrame({'Count': [CNT[p] for p in uq_phones]})
Duration = pd.DataFrame({'Duration': [DUR[p] for p in uq_phones]})
PP = P.join(Count).join(Duration)
# Save new phone info
PP.to_csv('../info/phone_code_freq_dur.csv')


def draw_freq_dur_plot(category):
    '''Draw frequency and duration plot (png)'''
    # Frequency plot
    ax = sns.barplot(x='Symbol', y='Count',
                     data=PP.loc[PP.Type == category].sort_values(
                         'Count', ascending=False))
    fig = ax.get_figure()
    fig.savefig(f'{category}_freq.png')
    plt.close()
    # Duration plot
    ax = sns.barplot(x='Symbol', y='Duration',
                     data=PP.loc[PP.Type == category].sort_values(
                         'Duration', ascending=False))
    ax.set_ylabel('msec')
    fig = ax.get_figure()
    fig.savefig(f'{category}_dur.png')
    plt.close()


# Vowels
draw_freq_dur_plot('Vowels')
# Stops
draw_freq_dur_plot('Stops')
# Fricatives
draw_freq_dur_plot('Fricatives')
# Affricates
draw_freq_dur_plot('Affricates')
# Others
draw_freq_dur_plot('Others')
