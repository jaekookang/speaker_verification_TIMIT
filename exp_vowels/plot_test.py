'''
Plot spectrogram (test)

2018-07-21
'''
import ipdb as pdb
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
from make_spkr_dict import hp

# Load directory
SDICT_DIR = 'spkr_dict.npy'
sdict = np.load(SDICT_DIR).item()
spkr = [s for s in sdict.keys()][0]
vowel = 'iy'
slices = [3, 12, 24, 36]

plotdata = sdict[spkr][vowel]

fig, ax = plt.subplots()
ax.imshow(plotdata, aspect='auto')
ax.set_title(f'Speaker: {spkr}, Vowel: {vowel}')
ymin, ymax = ax.get_ylim()
for s in slices:
    ax.plot([s - 0.6, s - 0.6], [ymin, ymax], 'r')
    ax.text(s - 0.6, ymax + 0.3, f'{s}', path_effects=[path_effects.Stroke(
        linewidth=2, foreground='white'), path_effects.Normal()])
ax.set_xlabel(f'Mel coefficients (={hp.n_mfcc})')
ax.set_ylabel('Number of samples')
ax.set_xticks([])
fig.savefig(f'spectrogram/{spkr}_{vowel}_samples.png')
plt.show()
