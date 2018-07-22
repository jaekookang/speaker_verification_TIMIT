'''
Plot spectrogram (test)

2018-07-21
'''
import ipdb as pdb
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np

# Load directory
SDICT_DIR = 'spkr_dict.npy'
sdict = np.load(SDICT_DIR).item()
spkr = [s for s in sdict.keys()][0]
vowel = 'ae'
slices = [3, 6, 9, 12, 24, 36]

plotdata = sdict[spkr][vowel]

fig, ax = plt.subplots()
ax.imshow(plotdata, aspect='auto')
ax.set_title(f'Speaker: {spkr}, Vowel: {vowel}')
ymin, ymax = ax.get_ylim()
for s in slices:
    ax.plot([s, s], [ymin, ymax], 'r')
    ax.text(s, ymax + 0.3, f'{s}', path_effects=[path_effects.Stroke(
        linewidth=2, foreground='white'), path_effects.Normal()])
ax.set_xlabel('Mel coefficients (=40)')
ax.set_ylabel('Number of samples')
plt.show()
