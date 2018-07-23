# Experiment: Vowel MFCC & Speaker variability

- Vowels: /iy/ or /ae/
- MFCCs: 3, 6, 12, 24, 36 
- Speakers: 630

## How to run:

- (1) `python make_spkr_dict.py`
- (2) `python plot_embedding.py embed_test`
- (3) Run tensorboard on the selected log directory

## Exp1) Within-speaker vowel variability

- Factor: Vowels, MFCCs
- Do /ae/ vectors cluster together? high-dim, low-dim
- Do /iy/ vectors cluster together? high-dim, low-dim
- [x] Seperate within-speaker data

## Exp2) Between-speaker vowel variability

- Factor: Vowels, MFCCs

## Exp3) Change of features

- DV: FFT, LPC, LSF
