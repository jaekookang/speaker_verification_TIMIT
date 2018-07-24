'''
Analyze PCs from PCA 

2018-07-23
'''

import ipdb as pdb
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.decomposition import PCA

# Load data
DATA_DIR = 'vis_tri_spkr-630'
X = np.load(os.path.join(DATA_DIR, 'x_data.npy'))
meta = np.load(os.path.join(DATA_DIR, 'x_meta.npy'))

# PCA
pca = PCA(n_components=10)
pca.fit(X)
print(pca.explained_variance_ratio_)
