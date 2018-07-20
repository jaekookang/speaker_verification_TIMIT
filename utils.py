import os
import re
import shutil
import logging
import numpy as np
from PIL import Image
from time import strftime, gmtime
import pandas as pd


def safe_mkdir(folder):
    '''Make directory if not exists'''
    if not os.path.exists(folder):
        os.mkdir(folder)


def safe_rmdir(folder):
    '''Remove directory if exists'''
    if os.path.exists(folder):
        shutil.rmtree(folder)


def get_time():
    '''Get current time'''
    return strftime('%Y-%m-%d_%H_%M_%S', gmtime())


def set_logger(save_dir):
    # Set logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)  # logging.DEBUG
    # create a file handler
    handler = logging.FileHandler(os.path.join(save_dir, 'info.log'))
    handler.setLevel(logging.INFO)  # logging.DEBUG
    # create a logging format
    formatter = logging.Formatter(
        '%(asctime)s - %(relativeCreated)d - %(message)s')
    handler.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(handler)
    return logger


def find_elements(pattern, my_list):
    '''Find elements in a list'''
    elements = []
    index = []

    for i, l in enumerate(my_list):
        if re.search(pattern, l):
            elements.append(my_list[i])
            index.append(i)
    return index, elements


def pad(array, ref_shape):
    '''Append zeros at the end
    array: target (2-d) np.array
    ref_shape: shape of the reference array; e.g., (20, 30)
    '''
    result = np.zeros(ref_shape)
    return result[:array.shape[0], :array.shape[1]]


def make_image(array, save_dir=None, name=None):
    '''Make numpy array into .png image'''
    # Scale values in the range of 0 ~ 1
    scaled = array - array.min()/(array - array.min()).max()
    img = Image.fromarray(np.uint8(scaled*255), 'L')
    if save_dir is not None:
        if name is not None:
            img.save(os.path.join(save_dir, name+'.png'))
        else:
            img.save(os.path.join(save_dir, 'test.png'))
    else:
        return img


def write_spkr_meta(spkr_id, save_dir, spkr_file='data/spkr_info.txt'):
    '''Write speaker meta file
    and return meta file path'''
    df = pd.read_table(spkr_file, sep=',')
    sex = df.Sex[df.ID.isin(spkr_id)].tolist()
    region = df.DR[df.ID.isin(spkr_id)].tolist()

    # Write meta file
    meta_dir = os.path.join(save_dir, 'meta.tsv')
    with open(meta_dir, 'w') as f:
        for s, x, r in zip(spkr_id, sex, region):
            f.write(f'{s}_{x}_{r}\n')
    return meta_dir
