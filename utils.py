import os
import re
import shutil
import logging
from time import strftime, gmtime


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
    out = []
    for i, l in enumerate(my_list):
        if re.search(pattern, l):
            out.append(my_list[i])
    return out
