import os
from time import strftime, gmtime


def safe_mkdir(folder):
    '''Make directory if not exists'''
    if os.path.exists(folder):
        os.mkdir(folder)
