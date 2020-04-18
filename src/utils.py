# coding: utf8

import numpy as np
import pandas as pd
import pickle

def dump_pickle(data, name):
    with open(name, 'wb') as file_features:
        pickle.dump(data, file_features, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(name):
    with open(name, 'rb') as data:
        return pickle.load(data)

        
