import json
import os
import random
import numpy as np 
import torch
import pickle

def load_json(file_path):
    """Load JSON data from a file."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def load_pickle(file_path):
    """Load pickle data from a file."""
    with open(file_path, 'r') as file:
        data = pickle.load(file)
    return data

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)   
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return