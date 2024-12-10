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

class TaskIterator:
    def __init__(self, start=0, end=0, tasks_keys=False):
        self.current = start
        self.end = end
        self.tasks_keys = tasks_keys
        if tasks_keys:
            self.end = len(tasks_keys)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current < self.end:
            if self.tasks_keys:
                value = self.tasks_keys[self.current]
            else:
                value = self.current
            self.current += 1
            return value
        else:
            raise StopIteration