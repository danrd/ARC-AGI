from typing import List
import numpy as np

class ARCSubtask:
    def __init__(self, label:str, train_inp:np.array, train_out:np.array):
        self.label = label
        self.train_inp = train_inp
        self.train_out = train_out
        self.train_inp_shape = train_inp.shape
        self.train_out_shape = train_out.shape
        self.prev_grid_size = 0
        self.max_int = 0    

class ARCTask:
    """Class for storing information for a task."""
    def __init__(self, label:str, subtasks:List[ARCSubtask], test_inp:np.array, test_out:np.array):
        self.label = label
        self.subtasks = subtasks
        self.test_inp = test_inp
        self.test_out = test_out
        self.test_subtask =  ARCSubtask(f'{label}_test', self.test_inp, self.test_out)