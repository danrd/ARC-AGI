from typing import List
import numpy as np
from symbolic.utils import adjust_grid_shape, check_grid_values

class ARCSubtask:
    def __init__(self, label:str, train_inp:np.array, train_out:np.array, validate:bool=False):
        self.label = label
        self.train_inp = adjust_grid_shape(train_inp)
        self.train_out = adjust_grid_shape(train_out)
        self.train_inp_shape = train_inp.shape
        self.train_out_shape = train_out.shape
        self.prev_grid_size = 0
        self.max_int = 0
        assert (self.train_inp.shape==(30,30) and self.train_out.shape==(30,30)), f"Grids shapes are not (30,30), instead: {self.train_inp.shape}, {self.train_out.shape}"
        if validate:
            assert (check_grid_values(self.train_inp)==True) and (check_grid_values(self.train_out)==True), f"Invalid grid values for subtask {self.label}\n Input grid: {self.train_inp}\n Output grid: {self.train_out}" 
        
class ARCTask:
    """Class for storing information for a task."""
    def __init__(self, label:str, subtasks:List[ARCSubtask], test_inp:np.array, test_out:np.array, validate:bool=False):
        self.label = label
        self.subtasks = subtasks
        self.test_inp = test_inp
        self.test_out = test_out
        self.test_subtask =  ARCSubtask(f'{label}_test', self.test_inp, self.test_out, validate)