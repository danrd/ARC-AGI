from typing import Union, List
import numpy as np
import json

class ARCSubtask:
    def __init__(self, label:str, train_inp:np.array, train_out:np.array):
        self.label = label
        self.train_inp = self.adjust_grid_size(train_inp)
        self.train_out = self.adjust_grid_size(train_out)
        self.train_inp_shape = train_inp.shape
        self.train_out_shape = train_out.shape
        self.prev_grid_size = 0
        self.max_int = 0
        self.target_size = self.calculate_relevant_blocks(self.train_out)
        assert (self.train_inp.shape==(30,30) and self.train_out.shape==(30,30)), f"Grids shapes are not (30,30), instead: {self.train_inp.shape}, {self.train_out.shape}"
        
    def adjust_grid_size(self, grid:np.array)->np.array:
        """Transform any grid to max shape (30,30)."""
        shape_x = grid.shape[0]
        shape_y = grid.shape[1]
        if shape_x!=30 or shape_y!=30:
            left_pad = (30-shape_x)//2
            right_pad = 30 - shape_x - left_pad
            upper_pad = (30-shape_y)//2
            down_pad = 30 - shape_y - upper_pad
            reshaped_grid = np.pad(grid, pad_width=[(left_pad,right_pad), (upper_pad, down_pad)], constant_values=10)
            return reshaped_grid/10
        else: 
            return grid   

    def step_intersection(self, grid):
        """
        Calculates the difference between the maximal intersection at previous step and the current one.
        Note that the method updates object fields to save the grid size.
        Parameters
        ----------
        grid : np.array
            Current grid state.
        """   
        max_int = self.maximal_intersection(grid)
        done = max_int == self.target_size
        right_placement = (max_int - self.max_int)
        grid_size = self.calculate_relevant_blocks(grid)
        self.prev_grid_size = grid_size
        self.max_int = max_int
        self.right_placement = right_placement
        return right_placement,  done
    
    def maximal_intersection(self, grid):
        """Calculates the number of common blocks for current grid and target grid."""
        intersection = ((grid==self.train_out) * (grid!=1) * (self.train_out!=1) ).sum()
        max_int = intersection 
        return max_int
    
    def calculate_relevant_blocks(self, grid):
        relevant_size = grid.size - (grid==1).sum().item() - (grid==0).sum().item()
        return relevant_size
    
    def reset(self):
        self.prev_grid_size = 0
        self.max_int = 0
    
class ARCTask:
    def __init__(self, label:str, subtasks:List[ARCSubtask], test_inp:np.array, test_out:np.array):
        self.label = label
        self.subtasks = subtasks
        self.test_inp = test_inp
        self.test_out = test_out
        self.test_out_shape = test_out.shape
        self.test_subtask =  ARCSubtask(f'{label}_test', self.test_inp, self.test_out)

class ARCDataset:
    def __init__(self, split:str='full'):
        self.load_dataset(split)
        self.tasks = self.create_tasks()
    
    @staticmethod
    def load_json(file_path):
        """Load JSON data from a file."""
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    
    def task_to_lists(self, task:dict)-> Union[List[np.array], List[np.array], np.array]:
        """Transform dictionary with task data into several lists for convenience."""
        train_inp = []
        train_out = []
        for d in self.training_challenges[task]['train']:
            train_inp.append(np.array(d['input']))
            train_out.append(np.array(d['output']))
        test_inp = np.array(self.training_challenges[task]['test'][0]['input'])
        test_out = np.array(self.training_solutions[task][0])
        return train_inp, train_out, test_inp, test_out
    
    def load_dataset(self, split:str='full'):
        """Load dataset files and set splitting for training.
        Parameters
        ----------
        split : str
            Possible options are train/full. If full It takes evaluation examples as well.
        """    
        cwd = os.getcwd()
        self.training_challenges = self.load_json(cwd+'/data/dataset/training_challenges.json')
        self.training_solutions = self.load_json(cwd+'/data/dataset/training_solutions.json')
        self.evaluation_challenges = self.load_json(cwd+'/data/dataset/evaluation_challenges.json')
        self.evaluation_solutions = self.load_json(cwd+'/data/dataset/evaluation_solutions.json')
        self.test_challenges = self.load_json(cwd+'/data/dataset/test_challenges.json')
        if split=='full':
            self.tasks_keys = list(training_challenges.keys()) + list(training_challenges.keys())
            self.training_challenges = self.training_challenges | self.evaluation_challenges
            self.training_solutions = self.training_solutions | self.evaluation_solutions
        elif split=='train':
            self.tasks_keys = list(training_challenges.keys())
        else:
            raise ValueError('You need to specify splitting for the dataset with options: train or full')
    
    def create_tasks(self):
        """Create a list of tasks for current splitting setting."""
        tasks = []
        for key in self.tasks_keys:
            train_inp, train_out, test_inp, test_out = self.task_to_lists(key)
            subtasks = []
            n = len(train_inp)
            for i in range(n):
                label = f'{key}_{i}'
                subtask = ARCSubtask(label, train_inp[i], train_out[i])
                subtasks.append(subtask)
            task = ARCTask(key, subtasks, test_inp, test_out)
            tasks.append(task)
        return tasks