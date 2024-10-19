import os
import json
import numpy as np
from typing import Union, List
from rl.ARC_task import ARCTask, ARCSubtask


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
        self.training_challenges = self.load_json(cwd+'/data/dataset/arc_agi_training_challenges.json')
        self.training_solutions = self.load_json(cwd+'/data/dataset/arc_agi_training_solutions.json')
        self.evaluation_challenges = self.load_json(cwd+'/data/dataset/arc_agi_evaluation_challenges.json')
        self.evaluation_solutions = self.load_json(cwd+'/data/dataset/arc_agi_evaluation_solutions.json')
        self.test_challenges = self.load_json(cwd+'/data/dataset/arc_agi_test_challenges.json')
        if split=='full':
            self.tasks_keys = list(self.training_challenges.keys()) + list(self.evaluation_challenges.keys())
            self.training_challenges = self.training_challenges | self.evaluation_challenges
            self.training_solutions = self.training_solutions | self.evaluation_solutions
        elif split=='train':
            self.tasks_keys = list(self.training_challenges.keys())
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