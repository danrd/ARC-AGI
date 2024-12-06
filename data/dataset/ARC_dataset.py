import os
import json
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from typing import Union, List
from rl.ARC_task import ARCTask, ARCSubtask
from llm.prompts import compose_prompt, prepare_grid_for_prompt, DETAILED_PROMPT, BASE_PROMPT, CONCISE_PROMPT 

class ARCDataset:
    def __init__(self, split:str='full', augmentation:bool=False):
        self.load_dataset(split)
        self.tasks = self.create_tasks(augmentation)
    
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
        self.training_challenges = self.load_json('data/dataset/training_challenges.json')
        self.training_solutions = self.load_json('data/dataset/training_solutions.json')
        self.evaluation_challenges = self.load_json('data/dataset/evaluation_challenges.json')
        self.evaluation_solutions = self.load_json('data/dataset/evaluation_solutions.json')
        self.test_challenges = self.load_json('data/dataset/test_challenges.json')
        if split=='full':
            self.tasks_keys = list(self.training_challenges.keys()) + list(self.evaluation_challenges.keys())
            self.training_challenges = self.training_challenges | self.evaluation_challenges
            self.training_solutions = self.training_solutions | self.evaluation_solutions
        elif split=='train':
            self.tasks_keys = list(self.training_challenges.keys())
        else:
            raise ValueError('You need to specify splitting for the dataset with options: train or full')
    
    def create_tasks(self, augmentation):
        """Create a list of tasks for current splitting setting."""
        tasks = []
        aug_tasks = []
        for key in self.tasks_keys:
            train_inp, train_out, test_inp, test_out = self.task_to_lists(key)
            subtasks = []
            aug_subtasks = [[] for _ in range(14)]
            n = len(train_inp)
            for i in range(n):
                label = f'{key}_{i}'
                subtask = ARCSubtask(label, train_inp[i], train_out[i])
                subtasks.append(subtask)
                if augmentation:
                    train_inp_aug_grids = augment_grid(train_inp[i])
                    train_out_aug_grids = augment_grid(train_out[i])
                    for j in range(14):
                        label = f'{key}_{i}_aug_{j}'
                        subtask = ARCSubtask(label, train_inp_aug_grids[j], train_out_aug_grids[j])
                        aug_subtasks[j].append(subtask)
            task = ARCTask(key, subtasks, test_inp, test_out)
            tasks.append(task) 
            if augmentation:
                test_inp_aug_grids = augment_grid(test_inp)
                test_out_aug_grids = augment_grid(test_out)
                for i in range(14):
                    aug_key = f'{key}_aug_{i}'
                    task = ARCTask(aug_key, aug_subtasks[i], test_inp_aug_grids[i], test_out_aug_grids[i])
                    aug_tasks.append(task)
        if augmentation:
            tasks = tasks + aug_tasks
        return tasks
    
def prepare_dataset(augmentation=False, test_augmentation=False, prompts_modifications={}, seed=42):
    """Prepare dataset creating prompts for all tasks."""
    ARC_dataset = ARCDataset(augmentation=augmentation)
    train = []
    test = []
    train_tasks = ARC_dataset.tasks[0:400]
    test_tasks = ARC_dataset.tasks[400:800]
    if augmentation:
        train_tasks += ARC_dataset.tasks[800:6400]
        if test_augmentation:
            test_tasks += ARC_dataset.tasks[6400:12000]
    for train_task in train_tasks:
        train_text = compose_prompt(train_task, BASE_PROMPT, prompts_modifications)
        train_task_dict = {'text':train_text, 'solution':repr(prepare_grid_for_prompt(train_task.test_subtask.train_out, train_task.test_subtask.train_out_shape, concise=False))}
        train.append(train_task_dict)
    for test_task in test_tasks:
        test_text = compose_prompt(test_task, BASE_PROMPT, prompts_modifications)
        test_task_dict = {'text':test_text, 'solution':repr(prepare_grid_for_prompt(test_task.test_subtask.train_out, test_task.test_subtask.train_out_shape, concise=False))}
        test.append(test_task_dict)
    train_df = pd.DataFrame(train)
    train_dataset = Dataset.from_pandas(train_df).shuffle(seed=seed)
    test_df = pd.DataFrame(test)
    test_dataset = Dataset.from_pandas(test_df).shuffle(seed=seed)
    dataset = DatasetDict({'train':train_dataset, 'test':test_dataset})   
    return dataset

def augment_grid(grid:np.array)->List[np.array]:
    new_grids = []
    new_grids.append(np.rot90(grid,k=1))
    new_grids.append(np.rot90(grid,k=2))
    new_grids.append(np.rot90(grid,k=3))
    new_grids.append(np.fliplr(grid))
    new_grids.append(np.flipud(grid))
    for inc in range(1, 10):
        new_grid = grid * 10
        new_grids.append(((new_grid+inc)%10)/10)
    return new_grids