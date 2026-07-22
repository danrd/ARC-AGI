import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.utils import load_json
from symbolic.utils import augment_grid
from datasets import Dataset, DatasetDict
from typing import Union, List
from rl.ARC_task import ARCTask, ARCSubtask

class ARCDataset:
    def __init__(self, additional_datasets:bool=False, augmentation:bool=False, 
                 ttt_augmentation:bool=False, filter_tasks:bool=False,
                 ver2:bool=False):
        self.subsets = {}
        self.additional_tasks = []
        self.load_dataset(additional_datasets, filter_tasks)
        self.tasks = self.create_tasks(augmentation)
        self.easy_tasks, self.hard_tasks = self.difficulty_filter()
        if ttt_augmentation:
            self.ttt_tasks = self.ttt_augmentation(augmentation)
        if ver2:
            self.load_ARC2()
        self.tasks.extend(self.additional_tasks)
        self.idx2label = {idx:task.label for idx, task in enumerate(self.tasks)}

    def task_to_lists(self, task_key:str)-> Union[List[np.array], List[np.array], np.array]:
        """Transform dictionary with task data into several lists for convenience."""
        train_inp, train_out, test_inp, test_out = [], [], [], []
        for d in self.training_challenges[task_key]['train']:
            train_inp.append(np.array(d['input']))
            train_out.append(np.array(d['output']))
        for idx, d in enumerate(self.training_challenges[task_key]['test']):
            test_inp.append(np.array(d['input']))
            test_out.append(np.array(self.training_solutions[task_key][idx]))
        return train_inp, train_out, test_inp, test_out
    
    def load_dataset(self, additional_datasets, filter_tasks):
        """
        Load dataset files and set splitting for training.
        Args:
            additional_datasets (Union[List[str], bool]) : If provided - list of additional datasets.
        """    
        self.datasets = {}
        self.training_challenges = load_json('data/datasets/ARC/training_challenges.json')
        self.training_solutions = load_json('data/datasets/ARC/training_solutions.json')
        self.evaluation_challenges = load_json('data/datasets/ARC/evaluation_challenges.json')
        self.evaluation_solutions = load_json('data/datasets/ARC/evaluation_solutions.json')
        self.test_challenges = load_json('data/datasets/ARC/test_challenges.json')
        self.tasks_keys = list(self.training_challenges.keys()) + list(self.evaluation_challenges.keys())
        self.training_challenges = self.training_challenges | self.evaluation_challenges
        self.training_solutions = self.training_solutions | self.evaluation_solutions
        self.task2difficulty = load_json('data/datasets/ARC/task2difficulty.json')
        self.task2dataset = {key : 'arc' for key in self.tasks_keys}
        self.datasets['arc'] = {'challenges':self.training_challenges, 'solutions':self.training_solutions, 'keys': self.tasks_keys}
        self.cur_idx = 800
        self.datasets_idxs = {'arc':(0, 799)}
        if additional_datasets:
            rejected_tasks = load_json('data/additional_datasets/rejected_tasks.json')
            for dataset in additional_datasets:
                dataset_challenges = load_json(f'data/additional_datasets/{dataset}/{dataset}_challenges.json')
                dataset_solutions = load_json(f'data/additional_datasets/{dataset}/{dataset}_solutions.json')
                if filter_tasks and dataset in rejected_tasks.keys():
                   dataset_challenges, dataset_solutions = self.filter_tasks(dataset_challenges, dataset_solutions, rejected_tasks[dataset])
                dataset_tasks_keys = list(dataset_challenges.keys())
                self.datasets_idxs[dataset] = (self.cur_idx, self.cur_idx+len(dataset_tasks_keys)-1)
                self.cur_idx += len(dataset_tasks_keys)
                self.datasets[dataset] = {'challenges':dataset_challenges, 'solutions':dataset_solutions, 'keys':dataset_tasks_keys}
                self.task2dataset |= {key:dataset for key in dataset_tasks_keys}
                self.training_challenges |=  dataset_challenges
                self.training_solutions |= dataset_solutions
                self.tasks_keys.extend(dataset_tasks_keys)
        self.aug_start_idx = self.cur_idx + 400 * 14

    def load_ARC2(self):
        """
        Load dataset files and set splitting for training.
        Args:
            additional_datasets (Union[List[str], bool]) : If provided - list of additional datasets.
        """    
        training_challenges = load_json('data/dataset/ARC2/arc-agi_training_challenges.json')
        training_solutions = load_json('data/dataset/ARC2/arc-agi_training_solutions.json')
        evaluation_challenges = load_json('data/dataset/ARC2/arc-agi_evaluation_challenges.json')
        evaluation_solutions = load_json('data/dataset/ARC2/arc-agi_evaluation_solutions.json')
        training_challenges = training_challenges | evaluation_challenges
        training_solutions = training_solutions | evaluation_solutions
        self.training_challenges =  self.training_challenges | training_challenges
        self.training_solutions = self.training_solutions | training_solutions | evaluation_solutions
        tasks_keys = list(training_challenges.keys())
        tasks = []
        additional_tasks = []
        for idx, key in enumerate(tasks_keys):
            train_inp, train_out, test_inp, test_out = self.task_to_lists(key)
            subtasks = []
            for i in range(len(train_inp)):
                label = f'{key}_{i}'
                subtask = ARCSubtask(label, train_inp[i], train_out[i])
                subtasks.append(subtask)
            for j in range(len(test_inp)):
                if j == 0:
                    task = ARCTask(key, subtasks, test_inp[j], test_out[j])
                    tasks.append(task)
                else:
                    task = ARCTask(f'{key}_{j+1}', subtasks, test_inp[j], test_out[j])
                    additional_tasks.append(task)
            if idx == 999:
                self.subsets['arc2_train_add'] = len(additional_tasks)
                self.additional_tasks.extend(additional_tasks)
                additional_tasks = []
            if idx == 1119:
                self.subsets['arc2_eval_add'] = len(additional_tasks)
                self.additional_tasks.extend(additional_tasks)
                additional_tasks = []  
        self.tasks.extend(tasks)
    
    @staticmethod
    def filter_tasks(challenges, solutions, rejected_tasks):
        exclude_list = []
        for key in challenges.keys():
            if key in rejected_tasks:
               exclude_list.append(key)
        for key in exclude_list:
            del challenges[key]
            del solutions[key]
        return challenges, solutions 

    def create_tasks(self, augmentation):
        """Create a list of tasks for current splitting setting."""
        tasks = []
        additional_tasks = []
        self.aug_tasks = []
        for idx, key in enumerate(self.tasks_keys[0:]):
            train_inp, train_out, test_inp, test_out = self.task_to_lists(key)
            subtasks = []
            for i in range(len(train_inp)):
                label = f'{key}_{i}'
                subtask = ARCSubtask(label, train_inp[i], train_out[i])
                subtasks.append(subtask)
            for j in range(len(test_inp)):
                if j == 0:
                    task = ARCTask(key, subtasks, test_inp[j], test_out[j])
                    tasks.append(task)
                else:
                    task = ARCTask(f'{key}_{j+1}', subtasks, test_inp[j], test_out[j])
                    additional_tasks.append(task)
            if augmentation:
                aug_task = self.augment_task(subtasks, test_inp/10, test_out/10, key)
                self.aug_tasks.extend(aug_task)
            if idx == 399:
                self.subsets['arc1_train_add'] = len(additional_tasks)
                self.additional_tasks.extend(additional_tasks)
                additional_tasks = []
            if idx == 799:
                self.subsets['arc1_eval_add'] = len(additional_tasks)
                self.additional_tasks.extend(additional_tasks)
                additional_tasks = []  
        if augmentation:
            tasks += self.aug_tasks[0:5600] + self.aug_tasks[11200:] # excluding aug tasks for test set 
        return tasks

    def augment_task(self, subtasks:List[ARCSubtask], test_inp:np.array, 
                     test_out:np.array, key:str)->List[List[ARCSubtask]]:
        """Create a list of additional tasks based on subtasks of a given initial task using grid augmentation."""
        aug_subtasks = [[] for _ in range(14)] # as with augmentation we have 14 new grids
        aug_tasks = []
        difficulty = self.task2difficulty[key]
        aug_key = f'{key}_aug'
        for i, subtask in enumerate(subtasks):
            train_inp_aug_grids = augment_grid(subtask.train_inp)
            train_out_aug_grids = augment_grid(subtask.train_out)
            for j in range(14):
                label = f'{aug_key}_{j}'
                new_subtask = ARCSubtask(label, train_inp_aug_grids[j], train_out_aug_grids[j])
                aug_subtasks[j].append(new_subtask)
        test_inp_aug_grids = augment_grid(test_inp)
        test_out_aug_grids = augment_grid(test_out)
        for i in range(14):
            key = f'{aug_key}_{i}'
            self.task2difficulty[key] = difficulty
            task = ARCTask(key, aug_subtasks[i], test_inp_aug_grids[i], test_out_aug_grids[i])
            aug_tasks.append(task)
        return aug_tasks

    def difficulty_filter(self):
        """Split tasks based on their predifined difficulty."""
        easy_tasks = []
        hard_tasks = []
        for task in self.tasks:
            if self.task2difficulty[task.label] == 'easy':
                easy_tasks.append(task)
            else:
                hard_tasks.append(task)
        return easy_tasks, hard_tasks

    def ttt_augmentation(self, augmentation):
        """Create training tasks from test tasks in line with test-time-training (ttt) concept."""
        ttt_tasks = []
        aug_ttt_tasks = []
        for idx, key in enumerate(self.tasks_keys[400:800]):
            ttt_key = f'{key}_ttt'
            train_inp, train_out, test_inp, test_out = self.task_to_lists(key)
            n = len(train_inp)
            test_idx = np.random.randint(0, n) # identify index of test subtask
            # exclude test subtask from train subtasks
            test_inp = train_inp.pop(test_idx) 
            test_out = train_out.pop(test_idx)
            subtasks = []
            for i in range(n-1):
                label = f'{ttt_key}_{i}'
                subtask = ARCSubtask(label, train_inp[i], train_out[i])
                subtasks.append(subtask)
            difficulty = self.task2difficulty[key]
            self.task2difficulty[ttt_key] = difficulty
            task = ARCTask(ttt_key, subtasks, test_inp, test_out)
            ttt_tasks.append(task) 
            if augmentation:
                aug_task = self.augment_task(subtasks, test_inp/10, test_out/10, ttt_key)
                aug_ttt_tasks.extend(aug_task)
        if augmentation:
            ttt_tasks += aug_ttt_tasks
        return ttt_tasks
    
    def prepare_eval_dataset(self, max_shape:tuple=(15,15)):
        eval_dataset = []
        for task in self.tasks[400:800]:
            shape = task.test_subtask.train_out_shape
            if shape[0] <= max_shape[0] and shape[1] <= max_shape[1]:
                eval_dataset.append(task)
        return eval_dataset    

class CustomCollateFn:
    def __init__(self, tokenizer, eval:bool=True):
        self.tokenizer = tokenizer
        if eval:
            self.tokenizer.padding_side = 'left'

    def __call__(self, batch):
        texts = [item['text'] for item in batch]
        labels = [item['labels'] for item in batch]
        
        model_inputs = self.tokenizer(texts, return_tensors='pt', truncation=False, padding=True)
        max_len = model_inputs['input_ids'][0].shape[-1]
        labels = self.tokenizer(labels, return_tensors='pt', truncation=False, padding='max_length', max_length=max_len).input_ids

        return {
            'input_ids': model_inputs['input_ids'],
            'attention_mask': model_inputs['attention_mask'],
            'labels': labels}
