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
from llm.prompts import compose_prompt, prepare_grid_for_prompt, DETAILED_PROMPT, BASE_PROMPT, CONCISE_PROMPT 

class ARCDataset:
    def __init__(self, additional_datasets:bool=['mini_arc', 're_arc', 'synth_arc', 'concept_arc', 
                                                'pqa_arc', 'so_arc', 'dbigham_arc', 'ns_arc',
                                                'tama_arc', 'com_arc'], 
                 augmentation:bool=False, ttt_augmentation:bool=False, filter_tasks:bool=False, validate:bool=False):
        self.validate = validate
        self.load_dataset(additional_datasets, filter_tasks)
        self.tasks = self.create_tasks(augmentation)
        self.easy_tasks, self.hard_tasks = self.difficulty_filter()
        if ttt_augmentation:
            self.ttt_tasks = self.ttt_augmentation(augmentation)
    
    def task_to_lists(self, task_key:str)-> Union[List[np.array], List[np.array], np.array]:
        """Transform dictionary with task data into several lists for convenience."""
        train_inp = []
        train_out = []
        for d in self.training_challenges[task_key]['train']:
            train_inp.append(np.array(d['input']))
            train_out.append(np.array(d['output']))
        test_inp = np.array(self.training_challenges[task_key]['test'][0]['input'])
        test_out = np.array(self.training_solutions[task_key][0])
        return train_inp, train_out, test_inp, test_out
    
    def load_dataset(self, additional_datasets, filter_tasks):
        """Load dataset files and set splitting for training.
        Parameters
        ----------
        additional_datasets : Union[List[str], bool]
            If provided - list of additional datasets.
        """    
        self.datasets = {}
        self.training_challenges = load_json('data/dataset/training_challenges.json')
        self.training_solutions = load_json('data/dataset/training_solutions.json')
        self.evaluation_challenges = load_json('data/dataset/evaluation_challenges.json')
        self.evaluation_solutions = load_json('data/dataset/evaluation_solutions.json')
        self.test_challenges = load_json('data/dataset/test_challenges.json')
        self.tasks_keys = list(self.training_challenges.keys()) + list(self.evaluation_challenges.keys())
        self.training_challenges = self.training_challenges | self.evaluation_challenges
        self.training_solutions = self.training_solutions | self.evaluation_solutions
        self.task2difficulty = load_json('data/dataset/task2difficulty.json')
        self.task2dataset = {key : 'arc' for key in self.tasks_keys}
        self.datasets['arc'] = {'challenges':self.training_challenges, 'solutions':self.training_solutions, 'keys': self.tasks_keys}
        if additional_datasets:
            rejected_tasks = load_json('data/additional_datasets/rejected_tasks.json')
            for dataset in additional_datasets:
                dataset_challenges = load_json(f'data/additional_datasets/{dataset}/{dataset}_challenges.json')
                dataset_solutions = load_json(f'data/additional_datasets/{dataset}/{dataset}_solutions.json')
                if filter_tasks and dataset in rejected_tasks.keys():
                   dataset_challenges, dataset_solutions = self.filter_tasks(dataset_challenges, dataset_solutions, rejected_tasks[dataset])
                dataset_tasks_keys = list(dataset_challenges.keys())
                self.datasets[dataset] = {'challenges':dataset_challenges, 'solutions':dataset_solutions, 'keys':dataset_tasks_keys}
                self.task2dataset |= {key : dataset for key in dataset_tasks_keys}
                self.training_challenges |=  dataset_challenges
                self.training_solutions |= dataset_solutions
                self.tasks_keys.extend(dataset_tasks_keys)
    
    @staticmethod
    def filter_tasks(challenges, solutions, rejected_tasks):
        for key in challenges.keys():
            if key in rejected_tasks:
               del challenges[key]
               del solutions[key]
        return challenges, solutions 

    def create_tasks(self, augmentation):
        """Create a list of tasks for current splitting setting."""
        tasks = []
        self.aug_tasks = []
        for idx, key in enumerate(self.tasks_keys[0:]):
            train_inp, train_out, test_inp, test_out = self.task_to_lists(key)
            subtasks = []
            for i in range(len(train_inp)):
                label = f'{key}_{i}'
                subtask = ARCSubtask(label, train_inp[i], train_out[i], self.validate)
                subtasks.append(subtask)
            task = ARCTask(key, subtasks, test_inp, test_out, self.validate)
            tasks.append(task) 
            if augmentation:
                aug_task = self.augment_task(subtasks, test_inp/10, test_out/10, key)
                self.aug_tasks.extend(aug_task)
        if augmentation:
            tasks += self.aug_tasks[0:5600] + self.aug_tasks[11200:] # excluding aug tasks for test set 
        return tasks

    def augment_task(self, subtasks:List[ARCSubtask], test_inp:np.array, 
                     test_out:np.array, key:str)->List[List[ARCSubtask]]:
        """Create a list of additional tasks based on subtasks of a given initial task using grid augmentation"""
        aug_subtasks = [[] for _ in range(14)] # as with augmentation we have 14 new grids
        aug_tasks = []
        difficulty = self.task2difficulty[key]
        aug_key = f'{key}_aug'
        for i, subtask in enumerate(subtasks):
            train_inp_aug_grids = augment_grid(subtask.train_inp)
            train_out_aug_grids = augment_grid(subtask.train_out)
            for j in range(14):
                label = f'{aug_key}_{j}'
                new_subtask = ARCSubtask(label, train_inp_aug_grids[j], train_out_aug_grids[j], self.validate)
                aug_subtasks[j].append(new_subtask)
        test_inp_aug_grids = augment_grid(test_inp)
        test_out_aug_grids = augment_grid(test_out)
        for i in range(14):
            key = f'{aug_key}_{i}'
            self.task2difficulty[key] = difficulty
            task = ARCTask(key, aug_subtasks[i], test_inp_aug_grids[i], test_out_aug_grids[i], self.validate)
            aug_tasks.append(task)
        return aug_tasks

    def difficulty_filter(self):
        """Split tasks based on their predifined difficulty"""
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
                subtask = ARCSubtask(label, train_inp[i], train_out[i], self.validate)
                subtasks.append(subtask)
            difficulty = self.task2difficulty[key]
            self.task2difficulty[ttt_key] = difficulty
            task = ARCTask(ttt_key, subtasks, test_inp, test_out, self.validate)
            ttt_tasks.append(task) 
            if augmentation:
                aug_task = self.augment_task(subtasks, test_inp/10, test_out/10, key)
                aug_ttt_tasks.extend(aug_task)
        if augmentation:
            ttt_tasks += aug_ttt_tasks
        return ttt_tasks
    
def prepare_dataset(tokenizer,
                    additional_datasets=['mini_arc', 're_arc', 'synth_arc', 'concept_arc', 
                                         'pqa_arc', 'so_arc', 'dbigham_arc', 'ns_arc',
                                         'tama_arc', 'com_arc'], 
                    augmentation=False, ttt_augmentation=False, 
                    prompts_modifications={}, max_tokens:int=None,
                    cur_learning:bool=False, seed:int=42,
                    grid_repr_type:str='ascii', filter_tasks:bool=False,
                    validate:bool=False):
    """Prepare dataset creating prompts for all tasks."""
    ARC_dataset = ARCDataset(additional_datasets, augmentation, ttt_augmentation, filter_tasks, validate)
    train_set_easy = []
    train_set_hard = []
    test_set = []
    train_tasks_easy = ARC_dataset.easy_tasks 
    train_tasks_hard = ARC_dataset.hard_tasks[400:]
    test_tasks = ARC_dataset.hard_tasks[0:400]
    rejected_train = 0
    rejected_test = 0
    if ttt_augmentation: 
        train_tasks_hard += ARC_dataset.ttt_tasks
    for train_task in tqdm(train_tasks_easy):
        train_text_easy = compose_prompt(train_task, BASE_PROMPT, prompts_modifications, tokenizer, max_tokens, grid_repr_type)
        if train_text_easy:
            train_task_dict_easy = {'text':train_text_easy, 
                                    'label_ids':repr(prepare_grid_for_prompt(train_task.test_subtask.train_out, train_task.test_subtask.train_out_shape, grid_repr_type))}
            train_set_easy.append(train_task_dict_easy)
        else:
            rejected_train += 1
    for train_task in tqdm(train_tasks_hard):
        train_text_hard = compose_prompt(train_task, BASE_PROMPT, prompts_modifications, tokenizer, max_tokens, grid_repr_type)
        if train_text_hard:
            train_task_dict_hard = {'text':train_text_hard, 
                                    'label_ids':repr(prepare_grid_for_prompt(train_task.test_subtask.train_out, train_task.test_subtask.train_out_shape, grid_repr_type))}
            train_set_hard.append(train_task_dict_hard)   
        else:
            rejected_train += 1
    for test_task in tqdm(test_tasks):
        test_text = compose_prompt(test_task, BASE_PROMPT, prompts_modifications, tokenizer, max_tokens, grid_repr_type)
        if test_text:
            test_task_dict = {'text':test_text, 'label_ids':repr(prepare_grid_for_prompt(test_task.test_subtask.train_out, test_task.test_subtask.train_out_shape, grid_repr_type))}
            test_set.append(test_task_dict)
        else:
            rejected_test += 1 
    train_df_easy = pd.DataFrame(train_set_easy).sample(frac=1)
    train_df_hard = pd.DataFrame(train_set_hard).sample(frac=1)
    test_df = pd.DataFrame(test_set).sample(frac=1).reset_index(drop=True)
    train_df = pd.concat([train_df_easy, train_df_hard], ignore_index = True).reset_index(drop=True)
    train_dataset = Dataset.from_pandas(train_df)
    if not cur_learning:
        train_dataset.shuffle(seed=seed)
    test_dataset = Dataset.from_pandas(test_df).shuffle(seed=seed)
    dataset = DatasetDict({'train':train_dataset, 'test':test_dataset}) 
    print(f"Train set: {len(dataset['train'])} examples\n Test set: {len(dataset['test'])} examples\n")
    print(f"Number of train filtered out examples: {rejected_train}\nNumber of test filtered out examples: {rejected_test}")  
    return dataset