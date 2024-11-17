import os
import json
import numpy as np
import pandas as pd
from datasets import Dataset
from typing import Union, List
from rl.ARC_task import ARCTask, ARCSubtask
from llm.prompts import compose_prompt, prepare_grid_for_prompt, DETAILED_PROMPT, BASE_PROMPT, CONCISE_PROMPT 

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
    
def prepare_dataset(test_ratio:float=0.2, use_eval_set:bool=False, prompts_modifications={}):
    """Prepare dataset creating prompts for all tasks."""
    all_tasks = []
    if use_eval_set:
        n_examples = 800
    else: 
       n_examples = 400
    ARC_tasks = ARCDataset().tasks[0:n_examples]
    for _, task in enumerate(ARC_tasks):
        text = compose_prompt(task, BASE_PROMPT, prompts_modifications)
        task_dict = {'text':text, 'solution':repr(prepare_grid_for_prompt(task.test_subtask.train_out, task.test_subtask.train_out_shape))}
        all_tasks.append(task_dict)
    df = pd.DataFrame(all_tasks)
    dataset = Dataset.from_pandas(df)
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.train_test_split(test_size=test_ratio)   
    return dataset