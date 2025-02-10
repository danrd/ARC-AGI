import math
import warnings
import os
import gymnasium
from typing import Optional
from gymnasium import spaces
from gym import Env, Wrapper as gymWrapper
from gym.spaces import Dict, Box, Discrete, Space
import numpy as np
from copy import copy, deepcopy
from rl.ARC_world import World, Agent
from rl.ARC_task import ARCSubtask
from rl.utils import repad 
from symbolic.utils import crop_pad

class ARCGridWorld(gymnasium.Env):
    def __init__(
            self, max_steps=1000, right_placement_scale=1.0, wrong_placement_scale=1.0,
            seed=None, font_color=0, padding=False, random_start=False, input_pattern=False,
            milestones_rewards=[1, 2, 3, 4], pad_val=1.0, int_colors=True) -> None:
        self.agent = Agent()
        self.step_no = 0
        self.right_placement_scale = right_placement_scale
        self.wrong_placement_scale = wrong_placement_scale
        self.max_steps = max_steps
        self.right_placement = 0
        self.wrong_placement = 0
        self.prev_action = -1
        self.seed = seed
        self.font_color = font_color
        self.padding = padding
        self.pad_val = pad_val*10 if int_colors else pad_val 
        self.low_val = self.pad_val if self.pad_val < 0 else 0
        self.random_start = random_start
        self.input_pattern = input_pattern
        self.milestones_rewards = milestones_rewards
        self.int_colors = int_colors
        self.grid_dtype = np.float64 if not int_colors else np.int8
        self.action_space = spaces.Discrete(14)
        self.observation_space = {
            'agent_position': spaces.Box(low=0, high=1, shape=(1, 2), dtype=np.float64),
            'grid': spaces.Box(low=self.low_val, high=1, shape=(30, 30), dtype=np.float64)}
        self.observation_space = spaces.Dict(self.observation_space)
        
    def add_block(self, position, color):
        x, y = position
        self.grid[x, y] = color

    def remove_block(self, position):
        x, y = position
        if self.grid[x, y] == self.pad_val:
            raise ValueError(f'Removal of non-existing block. address: x={x}, y={y}')
        self.grid[x, y] = self.font_color
        
    @staticmethod
    def find_upper_left_corner(grid_size:tuple)->tuple:
        """Finds left upper corner of the grid to take into account padding."""
        start_x = (30-grid_size[0])//2
        start_y = (30-grid_size[1])//2
        i = min(start_x-(grid_size[0]%2)*((grid_size[0]//2)), start_x-((grid_size[0]-1)%2)*(((grid_size[0]-1)//2)))
        j = min(start_y-(grid_size[1]%2)*((grid_size[1]//2)), start_y-((grid_size[1]-1)%2)*(((grid_size[1]-1)//2)))
        return (i, j)
    
    def initialize_agent_position(self):
        grid_size = self.subtask.train_out_shape
        position = self.find_upper_left_corner(grid_size) if self.padding else (0, 0)
        if self.random_start:
            i = np.random.randint(low=position[0], high=position[0]+grid_size[0]-1)
            j = np.random.randint(low=position[1], high=position[1]+grid_size[1]-1)
            position = (i, j)
        return position   

    def step_intersection(self, grid:np.array):
        """
        Calculates the difference between the maximal intersection at previous step and the current one.
        Parameters
        ----------
        grid : np.array
            Current grid state.
        """   
        max_int = self.maximal_intersection(grid)
        done = (max_int==self.target_int)
        right_placement = (max_int-self.max_int)
        self.max_int = max_int
        self.right_placement = right_placement
        return right_placement, done
    
    def maximal_intersection(self, grid:np.array):
        """Calculate the number of common blocks for current grid and target grid."""
        intersection = ((np.array(grid)==np.array(self.train_out)) * (grid!=self.pad_val) * (self.train_out!=self.pad_val)).sum()
        return intersection
    
    def calculate_relevant_blocks(self, grid:np.array):
        """Calculate a number of cells that are not white padding."""
        relevant_size = grid.size - (grid==self.pad_val).sum().item()
        return relevant_size

    def set_subtask(self, subtask:ARCSubtask):
        """
        Assigns provided task into the environment. On each .reset, the env
        Queries the .reset method for the task object. This method should drop
        the task state to the initial one.
        Note that the env can only work with non-None task or task generator.
        """
        self.subtask = deepcopy(subtask)
        if self.padding:
            self.subtask = repad(subtask, max_shape=self.padding)
            self.train_inp = self.subtask.train_inp
            self.train_out = self.subtask.train_out
        else:
            self.train_inp = crop_pad(self.subtask.train_inp*10, pad_val=10)/10
            self.train_out = crop_pad(self.subtask.train_out*10, pad_val=10)/10
        self.subtask_label = self.subtask.label
        if self.int_colors:
            self.train_inp = (self.train_inp*10).astype(np.int64)
            self.train_out = (self.train_out*10).astype(np.int64)
        self.reset(seed=self.seed)
        
    def initialize_grid(self, subtask):
        shape_x = self.subtask.train_out_shape[0]      
        shape_y = self.subtask.train_out_shape[1]
        starting_grid = np.zeros(self.subtask.train_out_shape)
        if self.padding:
            if shape_x != self.padding[0] or shape_y != self.padding[1]:
                left_pad = (self.padding[0]-shape_x)//2
                right_pad = self.padding[0] - shape_x - left_pad
                upper_pad = (self.padding[1]-shape_y)//2
                down_pad = self.padding[1] - shape_y - upper_pad
                starting_grid = np.pad(starting_grid, pad_width=[(left_pad,right_pad), (upper_pad, down_pad)], constant_values=self.pad_val)
                assert (starting_grid.shape==self.padding), f"Grids shapes are not {self.padding}, instead: {starting_grid.shape}"
                shape_x, shape_y = self.padding
        self.grid = starting_grid
        self.used_cells = list(zip(np.where(self.train_out==self.font_color)[0], np.where(self.train_out==self.font_color)[1]))
        self.world = World(build_zone=(shape_x, shape_y), font_color=self.font_color)
        self.world.forbidden_cells = self.define_forbidden_cells(starting_grid)
        self.observation_space = {
            'agent_position': spaces.Box(low=0, high=1, shape=(1, 2), dtype=np.float32)
        }
        if self.int_colors:
            self.observation_space['grid'] = spaces.Box(low=self.low_val*10, high=10, shape=(shape_x, shape_y), dtype=self.grid_dtype)
        else:
            self.observation_space['grid'] = spaces.Box(low=self.low_val, high=1, shape=(shape_x, shape_y), dtype=self.grid_dtype)
        if self.input_pattern:
            self.observation_space['input_pattern'] = spaces.Box(low=self.low_val, high=1, shape=(shape_x, shape_y), dtype=np.float64)  
        self.observation_space = spaces.Dict(self.observation_space)

    def initialize_targets(self):
        self.target_int = (self.train_out!=self.pad_val).sum() if self.padding else np.size(self.train_out)
        self.max_reward = (self.target_int - self.max_int) * self.right_placement_scale
        milestone_step = (self.target_int - self.max_int) // 4
        self.milestones = {(self.max_int + milestone_step * (step+1)):self.max_reward * self.milestones_rewards[step] for step in range(len(self.milestones_rewards))}
        self.base_int = copy(self.max_int)
    
    def define_forbidden_cells(self, starting_grid):
        return list(zip((starting_grid==self.pad_val).nonzero()[0], (starting_grid==self.pad_val).nonzero()[1]))
        
    def initialize_world(self, initial_poisition):
        """Set grid for the task and a starting position."""
        self.initial_position = tuple(initial_poisition)
        self.reset(seed=self.seed)

    def deinitialize_world(self):
        self.reset(seed=self.seed)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.initialize_grid(self.subtask)
        if self.world.initialized:
            self.world.deinit()
        self.world.initialize()
        self.max_int = self.maximal_intersection(self.grid)
        self.initialize_targets()
        self.initial_position = self.initialize_agent_position()
        self.step_no = 0
        self.agent.position = self.initial_position
        self.agent.world_size = self.grid.shape
        self.agent.encoded_position = self.agent.encode_position()
        obs = {
            'agent_position': np.array(self.agent.encoded_position),
            'grid': np.array(self.grid),
        }  
        if self.input_pattern:
            obs['input_pattern'] = np.array(self.train_inp)
        info = {}
        return (obs, info)

    def step(self, action):
        if self.subtask is None:
                raise ValueError('Subtask is not initialized!')
        self.right_placement = 0
        self.step_no += 1
        strafe, add = self.world.parse_action(action)
        self.world.step(self.agent, strafe, add)
        if add != -1: 
            if self.int_colors:
                add *= 10
            self.add_block(position=self.agent.position, color=add)
        obs = {}
        obs['agent_position'] = np.array(self.agent.encoded_position)    
        obs['grid'] = self.grid.copy().astype(self.grid_dtype)
        if self.input_pattern:
            obs['input_pattern'] = self.train_inp.copy()  
        reward = 0
        if action in range(4):
            reward += -0.1 * self.wrong_placement_scale
            right_placement = 0
            done = (self.step_no==self.max_steps)
        else:
            right_placement, done = self.step_intersection(self.grid)
            done = done or (self.step_no==self.max_steps)
            if right_placement == 1 and self.agent.position not in self.used_cells:
                reward += right_placement * self.right_placement_scale
                self.used_cells.append(self.agent.position)
            else:
                reward += -1 * self.wrong_placement_scale
        if self.max_int in self.milestones.keys():
            reward += self.milestones[self.max_int]
            self.milestones[self.max_int] = 0
        if self.prev_action == action:
            reward  += -1 * self.wrong_placement_scale 
        self.prev_action = action
        
        truncated = False
        info = {'max_int':self.max_int, 'right_placement':right_placement, 
                'step_no':self.step_no, 'max_steps':self.max_steps}
        return obs, reward, done, truncated, info

def create_env(
        max_steps=1000, right_placement_scale=1.0, wrong_placement_scale=1.0, 
        seed=None, font_color=0.0, padding=False, random_start=False, input_pattern=False,
        milestones_rewards=[1, 2, 3, 4], pad_val=1.0, int_colors=True
    ):
    env = ARCGridWorld(
        max_steps=max_steps, right_placement_scale=right_placement_scale,
        wrong_placement_scale=wrong_placement_scale, seed=seed, font_color=font_color,
        padding=padding, random_start=random_start, input_pattern=input_pattern,
        milestones_rewards=milestones_rewards, pad_val=pad_val, int_colors=int_colors
        )
    return env

gymnasium.envs.register(
     id='ARC-Gridworld-v0',
     entry_point='ARC_env:create_env',
     kwargs={}
)