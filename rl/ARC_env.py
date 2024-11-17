import math
import warnings
import os
import gymnasium
from typing import Optional
from gymnasium import spaces
from gym import Env, Wrapper as gymWrapper
from gym.spaces import Dict, Box, Discrete, Space
import numpy as np
from copy import copy
from rl.ARC_world import World, Agent
from rl.ARC_task import ARCSubtask
class ARCGridWorld(gymnasium.Env):
    def __init__(
            self, max_steps=1000, right_placement_scale=1., wrong_placement_scale=0.1,
            seed=None, actions_pred=None, target_grid=False) -> None:
        self.agent = Agent()
        self.world = World()
        self.grid = np.zeros((30, 30), dtype=np.float32)
        self.subtask = None
        self.step_no = 0
        self.right_placement_scale = right_placement_scale
        self.wrong_placement_scale = wrong_placement_scale
        self.max_steps = max_steps
        self.right_placement = 0
        self.wrong_placement = 0
        self.initial_position = (14, 14)
        self.seed = seed
        self.max_reward = 0
        self.actions_pred = actions_pred
        self.target_grid = target_grid
        self.action_space = spaces.Discrete(14)
        self.observation_space = {
            'agent_position': spaces.Box(low=0, high=9, shape=(30, 30), dtype=np.int32),
            'grid': spaces.Box(low=0, high=9, shape=(30, 30), dtype=np.float32),
            'input_pattern': spaces.Box(low=0, high=9, shape=(30, 30), dtype=np.float32)   
        }
        self.observation_space = spaces.Dict(self.observation_space)
        if actions_pred is not None:
            self.observation_space['actions_pred'] = spaces.Box(low=1, high=14, shape=(1, 100), dtype=np.int32)
        if target_grid:
            self.observation_space['target_grid'] = spaces.Box(low=0, high=9, shape=(30, 30), dtype=np.float32) 
        
    def add_block(self, position, color):
        x, y = position
        self.grid[x, y] = color

    def remove_block(self, position):
        x, y = position
        if self.grid[x, y] == 0:
            raise ValueError(f'Removal of non-existing block. address: x={x}, y={y}')
        self.grid[x, y] = 0
        
    @staticmethod
    def find_upper_left_grid_corner(grid_size:int):
        return min(14-(grid_size%2)*((grid_size//2)), 14-((grid_size-1)%2)*(((grid_size-1)//2)))
    
    def initialize_agent_position(self):
        grid_size = self.subtask.train_inp_shape[0]
        ul = self.find_upper_left_grid_corner(grid_size)
        i = np.random.randint(low=ul, high=ul+grid_size)
        j = np.random.randint(low=ul, high=ul+grid_size)
        position = (i, j)
        return position 

    def set_subtask(self, subtask:ARCSubtask):
        """
        Assigns provided task into the environment. On each .reset, the env
        Queries the .reset method for the task object. This method should drop
        the task state to the initial one.
        Note that the env can only work with non-None task or task generator.
        """
        self.subtask = subtask
        self.subtask_label = subtask.label
        self.reset(seed=self.seed)
        
    def initialize_grid(self, subtask):
        self.max_reward = subtask.calculate_relevant_blocks(subtask.train_out)
        shape_x = self.subtask.train_out_shape[0]      
        shape_y = self.subtask.train_out_shape[1]
        grid = np.zeros(self.subtask.train_out_shape)
        if shape_x!=30 or shape_y!=30:
            left_pad = (30-shape_x)//2
            right_pad = 30 - shape_x - left_pad
            upper_pad = (30-shape_y)//2
            down_pad = 30 - shape_y - upper_pad
            starting_grid = np.pad(grid, pad_width=[(left_pad,right_pad), (upper_pad, down_pad)], constant_values=1)
        self.grid = starting_grid
        self.world.forbidden_cells = self.define_forbidden_cells(starting_grid)
        assert (starting_grid.shape==(30,30)), f"Grids shapes are not (30,30), instead: {starting_grid.shape}"
        
    def define_forbidden_cells(self, starting_grid):
        return list(zip((starting_grid==1).nonzero()[0], (starting_grid==1).nonzero()[1]))
        
    def initialize_world(self, initial_poisition):
        """Set grid for the task and a starting position."""
        self.initial_position = tuple(initial_poisition)
        self.reset(seed=self.seed)

    def deinitialize_world(self):
        self.reset(seed=self.seed)

    def reset(self, seed=None):
        super().reset(seed=seed)
        if self.world.initialized:
            self.world.deinit()
        self.world.initialize()
        self.subtask.reset()
        self.initialize_grid(self.subtask)
        self.initial_position = self.initialize_agent_position()
        self.step_no = 0
        self.agent.position = self.initial_position
        self.agent.encoded_position = self.agent.encode_position()
        self.subtask.max_int = self.subtask.maximal_intersection(self.grid)
        self.prev_grid_size = 0
        obs = {
            'agent_position': np.array(self.agent.encoded_position),
            'grid': np.array(self.grid),
            'input_pattern': np.array(self.subtask.train_inp)
        }
        if self.actions_pred is not None:
            obs['actions_pred'] = self.actions_pred
        if self.target_grid:
            obs['target_grid'] = self.subtask.train_out     
        info = {}
        return (obs, info)

    def step(self, action):
        if self.subtask is None:
                raise ValueError('Subtask is not initialized!')
        self.step_no += 1
        self.world.step(self.agent, action)
        x, y = self.agent.position
        strafe, add = self.world.parse_action(action)
        if add>=0: 
            self.add_block(position=self.agent.position, color=add)
        obs = {}
        obs['agent_position'] = np.array(self.agent.encoded_position)    
        obs['grid'] = self.grid.copy().astype(np.float32)
        obs['input_pattern'] = self.subtask.train_inp.copy().astype(np.float32)
        if action in range(5):
            reward = 0
            done = self.step_no == self.max_steps
        else:
            right_placement, done = self.subtask.step_intersection(self.grid)
            done = done or (self.step_no == self.max_steps)
            if right_placement == 0:
                reward = 0
            else:
                reward = right_placement * self.right_placement_scale
        if done and self.subtask.max_int==self.max_reward:
            reward += 100
        if self.actions_pred is not None:
            obs['actions_pred'] = self.actions_pred
        if self.target_grid:
            obs['target_grid'] = self.subtask.train_out.copy().astype(np.float32)    
        truncated = False
        info = {}
        norm_reward = reward/(self.max_reward+100)
        return obs, norm_reward, done, truncated, info

def create_env(
        max_steps=1000, right_placement_scale=1., wrong_placement_scale=0.1, 
        seed=None, actions_pred=None, target_grid=False
    ):
    env = ARCGridWorld(
        max_steps=max_steps, right_placement_scale=right_placement_scale,
        seed=seed, actions_pred=actions_pred, target_grid=target_grid
        )
    return env

gymnasium.envs.register(
     id='ARC-Gridworld-v0',
     entry_point='ARCGridWorld_env:create_env',
     kwargs={}
)