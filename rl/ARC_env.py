import time
import cProfile
import numpy as np
import gymnasium
from gymnasium import spaces
from copy import copy, deepcopy
from rl.utils import repad
from rl.ARC_world import World
from symbolic.utils import pad_grid
from symbolic.summaries import GridSummary
class ARCGridWorld(gymnasium.Env):
    def __init__(
                self, max_episode_len=25, right_placement_reward=5.0, action_penalty=1.0, repetitive_actions_penalty=1.0,
                seed=None, font_color=0, padding=False, input_pattern=False,
                milestones_rewards=(1, 2, 3, 4), pad_val=10, reward_approach=1, repr_level=1,
                observation_space_elements = ["objects_emb", "relations_emb"],
                feasible_actions={0:'submit'},
                ):
        self.step_no = 0
        self.right_placement_reward = right_placement_reward
        self.action_penalty = action_penalty
        self.repetitive_actions_penalty = repetitive_actions_penalty
        self.max_episode_len = max_episode_len
        self.right_placement = 0
        self.wrong_placement = 0
        self.prev_action = None
        self.seed = seed
        self.font_color = font_color
        self.padding = padding
        self.pad_val = pad_val
        self.low_val = self.pad_val if self.pad_val < 0 else 0
        self.max_val = self.pad_val if self.pad_val > 0 else 9
        self.input_pattern = input_pattern
        self.milestones_rewards = milestones_rewards
        self.reward_approach = reward_approach
        self.repr_level = repr_level
        self.observation_space_elements = observation_space_elements
        self.grid_dtype = np.int64
        self.actions_dict = feasible_actions
        self.action_name_to_idx = {name: idx for idx, name in self.actions_dict.items()}
        self.objects = []
        self.action_space = spaces.MultiDiscrete([
            1, # Action types
            30,    # Object 1 index
            30,    # Object 2 index
        ])
        # Initialize observation space
        self.observation_space = {
            'grid': spaces.Box(low=self.low_val, high=self.max_val, shape=(30, 30), dtype=self.grid_dtype),
        }
        self.observation_space = spaces.Dict(self.observation_space)

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
        intersection = ((np.array(grid)==np.array(self.train_out)) * (grid!=self.pad_val) * (self.train_out!=self.pad_val)).sum() - ((np.array(grid)!=np.array(self.train_out)) * (grid!=self.pad_val) * (self.train_out!=self.pad_val)).sum()
        return intersection

    def set_subtask(self, subtask):
        """
        Assigns provided task into the environment. On each .reset, the env
        Queries the .reset method for the task object. This method should drop
        the task state to the initial one.
        Note that the env can only work with non-None task or task generator.
        """
        self.subtask = deepcopy(subtask)
        if self.padding:
            self.subtask = repad(self.subtask, max_shape=self.padding, pad_val=self.pad_val)
            grid_without_padding = copy(self.subtask.train_inp)
            grid_without_padding[grid_without_padding==self.pad_val] = self.font_color
            self.initial_grid_summary = GridSummary(grid=grid_without_padding, shape=self.subtask.train_inp_shape, levels=[self.repr_level])
        else:
            self.initial_grid_summary = GridSummary(grid=self.subtask.train_inp, shape=self.subtask.train_inp_shape, levels=[self.repr_level])
        self.initial_objects = self.initial_grid_summary.repr_levels[self.repr_level].objects
        self.train_inp = self.subtask.train_inp
        self.train_out = self.subtask.train_out
        self.subtask_label = self.subtask.label
        self.action_space = spaces.MultiDiscrete([ # adapt action space to specific task
            len(self.actions_dict.keys()),
            len(self.initial_objects),
            len(self.initial_objects),
        ])
        if "objects_emb" in self.observation_space_elements:
            self.initial_objects_emb = np.array([obj.create_embedding() for obj in self.initial_objects])
        if "relations_emb" in self.observation_space_elements:
            self.initial_grid_summary.get_relation_embeddings_as_numpy(level=self.repr_level)
            self.initial_relation_emb = self.initial_grid_summary.get_relation_embeddings_as_numpy(level=self.repr_level)
        self.reset(seed=self.seed)

    def initialize_observation_space(self, subtask):
        shape_x = self.subtask.train_out_shape[0]
        shape_y = self.subtask.train_out_shape[1]
        shape_x_inp = self.subtask.train_inp_shape[0]
        shape_y_inp = self.subtask.train_inp_shape[1]
        if self.input_pattern == 'start':
            starting_grid = copy(self.subtask.train_inp)
            if shape_x_inp < shape_x or shape_y_inp < shape_y:
                starting_grid = pad_grid(starting_grid, (shape_x, shape_y), self.font_color)
        else:
            starting_grid = np.zeros(self.subtask.train_out_shape)
        if self.padding:
            if shape_x != self.padding[0] or shape_y != self.padding[1]:
                starting_grid = pad_grid(starting_grid, self.padding, self.pad_val)
                assert (starting_grid.shape==self.padding), f"Grids shapes are not {self.padding}, instead: {starting_grid.shape}"
                shape_x, shape_y = self.padding
        self.grid = starting_grid
        self.world = World(objects=self.initial_objects, actions_dict=self.actions_dict, font_color=self.font_color)
        # Update observation space for current grid size
        self.observation_space = {}
        self.observation_space['grid'] = spaces.Box(low=self.low_val, high=self.max_val, shape=(shape_x, shape_y), dtype=self.grid_dtype)
        self.observation_space['action_space'] = spaces.Box(low=0, high=900, shape=(1, 3), dtype=np.int64)
        if self.input_pattern == 'separate':
            self.observation_space['input_pattern'] = spaces.Box(low=self.low_val, high=self.max_val, shape=(shape_x_inp, shape_y_inp), dtype=self.grid_dtype)
        if "target" in self.observation_space_elements:
            self.observation_space['target'] = spaces.Box(low=self.low_val, high=self.max_val, shape=(shape_x, shape_y), dtype=self.grid_dtype)
        if "objects_emb" in self.observation_space_elements:
            self.observation_space['objects_emb'] = spaces.Box(low=0, high=1, shape=(len(self.initial_objects), 32), dtype=self.grid_dtype)
        if "relations_emb" in self.observation_space_elements:
            self.observation_space['relations_emb'] = spaces.Box(low=0, high=1, shape=(len(self.initial_objects), len(self.initial_objects)*8), dtype=self.grid_dtype)
        self.observation_space = spaces.Dict(self.observation_space)

    def initialize_targets(self):
        self.target_int = (self.train_out!=self.pad_val).sum()
        if self.target_int == self.max_int:
            self.milestones = {self.target_int:self.right_placement_reward * self.milestones_rewards[step] for step in range(len(self.milestones_rewards)-1)}
            self.max_reward = self.right_placement_reward
        else:
            milestone_step = (self.target_int - self.max_int) / 4
            self.max_reward = (self.target_int - self.max_int) * self.right_placement_reward
            self.milestones = {int(self.max_int + milestone_step * (step+1)):self.max_reward * self.milestones_rewards[step] for step in range(len(self.milestones_rewards)-1)}
            self.milestones[self.target_int] = self.max_reward * self.milestones_rewards[-1]
        self.base_int = copy(self.max_int)
        if self.reward_approach == 2:
            self.max_reward += sum(self.milestones.values())
        elif self.reward_approach == 4:
            self.max_reward *= 2
        elif self.reward_approach in [1,3]:
            self.max_reward += self.milestones[self.target_int]

    def submit_grid(self):
        obs = {}
        obs['grid'] = self.grid.copy().astype(self.grid_dtype)
        obs['action_space'] = np.array(self.action_space.nvec)
        if self.input_pattern == 'separate':
            obs['input_pattern'] = self.train_inp.copy()
        if "target" in self.observation_space_elements:
            obs['target'] = np.array(self.train_out).copy().astype(self.grid_dtype)
        if "objects_emb" in self.observation_space_elements:
            obs['objects_emb'] = self.objects_emb.copy().astype(self.grid_dtype)
        if "relations_emb" in self.observation_space_elements:
            obs['relations_emb'] = self.relations_emb.copy().astype(self.grid_dtype)
        truncated = False
        info = {}
        reward = 0
        i = 0
        if self.reward_approach in [1, 3]:
            for idx, milestone_int in enumerate(self.milestones.keys()):
                if self.max_int >= milestone_int:
                    i = idx+1
            if i==len(self.milestones.keys()):
                reward = self.milestones_rewards[-1]
            else:
                if self.reward_approach == 1:
                   reward = -1 * list(reversed(self.milestones_rewards))[i] # reward only for the whole result
                elif self.reward_approach == 3: # no negative rewards for partial result
                    reward = 0

        elif self.reward_approach == 2: # partial reward for some achieved milestones
            for idx, milestone_int in enumerate(self.milestones.keys()):
                if self.max_int >= milestone_int:
                    reward += self.milestones_rewards[idx]
                else:
                    reward = -1 * self.milestones_rewards[-1]
                    break
        elif self.reward_approach == 4: # monotonic scaling reward based on percentage of the task complition
            completion_share = (self.max_int-self.base_int) / (self.target_int-self.base_int)
            reward = self.max_reward_base
        done = True
        return obs, reward, done, truncated, info

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.objects = deepcopy(self.initial_objects)
        self.grid_summary = copy(self.initial_grid_summary)
        self.initialize_observation_space(self.subtask)
        self.max_int = self.maximal_intersection(self.grid)
        self.initialize_targets()
        self.step_no = 0
        self.prev_action = None

        obs = {
            'grid': np.array(self.grid).astype(self.grid_dtype),
            'action_space': np.array(self.action_space.nvec)
        }

        if self.input_pattern == 'separate':
            obs['input_pattern'] = np.array(self.train_inp).copy().astype(self.grid_dtype)
        if "target" in self.observation_space_elements:
            obs['target'] = np.array(self.train_out).copy().astype(self.grid_dtype)
        if "objects_emb" in self.observation_space_elements:
            self.objects_emb = self.initial_objects_emb.copy()
            obs['objects_emb'] = self.objects_emb.copy().astype(self.grid_dtype)
        if "relations_emb" in self.observation_space_elements:
            self.relations_emb = self.initial_relation_emb.copy()
            obs['relations_emb'] = self.relations_emb.copy().astype(self.grid_dtype)

        info = {}
        return (obs, info)

    def step(self, action):
        start = time.time()
        reward = 0
        if self.subtask is None:
            raise ValueError('Subtask is not initialized!')
        self.right_placement = 0
        self.step_no += 1
        # Submit grid (final action)
        if self.actions_dict[action[0]] == 'submit':
           return self.submit_grid()
        # Parse action with MultiDiscrete functionality
        add, transform = self.world.parse_action(action)
        object_1 = self.objects[action[1]]
        object_2 = self.objects[action[2]]
        # Apply action and get modified grid if needed
        with cProfile.Profile() as pr:
            new_grid = self.world.step(add, transform, object_1, object_2, self.grid, self.objects, self.initial_grid_summary.repr_levels[self.repr_level].cell2obj)
            opp_time = (time.time()-start) / 60
            if opp_time > 0.01 :
                print(f'Transform "{transform}" performed with {opp_time} operation time')
                pr.print_stats()

        eq_check = np.array_equal(new_grid, self.grid)
        # Update grid if it was transformed
        if new_grid is not None and eq_check:
            reward += -1 * self.action_penalty # penalty for ineffective actions

        obs = {}
        obs['grid'] = copy(new_grid)
        obs['action_space'] = np.array(self.action_space.nvec)
        self.grid = copy(new_grid)


        if self.input_pattern == 'separate':
            obs['input_pattern'] = self.train_inp.copy().astype(self.grid_dtype)
        if "target" in self.observation_space_elements:
            obs['target'] = self.train_out.copy().astype(self.grid_dtype)
        if "objects_emb" in self.observation_space_elements:
            obs['objects_emb'] = np.array([obj.create_embedding() for obj in self.objects])
        if "relations_emb" in self.observation_space_elements:
            for obj_idx in list(set([action[1], action[2]])): # update involved objects relation embeddings
                self.grid_summary.update_representation_level(self.repr_level, self.objects[obj_idx])
            obs['relations_emb'] = self.grid_summary.get_relation_embeddings_as_numpy(level=self.repr_level)

        right_placement, done = self.step_intersection(self.grid)
        done = (self.step_no==self.max_episode_len)
        reward += right_placement * self.right_placement_reward  # Bonus for effective transformations

        # Discourage action repetition
        if self.prev_action is not None and np.array_equal(self.prev_action, action):
            reward += -1 * self.repetitive_actions_penalty

        self.prev_action = action.copy()

        # Reward normalization
        reward = round(reward / self.max_reward, 2)

        truncated = False
        info = {
            'right_placement': right_placement,
            'change_of_grid': not eq_check,
            'action_space_shape': self.action_space.nvec,
        }
        return obs, reward, done, truncated, info

    def get_state(self):
        """
        Capture the complete state of the environment for later restoration.
        Returns a dictionary containing all necessary state information.
        """
        state = {
            'grid': self.grid.copy(),
            'step_no': self.step_no,
            'prev_action': self.prev_action.copy() if self.prev_action is not None else None,
            'right_placement': self.right_placement,
            'wrong_placement': self.wrong_placement,
        }

        return state

    def set_state(self, state):
        """Restore the environment to a previously captured state.

        Args:
            state: State dictionary returned by get_state().
        """
        self.grid = state['grid'].copy()
        self.step_no = state['step_no']
        self.prev_action = state['prev_action'].copy() if state['prev_action'] is not None else None
        self.right_placement = state['right_placement']
        self.wrong_placement = state['wrong_placement']

def create_env(
                max_episode_len=25, right_placement_reward=5.0, action_penalty=1.0, repetitive_actions_penalty=1.0,
                seed=None, font_color=0, padding=False, input_pattern=False, milestones_rewards=(1, 2, 3, 4),
                pad_val=10, reward_approach=1, repr_level=1, observation_space_elements = ["objects_emb", "relations_emb"],
                feasible_actions={0:"submit"}
               ):
    env = ARCGridWorld(
        max_episode_len=max_episode_len, right_placement_reward=right_placement_reward,
        action_penalty=action_penalty, repetitive_actions_penalty=repetitive_actions_penalty,
        seed=seed, font_color=font_color, padding=padding, input_pattern=input_pattern, repr_level=repr_level,
        reward_approach=reward_approach, milestones_rewards=milestones_rewards, pad_val=pad_val,
        feasible_actions=feasible_actions,observation_space_elements=observation_space_elements,
        )
    return env

gymnasium.envs.register(
     id='ARC-Gridworld-v1',
     entry_point='ARC_env:create_env',
     kwargs={}
)
