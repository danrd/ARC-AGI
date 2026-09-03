import numpy as np
import gymnasium
from gymnasium import spaces
from copy import copy, deepcopy
from rl.utils import repad
from rl.arc_world import World
from symbolic.utils import pad_grid
from symbolic.objects_analysis import OBJECT_DIM
from symbolic.summaries import GridSummary, RELATION_DIM

#: Embeddings are real-valued, unlike every other part of the observation
#: (grids and the action space are integer). Casting them to the grid's dtype
#: rounds every fraction away - `shape_similarity`, `normalized_distance` and
#: the rest collapse to 0 or 1 - so they carry their own.
EMBEDDING_DTYPE = np.float32

#: How many objects an observation has room for. Object-derived parts of the
#: observation are padded to this, and the action space is sized by it, so
#: both are the same shape for every task - which is what lets one agent see
#: rollouts from several subtasks instead of being trained on one at a time.
#: Without it the shapes follow each task's object count and a vec env over
#: two subtasks cannot even be built.
#:
#: 16 covers 97.2% of grids in ARC-AGI-2's training set (median 2 objects,
#: p90 8, p99 23, max 73 at representation level 1). Past that the cost grows
#: quadratically - the relation block is MAX_OBJECTS x (MAX_OBJECTS-1) x
#: RELATION_DIM - for a thinning tail: 24 buys 99.1% at 2.3x the width. What
#: to do about the grids that overflow is open; today the extra objects are
#: neither shown nor addressable (see visible_object_count).
MAX_OBJECTS = 16

#: A padded row is all zeros, and a real object's embedding never is -
#: checked over 742 objects, and it follows from OBJECT_SCHEMA, whose size
#: and bounding-box fields are positive for any non-empty object. So the
#: padding is distinguishable without a separate mask, which is the
#: convention ARCGNNExtractor already reads (`obj_emb.sum(dim=1) != 0`).
class ARCGridWorld(gymnasium.Env):
    def __init__(
                self, max_episode_len=25, right_placement_reward=5.0, action_penalty=1.0, repetitive_actions_penalty=1.0,
                seed=None, font_color=0, padding=False, input_pattern=False,
                milestones_rewards=(1, 2, 3, 4), pad_val=10, reward_approach=1, repr_level=1,
                observation_space_elements = ["objects_emb", "relations_emb"],
                feasible_actions={0:'submit'},
                max_objects=MAX_OBJECTS,
                ):
        self.max_objects = max_objects
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
        # Sized by max_objects rather than by the subtask, and so identical
        # for every subtask - set_subtask only fills in the action count.
        self.action_space = spaces.MultiDiscrete([
            len(self.actions_dict),  # Action types
            self.max_objects,        # Object 1 index
            self.max_objects,        # Object 2 index
        ])
        # Initialize observation space
        self.observation_space = {
            'grid': spaces.Box(low=self.low_val, high=self.max_val, shape=(30, 30), dtype=self.grid_dtype),
        }
        self.observation_space = spaces.Dict(self.observation_space)

    def step_intersection(self, grid:np.array):
        """Calculates the difference between the maximal intersection at previous step and the current one.
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

    @property
    def train_out(self):
        return self._train_out

    @train_out.setter
    def train_out(self, value):
        """Sets the validity mask maximal_intersection reads along with it.

        A property rather than two assignments side by side, because the
        mask is derived from the target and the two are only meaningful
        together - set_subtask is not the only place the target is assigned,
        and one that forgot the mask would leave it describing the previous
        subtask, scoring every grid against the wrong region.
        """
        self._train_out = np.asarray(value)
        self._target_valid = (self._train_out != self.pad_val)

    def maximal_intersection(self, grid:np.array):
        """Cells matching the target, less cells contradicting it, over the
        region both grids call real.

        The same count as before, arrived at with a third of the work. Both
        halves ranged over the same valid region, so their sizes add up to
        it: matches - misses is matches - (valid - matches). That leaves one
        comparison instead of two, and the target's half of the validity
        mask is fixed for the subtask rather than rebuilt per call - which
        matters at the ~117k calls a single MCTS search makes.
        """
        grid = np.asarray(grid)
        valid = (grid != self.pad_val) & self._target_valid
        matches = np.count_nonzero((grid == self.train_out) & valid)
        return 2 * matches - np.count_nonzero(valid)

    def set_subtask(self, subtask):
        """Assigns provided task into the environment. On each .reset, the env
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
        # The action space stays as __init__ built it: sized by max_objects,
        # the same for every subtask. Indices past this subtask's object
        # count address nothing and are handled in step().
        if "objects_emb" in self.observation_space_elements:
            self.initial_objects_emb = self._pad_objects(
                [obj.create_embedding() for obj in self.initial_objects])
        if "relations_emb" in self.observation_space_elements:
            self.initial_relation_emb = self._pad_relations(
                self.initial_grid_summary.get_relation_embeddings_as_numpy(level=self.repr_level))
        self.reset(seed=self.seed)

    def visible_object_count(self) -> int:
        """Objects this env shows and lets actions address. Objects past
        max_objects are still in self.objects - World reads that list, and
        cell2obj maps cells to it - but they have no row in the observation
        and no index in the action space, so nothing can name them.

        Public because the action space is sized by max_objects whatever the
        subtask holds, so anything enumerating actions (rl.mcts) needs to
        know where the real objects stop - the slots past this point are all
        the same no-op, and there are (max_objects/n)^2 of them.
        """
        return min(len(self.objects) or len(self.initial_objects), self.max_objects)

    def _pad_objects(self, embeddings) -> np.ndarray:
        """(max_objects, OBJECT_DIM), zero-padded. A real object's embedding
        is never all zeros, so the padding stays distinguishable."""
        padded = np.zeros((self.max_objects, OBJECT_DIM), dtype=EMBEDDING_DTYPE)
        rows = np.asarray(embeddings, dtype=EMBEDDING_DTYPE).reshape(-1, OBJECT_DIM)
        keep = min(len(rows), self.max_objects)
        padded[:keep] = rows[:keep]
        return padded

    def _pad_relations(self, embeddings) -> np.ndarray:
        """(max_objects, (max_objects - 1) * RELATION_DIM), zero-padded.

        Each row holds one object's vectors against the others laid end to
        end, so a row from a grid with fewer objects is shorter than the slot
        it goes into and the remainder stays zero - the same "no relation
        recorded" that an absent pair leaves.
        """
        padded = np.zeros((self.max_objects, (self.max_objects - 1) * RELATION_DIM),
                          dtype=EMBEDDING_DTYPE)
        rows = np.asarray(embeddings, dtype=EMBEDDING_DTYPE)
        if rows.ndim == 2 and rows.size:
            keep_rows = min(rows.shape[0], self.max_objects)
            keep_cols = min(rows.shape[1], padded.shape[1])
            padded[:keep_rows, :keep_cols] = rows[:keep_rows, :keep_cols]
        return padded

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
        # Flat, matching np.array(self.action_space.nvec) - the three entries
        # are action count, object slots, object slots. Bounded by 900
        # because a 30x30 grid cannot hold more objects than cells.
        self.observation_space['action_space'] = spaces.Box(low=0, high=900, shape=(3,), dtype=np.int64)
        if self.input_pattern == 'separate':
            self.observation_space['input_pattern'] = spaces.Box(low=self.low_val, high=self.max_val, shape=(shape_x_inp, shape_y_inp), dtype=self.grid_dtype)
        if "target" in self.observation_space_elements:
            self.observation_space['target'] = spaces.Box(low=self.low_val, high=self.max_val, shape=(shape_x, shape_y), dtype=self.grid_dtype)
        # Both embedding blocks are sized by max_objects, not by this
        # subtask's object count, so they are the same shape for every
        # subtask - see MAX_OBJECTS.
        if "objects_emb" in self.observation_space_elements:
            # Width and bounds from OBJECT_SCHEMA, which is what
            # GridObject.create_embedding fills: every field it declares is
            # normalised into [0, 1], and zero rows are padding.
            self.observation_space['objects_emb'] = spaces.Box(
                low=0, high=1, shape=(self.max_objects, OBJECT_DIM), dtype=EMBEDDING_DTYPE)
        if "relations_emb" in self.observation_space_elements:
            # One row per object slot, holding its vector against each of the
            # others - see GridSummary.get_relation_embeddings_as_numpy, which
            # this shape has to agree with. Unbounded: size_ratio is a ratio of
            # areas (379 at the widest measured over ARC-AGI-2's training set)
            # and the offsets are signed, so any finite bound here would be a
            # guess that observations fall outside of.
            self.observation_space['relations_emb'] = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.max_objects, (self.max_objects - 1) * RELATION_DIM),
                dtype=EMBEDDING_DTYPE)
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

    def _submit_reward(self, max_int):
        """Milestone-based reward for a submit action, parameterized on
        max_int (rather than reading self.max_int) so it can be reused by
        simulate_action() for MCTS tree search without touching self.*."""
        reward = 0
        i = 0
        if self.reward_approach in [1, 3]:
            for idx, milestone_int in enumerate(self.milestones.keys()):
                if max_int >= milestone_int:
                    i = idx+1
            if i==len(self.milestones.keys()):
                reward = self.milestones_rewards[-1]
            else:
                if self.reward_approach == 1:
                   reward = -1 * list(reversed(self.milestones_rewards))[i] # reward only for the whole result
                elif self.reward_approach == 3: # no negative rewards for partial result
                    reward = 0

        elif self.reward_approach == 2: # partial reward for some achieved milestones
            # The partial credit is kept at the first milestone that was not
            # reached, not overwritten by the penalty. Assigning there paid
            # the same -milestones_rewards[-1] for every submit short of the
            # solved one, so this approach was flat where it is meant to have
            # a gradient - and the only positive submit in any approach was
            # the fully solved one, which is why nothing positive ever came
            # out of a search. The penalty is now what a submit that achieved
            # nothing is worth, so giving up still costs.
            for idx, milestone_int in enumerate(self.milestones.keys()):
                if max_int >= milestone_int:
                    reward += self.milestones_rewards[idx]
                else:
                    break
            if reward == 0:
                reward = -1 * self.milestones_rewards[-1]
        elif self.reward_approach == 4: # monotonic scaling reward based on percentage of the task complition
            reward = self.max_reward_base
        return reward

    def submit_grid(self):
        obs = {}
        obs['grid'] = self.grid.copy().astype(self.grid_dtype)
        obs['action_space'] = np.array(self.action_space.nvec)
        if self.input_pattern == 'separate':
            obs['input_pattern'] = self.train_inp.copy()
        if "target" in self.observation_space_elements:
            obs['target'] = np.array(self.train_out).copy().astype(self.grid_dtype)
        if "objects_emb" in self.observation_space_elements:
            obs['objects_emb'] = self.objects_emb.copy().astype(EMBEDDING_DTYPE)
        if "relations_emb" in self.observation_space_elements:
            obs['relations_emb'] = self.relations_emb.copy().astype(EMBEDDING_DTYPE)
        truncated = False
        info = {}
        reward = self._submit_reward(self.max_int)
        done = True
        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        # options: unused, but gymnasium.Env.reset()'s signature requires it -
        # every wrapper gym.make() adds (OrderEnforcing, PassiveEnvChecker, ...)
        # calls reset(seed=..., options=...) unconditionally.
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
            obs['objects_emb'] = self.objects_emb.copy().astype(EMBEDDING_DTYPE)
        if "relations_emb" in self.observation_space_elements:
            self.relations_emb = self.initial_relation_emb.copy()
            obs['relations_emb'] = self.relations_emb.copy().astype(EMBEDDING_DTYPE)

        info = {}
        return (obs, info)

    def step(self, action):
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
        # The action space has max_objects slots whatever the subtask holds,
        # so an index can name a slot no object occupies. That is an action
        # that does nothing, scored like any other ineffective one - not an
        # error, and not silently redirected to some other object, which
        # would teach the policy that a wrong index still works.
        visible = self.visible_object_count()
        if action[1] >= visible or action[2] >= visible:
            new_grid = self.grid
            eq_check = True
        else:
            object_1 = self.objects[action[1]]
            object_2 = self.objects[action[2]]
            # Apply action and get modified grid if needed
            new_grid = self.world.step(add, transform, object_1, object_2, self.grid, self.objects, self.initial_grid_summary.repr_levels[self.repr_level].cell2obj)
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
            # Same shape and dtype as reset() and submit_grid() hand back: an
            # episode whose observations change shape or type partway through
            # is one the policy sees two different things in.
            self.objects_emb = self._pad_objects([obj.create_embedding() for obj in self.objects])
            obs['objects_emb'] = self.objects_emb.copy()
        if "relations_emb" in self.observation_space_elements:
            for obj_idx in list(set([action[1], action[2]])): # update involved objects relation embeddings
                if obj_idx < visible:
                    self.grid_summary.update_representation_level(self.repr_level, self.objects[obj_idx])
            self.relations_emb = self._pad_relations(
                self.grid_summary.get_relation_embeddings_as_numpy(level=self.repr_level))
            obs['relations_emb'] = self.relations_emb.copy()

        right_placement, done = self.step_intersection(self.grid)
        reward += right_placement * self.right_placement_reward  # Bonus for effective transformations

        # Discourage action repetition
        if self.prev_action is not None and np.array_equal(self.prev_action, action):
            reward += -1 * self.repetitive_actions_penalty

        self.prev_action = action.copy()

        # Reward normalization
        # Normalised but not rounded. round(x, 2) used to sit here, and
        # max_reward scales with the distance to the target, so on any task
        # further away than a few cells the penalty for a useless action -
        # -1/max_reward - rounded to exactly 0.00. Measured over 2000 random
        # steps per task: with the rounding, 93% to 99% of steps paid exactly
        # zero on three of four tasks; without it, 0.1% to 0.7%. A search
        # does not care - playouts pick actions without consulting reward,
        # and removing the rounding changed nothing over 24 tasks - but PPO
        # learns from nothing else, and was being trained on a signal that
        # was zero almost everywhere.
        reward = reward / self.max_reward

        truncated = (self.step_no >= self.max_episode_len)
        info = {
            'right_placement': right_placement,
            'change_of_grid': not eq_check,
            'action_space_shape': self.action_space.nvec,
        }
        return obs, reward, done, truncated, info

    def simulate_action(self, action, objects, grid, max_int, prev_action):
        """Side-effect-free version of one step's physics + reward: runs
        against caller-supplied objects/grid/max_int/prev_action instead of
        self.objects/self.grid/self.max_int/self.prev_action, and never
        touches them or self.step_no/self.right_placement - only the two
        objects the action targets get mutated (World.apply_transform's
        usual contract), and only if the caller passed in copies of those.

        Built for MCTS tree search (rl.mcts.EnvironmentSimulator), which
        needs to explore many candidate actions per real step without
        paying for a full env.reset()/deepcopy of every object on every
        simulated node. Does not build a full observation (no objects_emb/
        relations_emb refresh) - simulated nodes only need the reward/done
        signal, not something to hand to a policy.

        Returns (new_grid, objects, new_max_int, reward, done). `objects`
        is returned as received (its 1-2 mutated entries are exactly the
        ones the caller should hold onto for the next simulated step).
        """
        if self.actions_dict[action[0]] == 'submit':
            return grid, objects, max_int, self._submit_reward(max_int), True

        add, transform = self.world.parse_action(action)
        # An index can name a slot no object occupies - the action space has
        # max_objects of them whatever the subtask holds. Scored exactly as
        # step() scores it, or a simulated rollout would value an action the
        # real env does not.
        visible = min(len(objects), self.max_objects)
        if action[1] >= visible or action[2] >= visible:
            new_grid, eq_check = grid, True
        else:
            object_1 = objects[action[1]]
            object_2 = objects[action[2]]
            cell2obj = self.initial_grid_summary.repr_levels[self.repr_level].cell2obj
            new_grid = self.world.step(add, transform, object_1, object_2, grid, objects, cell2obj)
            eq_check = np.array_equal(new_grid, grid)

        reward = -1 * self.action_penalty if (new_grid is not None and eq_check) else 0.0

        new_max_int = self.maximal_intersection(new_grid)
        done = (new_max_int == self.target_int)
        right_placement = new_max_int - max_int
        reward += right_placement * self.right_placement_reward

        if prev_action is not None and np.array_equal(prev_action, action):
            reward += -1 * self.repetitive_actions_penalty

        # Normalised but not rounded. round(x, 2) used to sit here, and
        # max_reward scales with the distance to the target, so on any task
        # further away than a few cells the penalty for a useless action -
        # -1/max_reward - rounded to exactly 0.00. Measured over 2000 random
        # steps per task: with the rounding, 93% to 99% of steps paid exactly
        # zero on three of four tasks; without it, 0.1% to 0.7%. A search
        # does not care - playouts pick actions without consulting reward,
        # and removing the rounding changed nothing over 24 tasks - but PPO
        # learns from nothing else, and was being trained on a signal that
        # was zero almost everywhere.
        reward = reward / self.max_reward
        return new_grid, objects, new_max_int, reward, done

    def get_state(self):
        """Capture the complete state of the environment for later restoration.
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
     entry_point='rl.arc_env:create_env',
     kwargs={}
)
