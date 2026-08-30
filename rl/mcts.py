import numpy as np
import torch
import collections
import itertools
import random
import math
from typing import Dict, Any, List, Optional
from copy import copy, deepcopy
from tqdm import tqdm


def process_observations(observations, device, pad_inp=True, multi_env=False):
    """Process different types of observations to make them compatible with the policy.

    Args:
        observations: Batch of observations (could be dicts, arrays, etc.)
        device: Target device for tensors
        pad_inp: Whether to pad inputs (default: True)
        multi_env: Whether to add an additional first dimension for multiple environments (default: False)

    Returns:
        Processed observations ready for the model, with an additional first dimension if multi_env=True
    """
    # Dictionary observations from DataLoader
    if isinstance(observations, dict):
        result = {}
        for key in observations:
            try:
                # If it's already a tensor, just move to device
                if isinstance(observations[key], torch.Tensor):
                    tensor = observations[key].to(device)
                else:
                    # Otherwise convert to tensor
                    tensor = torch.tensor(observations[key], device=device)

                # Add environment dimension if needed
                if multi_env and tensor.dim() > 0:
                    tensor = tensor.unsqueeze(0)  # Add env dimension as first dimension

                result[key] = tensor
            except Exception as e:
                print(f"Error processing key {key}: {e}")
                # Try to handle numpy arrays specifically
                if isinstance(observations[key], np.ndarray):
                    tensor = torch.from_numpy(observations[key]).to(device)
                    if multi_env and tensor.dim() > 0:
                        tensor = tensor.unsqueeze(0)
                    result[key] = tensor
        return result

    # Unknown observation type
    else:
        print(f"Warning: Unknown observation type: {type(observations)}")
        return observations

def submit_index(env) -> Optional[int]:
    """Which action index means submit, or None if this env has no such
    action. Read from actions_dict rather than assumed to be 0: the mapping
    is caller-supplied (see ARCGridWorld's feasible_actions) and nothing
    guarantees the order."""
    actions = getattr(env, "actions_dict", None) or {}
    return next((int(i) for i, name in actions.items() if name == 'submit'), None)


def enumerate_actions(env, include_submit: bool = False) -> List[List[int]]:
    """Every action worth trying on `env`, as [action, obj_1, obj_2] lists.

    Not the whole action space: that has ARCGridWorld.MAX_OBJECTS slots
    whatever the subtask holds, so it is the same shape for every subtask and
    a single agent can train across several. Indices past the objects this
    subtask actually has name nothing - the env scores them as an action that
    changes nothing - and there are (max_objects / n)^2 of them, which for a
    two-object grid at max_objects=16 is 64 identical no-ops for every real
    action. A search that enumerates them spends almost all of its budget
    rediscovering that they do nothing.

    Submit is excluded for the same reason, and it is the more expensive of
    the two. A search never needs to decide to submit - it knows when it has
    reached the target, and replay_solution appends the submit that a
    behavioural-cloning trace needs - so the action can only end an episode
    early. Measured over four configurations it ended 19-26% of them, each
    one cut short of the cap by an action that cannot improve the grid, and
    it put a terminal child under every node of the tree whose value (the
    submit reward, negative while unsolved) then dragged that branch's UCB
    down. `include_submit` is for measuring the submit reward itself, which
    only reaches the tree through those children.
    """
    visible = env.visible_object_count() if hasattr(env, "visible_object_count") else None
    dims = list(env.action_space.nvec)
    if visible:
        dims[1] = min(dims[1], visible)
        dims[2] = min(dims[2], visible)
    actions = [list(action) for action in itertools.product(*[range(int(d)) for d in dims])]
    if include_submit:
        return actions
    index = submit_index(env)
    return actions if index is None else [a for a in actions if a[0] != index]


def test_individual_actions(env, max_actions: int = None) -> Dict[int, Dict[str, Any]]:
    """Test each action individually (episode length 1) to identify promising actions.

    Args:
        env: Environment to test actions in
        max_actions: Maximum number of actions to test (None = test all)

    Returns:
        Dictionary mapping action_id to results (reward, observation, etc.)
    """
    action_results = {}

    all_actions = enumerate_actions(env)

    if max_actions:
        all_actions = all_actions[:max_actions]

    print(f"Testing {len(all_actions)} individual actions...")

    for action in tqdm(all_actions):
        observation = env.reset()[0]

        # Take single action
        next_observation, reward, done, truncated, info = env.step(action)

        action_results[tuple(action)] = {
            'initial_observation': observation,
            'action': action,
            'reward': reward,
            'next_observation': next_observation,
            'done': done,
            'truncated': truncated,
            'info': info
        }

    return action_results

def identify_promising_actions(action_results: Dict[int, Dict[str, Any]],
                              reward_threshold: float = 0.0,
                              keep_best: int = 0) -> List[int]:
    """Actions worth building rollouts from, best first.

    Args:
        action_results: Results from test_individual_actions
        reward_threshold: Minimum reward to consider an action promising
        keep_best: When nothing clears the threshold, fall back to this many
            best-scoring actions instead of none. On this reward scale a
            single step almost never scores above zero - measured over 42
            ARC-AGI-1 tasks, 13 had any improving single action at all - so
            a bare threshold empties the pool on most tasks and rollout
            collection falls back to sampling the whole action space. The
            ordering is still informative when the level is not: the least
            bad actions are the ones that changed something without making
            the grid worse. 0 keeps the old all-or-nothing behaviour.

    Returns:
        List of promising action IDs
    """
    ranked = sorted(action_results, key=lambda a: action_results[a]['reward'], reverse=True)
    promising_actions = [a for a in ranked if action_results[a]['reward'] > reward_threshold]

    if not promising_actions and keep_best:
        promising_actions = ranked[:keep_best]
        print(f"No action scored above {reward_threshold}; keeping the {len(promising_actions)} best")

    print(f"Found {len(promising_actions)} promising actions with reward > {reward_threshold}")
    for i, action_id in enumerate(promising_actions[:10]):  # Show top 10
        reward = action_results[action_id]['reward']
        print(f"  Action {action_id}: reward = {reward:.4f}")

    return promising_actions

def collect_random_rollouts(env,
                           promising_actions: List[List[int]],
                           n_rollouts: int = 100,
                           max_episode_len: int = 50) -> List[Dict[str, Any]]:
    """Collect rollouts focusing on promising actions but with some exploration.

    Args:
        env: Environment
        promising_actions: List of actions that showed positive rewards
        n_rollouts: Number of rollouts to collect
        max_episode_len: Maximum episode length
        exploration_prob: Probability of taking random action vs promising action

    Returns:
        List of rollout dictionaries
    """
    rollouts = []

    for i in tqdm(range(n_rollouts)):
        observation = env.reset()[0]
        done = False
        truncated = False

        rollout = {
            'observations': [],
            'next_observations': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'infos': []
        }

        total_reward = 0
        step_count = 0

        while not (done or truncated) and step_count < max_episode_len:
            if promising_actions:
                action = random.choice(promising_actions)
                action = list(action)
            else:
                action = env.action_space.sample()

            try:
                # Now take the actual step in the real environment
                next_observation, reward, done, truncated, info = env.step(action)
            except KeyError:
                print(action)

            rollout['observations'].append(observation)
            rollout['next_observations'].append(next_observation)
            rollout['actions'].append(action)
            rollout['rewards'].append(reward)
            rollout['dones'].append(done)
            rollout['infos'].append(info)

            total_reward += reward
            step_count += 1
            observation = next_observation

        rollout['total_reward'] = total_reward
        rollout['length'] = step_count
        # Not any(dones): submit_grid ends the episode with done=True
        # whatever it submitted, so "the episode finished" and "the answer
        # was right" are different questions and only step_intersection's
        # done answers the second. The env's own counters answer it directly.
        rollout['solved'] = bool(env.max_int == env.target_int)
        # Progress, alongside the reward. total_reward is denominated in
        # whichever reward_approach the env was built with, and the
        # approaches do not share a scale - approach 1 runs -4..+4 over the
        # milestones where approach 2 runs -4..+10 - so rewards from two of
        # them cannot be compared, and a search that is doing better can
        # score lower. The intersection is the same count of cells whatever
        # the approach, which makes these three the comparable record.
        rollout['max_int'] = int(env.max_int)
        rollout['base_int'] = int(env.base_int)
        rollout['target_int'] = int(env.target_int)

        # Every rollout is kept, and select_best_rollouts ranks them.
        # Selecting on total_reward > 0 here threw away the entire run on
        # this reward scale: measured over 42 ARC-AGI-1 tasks with the real
        # transform vocabulary, a rollout's total is 0.0 or slightly
        # negative almost always - reward_approach 3 pays 0 for submitting
        # an incomplete answer while every ineffective step costs
        # action_penalty, so a total clears 0 only when one step raises the
        # intersection by more than the steps around it cost. Ranking needs
        # something to rank.
        rollouts.append(rollout)

    return rollouts

# Monte Carlo Tree Search Implementation
class MCTSNode:
    def __init__(self, state, action=None, parent=None, reward=0.0, untried_actions=None):
        self.state = state  # {"grid", "objects", "max_int", "prev_action"} snapshot - see EnvironmentSimulator
        self.action = action  # Action that led to this node
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        self.immediate_reward = reward  # Reward received when reaching this state
        self.is_terminal = False
        self.untried_actions = untried_actions

    def is_fully_expanded(self, env_simulator):
        if self.untried_actions is None:
            self.untried_actions = list(env_simulator.all_actions)
        return len(self.untried_actions) == 0 and len(self.children) > 0

    def select_child(self, c=1.414):
        """Select child using UCB1 formula"""
        if not self.children:
            return None

        def ucb1(node):
            if node.visits == 0:
                return float('inf')
            return (node.total_reward / node.visits) + c * math.sqrt(math.log(self.visits) / node.visits)

        return max(self.children.values(), key=ucb1)

    def expand(self, env_simulator):
        """Expand node by adding a new child using environment simulator"""
        if self.untried_actions is None:
            self.untried_actions = list(env_simulator.all_actions)

        if not self.untried_actions:
            return None
        action = np.array(self.untried_actions.pop(0))

        # Purely functional - never touches env_simulator.env.
        next_state, reward, done, truncated, info = env_simulator.simulate_step(self.state, action)

        # Each child gets its own copy of the remaining actions: sharing
        # self.untried_actions between parent and child meant popping from
        # one silently drained the other's list too.
        child = MCTSNode(next_state, action, self, reward, untried_actions=list(self.untried_actions))
        child.is_terminal = done or truncated
        self.children[tuple(action)] = child

        return child

    def action_path(self):
        """The actions from the root down to this node, in order."""
        path = []
        node = self
        while node.parent is not None:
            path.append(list(node.action))
            node = node.parent
        path.reverse()
        return path

    def simulate(self, env_simulator, max_depth=10):
        """Random rollout from this node's own state snapshot - stays on
        copies throughout, never touches the live environment.

        A playout that reaches the target hands the whole sequence that got
        there to the simulator before returning. Only the reward used to
        travel back up the tree, so a playout could match the target exactly
        and the actions that did it went nowhere - the rollouts a search
        returns are built separately, from the tree's chosen action at each
        real step, and never see inside a playout. Measured over 8 tasks
        that happens twice, on one of them; twice is not many, and it was
        the whole of what the search had to show.
        """
        state = self.state
        total_reward = 0
        depth = 0
        done = False
        played = []
        while not done and depth < max_depth:
            action = env_simulator.sample_action()
            state, reward, done, truncated, _info = env_simulator.simulate_step(state, action)
            played.append(list(action))
            total_reward += reward
            depth += 1
            if state['max_int'] == env_simulator.env.target_int:
                env_simulator.record_solution(self.action_path() + played)
                break
            done = done or truncated
        return total_reward

    def backpropagate(self, reward):
        """Backpropagate reward up the tree"""
        self.visits += 1
        self.total_reward += reward

        if self.parent:
            self.parent.backpropagate(reward)

class PlayoutPolicy:
    """Which action a playout tries next, weighted by what the search has
    already learned about the pool.

    A uniform playout over the pool is not neutral here. Measured over
    629,331 simulated steps, 3.1% of them moved the intersection at all,
    the per-action mean ranges from -80 to +4, and 78.9% of the budget goes
    to actions with five or more fruitless tries already behind them. So a
    uniform draw spends almost everything re-confirming what the search
    knows, and the few actions that pay are drawn as rarely as the ones
    that wreck the grid.

    Two terms, and they answer different questions.

    `mean_delta` is the value: how far this action moved the intersection,
    on average, when it was tried. Normalised by the largest magnitude seen
    so far, because the raw range would collapse a softmax onto whichever
    action happened to score -80.

    The fruitless count is the confidence. A mean of zero from one try and
    a mean of zero from two hundred are the same number and different
    states of knowledge - the first is untested, the second is settled. Of
    the 20,383 actions tried twenty times without effect, 20,254 never had
    any effect at all, so the count predicts. The 129 that eventually did
    are why this lowers a weight rather than removing an action: nothing
    reaches probability zero.

    Counted on fruitless tries rather than on tries, because only 3.1% of
    steps move anything - a penalty on use as such would push down the
    handful of actions that work along with the rest.

    Measured over 12 tasks against the playout's default, which samples the
    padded action space and so spends ~91% of its draws on slots naming no
    object. Scored by the best intersection reached at any point in the
    search: +0.242 against +0.186, better on 4 tasks and worse on none.

    Scored instead by where rollouts *ended* it looks far worse (-0.236
    against +0.033), and that number is the one to distrust. A search is
    looking for a path and keeps the best prefix of one; a rollout that
    touched the target at step seven and wandered off by step twenty-five
    ends up recorded as damage. The endpoint metric also rewards a playout
    for doing nothing, which scores exactly 0.0 and solves nothing.
    """

    def __init__(self, actions, temperature=1.0, floor=0.1,
                 fruitless_penalty=1.0, refresh_every=128, rng=None):
        self.actions = [list(action) for action in actions]
        self.temperature = temperature
        self.floor = floor
        self.fruitless_penalty = fruitless_penalty
        self.refresh_every = refresh_every
        self.rng = rng or random
        self._index = {tuple(a): i for i, a in enumerate(self.actions)}
        size = len(self.actions)
        self._total_delta = np.zeros(size)
        self._tried = np.zeros(size)
        self._fruitless = np.zeros(size)
        self._since_refresh = refresh_every  # build the table on first use
        self._cumulative = None

    def record(self, action, delta):
        """What one simulated step of `action` did to the intersection."""
        index = self._index.get(tuple(int(x) for x in np.asarray(action).reshape(-1)))
        if index is None:
            return
        self._total_delta[index] += delta
        self._tried[index] += 1
        if delta == 0:
            self._fruitless[index] += 1
        self._since_refresh += 1

    def weights(self):
        """The sampling distribution, rebuilt from the counts."""
        tried = np.maximum(self._tried, 1)
        mean_delta = self._total_delta / tried
        spread = np.abs(mean_delta).max()
        value = mean_delta / spread if spread else mean_delta
        # Over tries + 1, so one fruitless try counts for little and a long
        # run of them approaches the full penalty - the same shape as a
        # visit-count term, and for the same reason.
        confidence = self._fruitless / (1.0 + self._tried)
        score = value - self.fruitless_penalty * confidence

        score = score - score.max()  # for exp(), not a change of ranking
        weights = np.exp(score / self.temperature)
        weights /= weights.sum()
        # Nothing reaches zero: an action can be wrong about a grid it has
        # not seen, and the pool is searched again after every round.
        return (1 - self.floor) * weights + self.floor / len(weights)

    def sample(self):
        """One action, drawn from the current distribution.

        Through a cumulative table and a binary search rather than
        numpy.random.choice, which is linear in the pool for every draw -
        a search takes hundreds of thousands of them, and the pool runs to
        over a thousand actions on a four-object grid. The table is rebuilt
        every `refresh_every` records instead of on every draw, for the
        same reason.
        """
        if self._cumulative is None or self._since_refresh >= self.refresh_every:
            self._cumulative = np.cumsum(self.weights())
            self._since_refresh = 0
        position = self.rng.random() * self._cumulative[-1]
        return list(self.actions[int(np.searchsorted(self._cumulative, position))])


class EnvironmentSimulator:
    """Runs functional (non-mutating) simulated steps against an
    ARCGridWorld for MCTS tree search.

    Every simulate_step() call runs env.simulate_action() on a state
    snapshot ({"grid", "objects", "max_int", "prev_action"}) and returns a
    new snapshot - env.grid/env.objects/env.max_int are never read from or
    written to here. The real environment is mutated exactly once per real
    step, by a normal env.step() call on whichever action the search
    settles on (see collect_mcts_rollouts) - never by this class, which is
    why simulating deep/wide trees doesn't need get_state()/set_state() at
    all: nothing real ever changes during the search.
    """
    def __init__(self, env, actions=None, policy=None):
        self.env = env
        self.action_space = env.action_space
        # Computed once, from the env rather than from the action space: the
        # space is padded to max_objects and most of it addresses empty slots.
        # See enumerate_actions. `actions` narrows it further to a pruned pool.
        self.all_actions = [list(action) for action in
                            (actions if actions is not None else enumerate_actions(env))]
        # None keeps the playout sampling the raw action space, padded slots
        # and all - which is what it has always done, and is not the same
        # thing as sampling the pool the tree expands over.
        self.policy = policy
        #: Action sequences that reached the target, shortest first.
        #: Candidates rather than results - they come off the simulator, and
        #: replay_solution has to confirm one against the real env before
        #: anything treats it as an answer.
        self.solutions = []

    def record_solution(self, actions, keep=8):
        """One sequence that reached the target, from wherever it was found.

        Shorter is better and duplicates are dropped: the same solution
        turns up again every time the tree revisits the branch holding it,
        and a long sequence with the answer somewhere in the middle is a
        worse trace to learn from than the short one that stops there.
        """
        candidate = [list(a) for a in actions]
        key = tuple(tuple(a) for a in candidate)
        if any(key == tuple(tuple(a) for a in known) for known in self.solutions):
            return
        self.solutions.append(candidate)
        self.solutions.sort(key=len)
        del self.solutions[keep:]

    def simulate_step(self, state, action):
        """state: {"grid", "objects", "max_int", "prev_action"} snapshot.
        Returns (next_state, reward, done, truncated, info)."""
        objects = self._copy_touched_objects(state['objects'], action)
        new_grid, new_objects, new_max_int, reward, done = self.env.simulate_action(
            action, objects, state['grid'], state['max_int'], state['prev_action'],
        )
        next_state = {
            'grid': new_grid, 'objects': new_objects, 'max_int': new_max_int,
            'prev_action': np.asarray(action),
        }
        if self.policy is not None:
            # The change in intersection, not the reward: the reward is
            # denominated in whichever reward_approach the env carries and
            # the approaches do not share a scale, while this is the same
            # count of cells under all of them.
            self.policy.record(action, new_max_int - state['max_int'])
        return next_state, reward, done, False, {}

    def _copy_touched_objects(self, objects, action):
        """Copy only the objects this action can mutate, not the whole
        list - World.apply_transform only ever mutates the two objects
        it's given, with one exception: the "object_recolor"
        emission-collision variant can also recolor a third object it
        looks up independently via cell2obj - copy every object in that
        one case, to stay safe.

        Nothing is copied for an action that will not run a transform at
        all. simulate_action returns early for submit, and treats an index
        past the objects this subtask has as an action that changes nothing
        - the action space carries a slot per max_objects whatever the
        subtask holds, so most sampled actions name an empty one. Neither
        reaches a transform, so neither can mutate anything.

        The check comes first, before the object_recolor branch rather than
        inside the per-index loop below it: that branch copies the whole
        list, and reached with an index naming nothing it copied every
        object on the grid to run no transform at all.
        """
        transform_name = self.env.actions_dict.get(int(action[0]), '')
        if transform_name == 'submit':
            return objects
        # Matching simulate_action's own bound, which is max_objects rather
        # than len(objects): objects past that are in the list but have no
        # index in the action space and nothing can name them.
        visible = min(len(objects), self.env.max_objects)
        first, second = int(action[1]), int(action[2])
        if first >= visible or second >= visible:
            return objects

        if 'object_recolor' in transform_name:
            return [deepcopy(obj) for obj in objects]
        objects = list(objects)
        for idx in {first, second}:
            objects[idx] = deepcopy(objects[idx])
        return objects

    def sample_action(self):
        """One action for a playout step.

        From the policy when there is one. Without it, from the raw action
        space - which is padded to max_objects whatever the subtask holds,
        so most of what it returns names an empty slot and does nothing.
        """
        if self.policy is not None:
            return self.policy.sample()
        return self.action_space.sample()

class MCTS:
    def __init__(self, env, max_iterations=1000, max_depth=10, c=1.414, actions=None):
        """`actions` narrows what the tree may expand into - the pool an
        iterative round has pruned down to (see rollout_preparation). None
        means every action the env offers."""
        self.env_simulator = EnvironmentSimulator(env, actions=actions)
        self.max_iterations = max_iterations
        self.max_depth = max_depth
        self.c = c
        self.all_actions = [tuple(action) for action in
                            (actions if actions is not None else enumerate_actions(env))]

    def search(self, initial_state):
        """Perform MCTS search from an initial state snapshot (see
        EnvironmentSimulator/ARCGridWorld.simulate_action) - never mutates
        the real environment."""
        root = MCTSNode(initial_state, untried_actions=copy(self.all_actions))

        for iteration in range(self.max_iterations):
            # Selection - traverse tree using UCB1
            node = root

            while not node.is_terminal and node.is_fully_expanded(self.env_simulator):
                node = node.select_child(self.c)
                if node is None:
                    break

            # Expansion - add new child if possible
            if not node.is_terminal and not node.is_fully_expanded(self.env_simulator):
                child = node.expand(self.env_simulator)
                if child:
                    node = child

            # A node can already be sitting on the answer - selection and
            # expansion reach states in their own right, and a task solved
            # in one action is solved before any playout begins. The
            # playout's own check only sees what the playout itself does.
            if node.state['max_int'] == self.env_simulator.env.target_int:
                self.env_simulator.record_solution(node.action_path())

            # Simulation - random rollout from current node
            if not node.is_terminal:
                simulation_reward = node.simulate(self.env_simulator, self.max_depth)
                total_reward = node.immediate_reward + simulation_reward
            else:
                total_reward = node.immediate_reward

            # Backpropagation - update all nodes in path
            node.backpropagate(total_reward)

        return root

    def get_best_action(self, root):
        """Get single best action from root"""
        if not root.children:
            return self.env_simulator.sample_action()

        # Select child with highest average reward
        best_child = max(root.children.values(),
                        key=lambda x: x.total_reward / max(x.visits, 1))

        return best_child.action

    def get_best_action_sequence(self, root, max_length=10):
        """Extract best action sequence from MCTS tree"""
        sequence = []
        node = root

        for _ in range(max_length):
            if not node.children:
                break

            # Select child with highest average reward
            best_child = max(node.children.values(),
                           key=lambda x: x.total_reward / max(x.visits, 1))

            sequence.append(best_child.action)
            node = best_child

        return sequence

def replay_solution(env, actions, submit_index=None) -> Dict[str, Any]:
    """Run a candidate solution on the real env and return it as a rollout,
    or None if it does not in fact solve the task.

    Replayed rather than trusted. A sequence found in a playout is a claim
    made by the simulator, and the simulator exists precisely because it is
    not the env - a candidate that only solves there is a bug to fail on
    rather than an answer to keep. The replay also produces the
    observations, which a playout never builds.

    The trace ends on submit, appended rather than searched for. The episode
    is over by then: the env ends it the moment the intersection reaches the
    target, so nothing in a search ever needs to decide to submit and no
    playout would have found the action. A trace meant to be imitated needs
    it anyway - an agent that never sees an episode end on submit has no
    reason to learn to end one.

    `env` is reset first and left holding the state the replay reached.
    """
    if submit_index is None:
        submit_index = next((i for i, name in env.actions_dict.items()
                             if name == 'submit'), None)
    env.reset()
    rollout = {'observations': [], 'actions': [], 'rewards': [],
               'dones': [], 'infos': []}
    total_reward = 0.0
    for action in actions:
        observation, reward, done, truncated, info = env.step(np.asarray(action))
        rollout['observations'].append(observation)
        rollout['actions'].append(list(action))
        rollout['rewards'].append(reward)
        rollout['dones'].append(done)
        rollout['infos'].append(info)
        total_reward += reward
        if env.max_int == env.target_int or done or truncated:
            break

    if env.max_int != env.target_int:
        return None

    if submit_index is not None:
        rollout['actions'].append([submit_index, 0, 0])
        rollout['rewards'].append(env._submit_reward(env.max_int))
        rollout['dones'].append(True)
        rollout['infos'].append({'appended_submit': True})
        total_reward += rollout['rewards'][-1]

    rollout['total_reward'] = total_reward
    rollout['length'] = len(rollout['actions'])
    rollout['solved'] = True
    rollout['max_int'] = int(env.max_int)
    rollout['base_int'] = int(env.base_int)
    rollout['target_int'] = int(env.target_int)
    return rollout


def env_state_snapshot(env) -> Dict[str, Any]:
    """Build a {"grid", "objects", "max_int", "prev_action"} snapshot of an
    ARCGridWorld's current real state, for MCTS.search()/EnvironmentSimulator
    to explore from without ever touching the real env. Deep-copies every
    object once here (cheap - happens once per real step, not per simulated
    tree node); simulated steps then only copy whichever 1-2 objects they
    actually touch, relative to whatever their parent node already holds."""
    return {
        'grid': env.grid.copy(),
        'objects': [deepcopy(obj) for obj in env.objects],
        'max_int': env.max_int,
        'prev_action': env.prev_action.copy() if env.prev_action is not None else None,
    }

def collect_mcts_rollouts(env,
                          n_rollouts: int = 50,
                          mcts_iterations: int = 500,
                          max_episode_len: int = 50,
                          actions=None,
                          c: float = 1.414) -> List[Dict[str, Any]]:
    """Collect rollouts using MCTS for action selection. MCTS search itself
    runs entirely on a snapshot of the env's state (see
    EnvironmentSimulator) - only the action it settles on for each real
    step is ever applied to `env` for real, via a normal env.step().

    `actions` restricts the tree to a pruned pool; None searches everything
    the env offers. `c` is UCB1's exploration constant - worth naming here
    because it is the honest knob for how widely the tree looks. It used to
    be reachable only by changing reward_approach, which divides every step
    reward by a different max_reward (5M under approach 1, 11M under 2, so
    approach 2's rewards are 2.2x smaller against the same fixed c) and so
    moved exploration as a side effect of a setting that says nothing about
    exploration.
    """
    rollouts = []
    mcts = MCTS(env, max_iterations=mcts_iterations, actions=actions, c=c)

    print(f"Collecting {n_rollouts} MCTS-guided rollouts...")

    for i in tqdm(range(n_rollouts)):
        observation, _ = env.reset()  # Reset environment for each rollout
        done = False
        truncated = False

        rollout = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'infos': []
        }

        total_reward = 0
        step_count = 0
        #: env.max_int after each step. Despite the name it is the current
        #: intersection, not a running maximum (arc_env assigns it from
        #: maximal_intersection every step), so it falls as well as rises -
        #: which is the whole reason the trace needs cutting.
        reached = []

        while not (done or truncated) and step_count < max_episode_len:
            # MCTS explores on a snapshot of the current real state.
            root = mcts.search(env_state_snapshot(env))
            action = mcts.get_best_action(root)

            # Only the chosen action is ever applied to the real environment.
            next_observation, reward, done, truncated, info = env.step(action)
            rollout['observations'].append(observation)
            rollout['actions'].append(action)
            rollout['rewards'].append(reward)
            rollout['dones'].append(done)
            rollout['infos'].append(info)
            reached.append(int(env.max_int))

            total_reward += reward
            step_count += 1
            observation = next_observation

        # Cut at the best state the rollout passed through. What follows the
        # peak is the search walking back downhill, and measured over four
        # configurations that walk is large: rollouts ended a mean 0.04-0.21
        # of the gap *below* where they started while their peaks stood at
        # +0.25. Cloning the whole trace teaches both halves - the prefix
        # that found something and the suffix that undid it - so a
        # behavioural-cloning trace has to stop where the grid was best.
        #
        # It also stops a solved rollout from being recorded as unsolved:
        # `solved` used to read env.max_int after the loop, which is the
        # final state, so a rollout that reached the target and then moved
        # off it scored as a failure.
        peak = max(reached) if reached else int(env.max_int)
        cut = len(reached)
        if reached and peak > int(env.base_int):
            cut = reached.index(peak) + 1
            for key in ('observations', 'actions', 'rewards', 'dones', 'infos'):
                rollout[key] = rollout[key][:cut]
            total_reward = sum(rollout['rewards'])
        rollout['truncated_at_peak'] = cut < step_count
        rollout['total_reward'] = total_reward
        rollout['length'] = cut
        # Not any(dones): submit_grid ends the episode with done=True
        # whatever it submitted, so "the episode finished" and "the answer
        # was right" are different questions and only step_intersection's
        # done answers the second. Read off the peak rather than the env,
        # because the trace now ends at the peak - see the cut above.
        rollout['solved'] = bool(peak == env.target_int)
        # Progress, alongside the reward. total_reward is denominated in
        # whichever reward_approach the env was built with, and the
        # approaches do not share a scale - approach 1 runs -4..+4 over the
        # milestones where approach 2 runs -4..+10 - so rewards from two of
        # them cannot be compared, and a search that is doing better can
        # score lower. The intersection is the same count of cells whatever
        # the approach, which makes these three the comparable record. The
        # peak, not env.max_int, since the trace was cut to end there.
        rollout['max_int'] = int(peak)
        rollout['base_int'] = int(env.base_int)
        rollout['target_int'] = int(env.target_int)

        # Every rollout is kept, and select_best_rollouts ranks them.
        # Selecting on total_reward > 0 here threw away the entire run on
        # this reward scale: measured over 42 ARC-AGI-1 tasks with the real
        # transform vocabulary, a rollout's total is 0.0 or slightly
        # negative almost always - reward_approach 3 pays 0 for submitting
        # an incomplete answer while every ineffective step costs
        # action_penalty, so a total clears 0 only when one step raises the
        # intersection by more than the steps around it cost. Ranking needs
        # something to rank.
        rollouts.append(rollout)

    # Anything a playout solved, verified and turned into a trace. The
    # search's own rollouts cannot contain these: they follow the tree's
    # chosen action at each real step, and a playout is explored and then
    # discarded. Verified after the loop rather than as they arrive, so the
    # replay does not disturb the env in the middle of a rollout.
    for candidate in mcts.env_simulator.solutions:
        solved = replay_solution(env, candidate)
        if solved is not None:
            rollouts.insert(0, solved)

    return rollouts

def rollout_preparation(env,
                        method: str = "random",  # "random or "mcts"
                        n_initial_rollouts: int = 100,
                        top_k: int = 10,
                        min_len: int = 0,
                        reward_threshold: float = 0.0,
                        keep_best_actions: int = 0,
                        mcts_iterations: int = 500,
                        n_rounds: int = 1,
                        keep_fraction: float = 0.5,
                        min_pool: int = 4,
                        c: float = 1.414,
                       ) -> List[Dict[str, Any]]:
    """Search a task's action space and return the rollouts worth reading.

    `env` is an ARCGridWorld, not a vec env and not an agent: the phases
    below call reset()/step() expecting gymnasium's (obs, info) pair, and
    MCTS reaches for env.grid/env.objects/env.max_int, none of which survive
    DummyVecEnv or a wrapper stack.

    Args:
        env: the environment to search
        method: "random" for promising-action-focused rollouts, "mcts" for
            MCTS-guided ones
        n_initial_rollouts: Number of rollouts to collect
        top_k: Number of best rollouts to select
        min_len: Minimum episode length, applied only to unsolved rollouts
        reward_threshold: Minimum reward for promising actions
        keep_best_actions: fall back to this many best actions when none
            clears reward_threshold - see identify_promising_actions
        mcts_iterations: MCTS iterations per decision
        n_rounds: how many collect-then-prune rounds to run. 1 is a single
            pass over the whole pool. More rounds spend the same rollouts on
            a shrinking set of actions: the first round is a broad sample,
            each later one re-searches only what survived, so the depth
            reachable within mcts_iterations grows as the branching drops.
        keep_fraction: share of the pool each round keeps, by the mean
            reward of the rollouts an action appeared in
        min_pool: never prune below this many actions - a pool of one makes
            every rollout identical and the round pointless

    Returns:
        The selected rollouts from every round, best first
    """
    # Step 1: Test individual actions
    print("Phase 1: Testing individual actions...")
    action_results = test_individual_actions(env)

    # Step 2: Identify promising actions
    print("Phase 2: Identifying promising actions...")
    promising_actions = identify_promising_actions(action_results, reward_threshold,
                                                   keep_best=keep_best_actions)

    # Step 3: Collect rollouts, narrowing the pool between rounds.
    pool = enumerate_actions(env)
    collected: List[Dict[str, Any]] = []
    for round_no in range(1, max(n_rounds, 1) + 1):
        print(f"Phase 3.{round_no}: Collecting rollouts using {method} method "
              f"over {len(pool)} actions...")
        if method == "random":
            rollouts = collect_random_rollouts(env, promising_actions or pool,
                                               n_initial_rollouts)
        elif method == "mcts":
            rollouts = collect_mcts_rollouts(env, n_initial_rollouts, mcts_iterations,
                                             actions=pool, c=c)
        else:
            raise ValueError(f"Unknown method: {method}")
        collected.extend(rollouts)

        if round_no == n_rounds:
            break
        pruned = prune_actions(pool, rollouts, keep_fraction=keep_fraction, min_pool=min_pool)
        if len(pruned) == len(pool):
            print("Pruning changed nothing; stopping early")
            break
        pool = pruned
        # The pool the next round searches is also the pool it samples from.
        promising_actions = [a for a in promising_actions if list(a) in pool]

    # Step 4: Select best rollouts
    print("Phase 4: Selecting best rollouts...")
    best_rollouts = select_best_rollouts(collected, top_k=top_k, min_len=min_len)

    return best_rollouts


def action_statistics(rollouts: List[Dict[str, Any]]) -> Dict[tuple, Dict[str, float]]:
    """Per action, how often it was used and what the rollouts using it
    scored - the record an iterative round prunes on.

    Scored by the *mean* of the rollouts it appears in, not the sum: an
    action tried in twenty rollouts would otherwise outrank a better one
    tried in two purely by turning up more often. `solved` counts the
    rollouts that finished the task, which is the signal worth keeping even
    when the reward says little.
    """
    stats: Dict[tuple, Dict[str, float]] = {}
    for rollout in rollouts:
        for action in rollout['actions']:
            key = tuple(int(x) for x in np.asarray(action).reshape(-1))
            entry = stats.setdefault(key, {'uses': 0, 'rollouts': 0,
                                           'total_reward': 0.0, 'solved': 0})
            entry['uses'] += 1
        for key in {tuple(int(x) for x in np.asarray(a).reshape(-1))
                    for a in rollout['actions']}:
            entry = stats[key]
            entry['rollouts'] += 1
            entry['total_reward'] += float(rollout['total_reward'])
            entry['solved'] += int(bool(rollout.get('solved')))
    for entry in stats.values():
        entry['mean_reward'] = entry['total_reward'] / entry['rollouts']
    return stats


def prune_actions(pool: List[List[int]], rollouts: List[Dict[str, Any]],
                   keep_fraction: float = 0.5, min_pool: int = 4) -> List[List[int]]:
    """The share of `pool` worth searching again, best first.

    An action no rollout used has no record, and is dropped before one that
    scored badly: the search had the chance to reach for it and did not.
    Actions that appeared in a solving rollout are kept whatever the
    fraction says - dropping something that finished the task to honour a
    ratio would be the same mistake the length filter used to make.
    """
    if not rollouts or len(pool) <= min_pool:
        return list(pool)

    stats = action_statistics(rollouts)
    keepers = {key for key, entry in stats.items() if entry['solved']}
    target = max(min_pool, int(len(pool) * keep_fraction), len(keepers))

    ranked = sorted(stats, key=lambda k: stats[k]['mean_reward'], reverse=True)
    kept = list(keepers) + [k for k in ranked if k not in keepers]
    kept = kept[:target]

    # Back to the pool's own ordering and shape, so callers see a subset of
    # what they passed rather than a differently-shaped list.
    kept_set = set(kept)
    return [action for action in pool if tuple(action) in kept_set]

def select_best_rollouts(rollouts: List[Dict[str, Any]], top_k: int = 10, min_len: int = 0) -> List[Dict[str, Any]]:
    """The top k rollouts by total reward.

    `min_len` drops rollouts too short to have explored anything - one step
    of submit, mostly. It never drops a rollout that solved the task: an
    episode ends the moment the intersection reaches the target, so a
    solution is short *because* it worked, and a length floor cuts exactly
    what the search is for. That is why it defaults to 0 rather than the 5
    it used to: with max_episode_len at 25 and solutions two or three steps
    long, 5 discarded them and kept the wandering.

    Args:
        rollouts: List of rollout dictionaries
        top_k: Number of best rollouts to select
        min_len: Minimum episode length, applied only to unsolved rollouts

    Returns:
        List of selected rollout dictionaries
    """
    rollouts = [r for r in rollouts if r.get('solved') or r['length'] >= min_len]

    # Solved first, then by total reward. Reward alone can rank a wandering
    # rollout above one that finished the task: an episode that solves it in
    # two steps collects two steps' worth, while a long one accumulates
    # whatever partial credit it picked up along the way.
    sorted_rollouts = sorted(rollouts, key=lambda x: (bool(x.get('solved')), x['total_reward']),
                              reverse=True)

    # Select top k
    selected_rollouts = sorted_rollouts[:top_k]

    print(f"Selected {len(selected_rollouts)} best rollouts from {len(rollouts)} total")
    for i, rollout in enumerate(selected_rollouts[:min(10, len(selected_rollouts))]):
        print(f"Rollout {i + 1}: Reward = {rollout['total_reward']:.2f}, Steps = {rollout['length']}")

    return selected_rollouts

def reconstruct_rollout(grids, actions, rewards, infos):
    rollout = {}
    rollout['observations'] = [{"grid": grid} for grid in grids]
    rollout['actions'] = actions
    rollout['rewards'] = rewards
    rollout['infos'] = infos
    return rollout

def extract_promising_actions(rollouts, feasible_actions, k=10):
    """Which transformations the best rollouts actually used, most-used first.

    `feasible_actions` is the env's `{index: name}` mapping (ARCGridWorld's
    actions_dict), which is the only place the correspondence lives - this
    used to read it out of `rollouts[0]['infos'][0]['action_mapping']`, a key
    ARCGridWorld has never put in `info`, and then iterate `feasible_actions`
    as if it were a collection of names, comparing an int against a string.
    Neither could run; neither ever did, because the reward filter upstream
    left `rollouts` empty every time.

    Counted rather than set-collected: an action three of the best rollouts
    reached for says more than one that appeared once, and the caller (the
    analyst's prompt) wants them in priority order.
    """
    if not rollouts:
        return []
    counts = collections.Counter()
    for rollout in rollouts[:k]:
        for action in rollout['actions']:
            index = int(np.asarray(action).reshape(-1)[0])
            name = feasible_actions.get(index)
            if name is not None:
                counts[name] += 1
    return [name for name, _ in counts.most_common()]
