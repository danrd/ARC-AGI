import numpy as np
import torch
import itertools
import random
import math
from typing import Dict, Any, List
from copy import copy
from tqdm import tqdm
from rl.training import create_agent, create_vec_env
from data.configs.rl_configs import rl_config, load_PPO_config


def process_observations(observations, device, pad_inp=True, multi_env=False):
    """
    Process different types of observations to make them compatible with the policy.

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

def test_individual_actions(env, max_actions: int = None) -> Dict[int, Dict[str, Any]]:
    """
    Test each action individually (episode length 1) to identify promising actions.

    Args:
        env: Environment to test actions in
        max_actions: Maximum number of actions to test (None = test all)

    Returns:
        Dictionary mapping action_id to results (reward, observation, etc.)
    """
    action_results = {}

    all_actions = itertools.product(*[range(x) for x in env.action_space.nvec]) # for each dim in action_space generate all possible values, then - cartesian product
    all_actions = [list(action) for action in list(all_actions)] # iterator to list than each element - list instead of tuple

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
                              reward_threshold: float = 0.0) -> List[int]:
    """
    Identify actions that resulted in positive rewards.

    Args:
        action_results: Results from test_individual_actions
        reward_threshold: Minimum reward to consider an action promising

    Returns:
        List of promising action IDs
    """
    promising_actions = []

    for action_id, result in action_results.items():
        if result['reward'] > reward_threshold:
            promising_actions.append(action_id)

    # Sort by reward (descending)
    promising_actions.sort(key=lambda x: action_results[x]['reward'], reverse=True)

    print(f"Found {len(promising_actions)} promising actions with reward > {reward_threshold}")
    for i, action_id in enumerate(promising_actions[:10]):  # Show top 10
        reward = action_results[action_id]['reward']
        print(f"  Action {action_id}: reward = {reward:.4f}")

    return promising_actions

def collect_random_rollouts(env,
                           promising_actions: List[List[int]],
                           n_rollouts: int = 100,
                           max_episode_len: int = 50) -> List[Dict[str, Any]]:
    """
    Collect rollouts focusing on promising actions but with some exploration.

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

        # Only keep rollouts with positive total reward
        if total_reward > 0:
            rollouts.append(rollout)

    return rollouts

# Monte Carlo Tree Search Implementation
class MCTSNode:
    def __init__(self, observation, action=None, parent=None, reward=0.0, untried_actions=None):
        self.observation = observation
        self.action = action  # Action that led to this node
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        self.immediate_reward = reward  # Reward received when reaching this state
        self.is_terminal = False
        self.untried_actions = untried_actions

    def is_fully_expanded(self, action_space):
        if self.untried_actions is None:
            self.untried_actions = list(itertools.product(*[range(x) for x in action_space.nvec]))
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
            self.untried_actions = list(itertools.product(*[range(x) for x in env_simulator.action_space.nvec]))

        if not self.untried_actions:
            return None
        action = np.array(self.untried_actions.pop(0))

        # Use simulator to get next state without affecting main environment
        next_observation, reward, done, truncated, info = env_simulator.simulate_step(
            self.observation, action
        )

        child = MCTSNode(next_observation, action, self, reward, untried_actions=self.untried_actions)
        child.is_terminal = done or truncated
        self.children[tuple(action)] = child

        return child

    def simulate(self, env_simulator, max_depth=10):
        """Simulate a random rollout from this node using simulator"""
        total_reward = 0
        depth = 0
        done = False
        original_state =  env_simulator.env.get_state()
        while not done and depth < max_depth:
            action = env_simulator.sample_action()
            _next_obs, reward, done, truncated, _ = env_simulator.env.step(action)
            total_reward += reward
            depth += 1
            done = done or truncated
        env_simulator.env.set_state(original_state)
        return total_reward

    def backpropagate(self, reward):
        """Backpropagate reward up the tree"""
        self.visits += 1
        self.total_reward += reward

        if self.parent:
            self.parent.backpropagate(reward)

class EnvironmentSimulator:
    """
    Wrapper that provides state simulation capabilities.
    This class should be adapted based on your specific environment.
    """
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def simulate_step(self, observation, action):
        """
        Simulate taking an action from a given observation.
        This is the key method that needs environment-specific implementation.
        """
        original_state = self.env.get_state()
        next_obs, reward, done, truncated, info = self.env.step(action)
        self.env.set_state(original_state) # Restore original state
        return next_obs, reward, done, truncated, info

    def _observation_to_key(self, observation):
        """Convert observation to hashable key"""
        if isinstance(observation, np.ndarray):
            return tuple(observation.flatten())
        elif isinstance(observation, dict):
            return tuple(sorted(observation.items()))
        else:
            return observation

    def sample_action(self):
        """Sample random action"""
        return self.action_space.sample()

class MCTS:
    def __init__(self, env, max_iterations=1000, max_depth=10, c=1.414):
        self.env_simulator = EnvironmentSimulator(env)
        self.max_iterations = max_iterations
        self.max_depth = max_depth
        self.c = c
        self.all_actions = list(itertools.product(*[range(x) for x in env.action_space.nvec]))

    def search(self, initial_observation):
        """Perform MCTS search from initial observation"""
        root = MCTSNode(initial_observation, untried_actions=copy(self.all_actions))

        for iteration in range(self.max_iterations):
            # Selection - traverse tree using UCB1
            node = root

            while not node.is_terminal and node.is_fully_expanded(self.env_simulator.action_space):
                node = node.select_child(self.c)
                if node is None:
                    break

            # Expansion - add new child if possible
            if not node.is_terminal and not node.is_fully_expanded(self.env_simulator.action_space):
                child = node.expand(self.env_simulator)
                if child:
                    node = child

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

def collect_mcts_rollouts(env,
                          n_rollouts: int = 50,
                          mcts_iterations: int = 500,
                          max_episode_len: int = 50) -> List[Dict[str, Any]]:
    """
    Collect rollouts using MCTS for action selection.
    Now properly manages environment state.
    """
    rollouts = []
    mcts = MCTS(env, max_iterations=mcts_iterations)

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

        while not (done or truncated) and step_count < max_episode_len:
            # Use MCTS to select action (this doesn't modify env state)
            root = mcts.search(observation)
            action = mcts.get_best_action(root)

            # Now take the actual step in the real environment
            next_observation, reward, done, truncated, info = env.step(action)
            rollout['observations'].append(observation)
            rollout['actions'].append(action)
            rollout['rewards'].append(reward)
            rollout['dones'].append(done)
            rollout['infos'].append(info)

            total_reward += reward
            step_count += 1
            observation = next_observation

        rollout['total_reward'] = total_reward
        rollout['length'] = step_count

        # Only keep rollouts with positive total reward
        if total_reward > 0:
            rollouts.append(rollout)

    return rollouts

def rollout_preparation(env,
                        method: str = "random",  # "random or "mcts"
                        n_initial_rollouts: int = 100,
                        top_k: int = 10,
                        min_len: int = 5,
                        reward_threshold: float = 0.0,
                        mcts_iterations: int = 500
                       ) -> List[Dict[str, Any]]:
    """
    Enhanced rollout preparation with individual action testing and sophisticated exploration.

    Args:
        model: Model with environment
        method: "focused" for promising-action-focused rollouts, "mcts" for MCTS-guided rollouts
        n_initial_rollouts: Number of rollouts to collect
        top_k: Number of best rollouts to select
        min_len: Minimum episode length
        reward_threshold: Minimum reward for promising actions
        mcts_iterations: MCTS iterations per decision

    Returns:
        Dict with best rollouts dictionaries
    """
    # Step 1: Test individual actions
    print("Phase 1: Testing individual actions...")
    action_results = test_individual_actions(env)

    # Step 2: Identify promising actions
    print("Phase 2: Identifying promising actions...")
    promising_actions = identify_promising_actions(action_results, reward_threshold)

    # Step 3: Collect rollouts using selected method
    print(f"Phase 3: Collecting rollouts using {method} method...")
    if method == "random":
        rollouts = collect_random_rollouts(
            env, promising_actions, n_initial_rollouts
        )
    elif method == "mcts":
        rollouts = collect_mcts_rollouts(
            env, n_initial_rollouts, mcts_iterations
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # Step 4: Select best rollouts
    print("Phase 4: Selecting best rollouts...")
    best_rollouts = select_best_rollouts(rollouts, top_k=top_k, min_len=min_len)

    return best_rollouts

def select_best_rollouts(rollouts: List[Dict[str, Any]], top_k: int = 10, min_len: int = 5) -> List[Dict[str, Any]]:
    """
    Select the top k rollouts based on total reward and episode length.

    Args:
        rollouts: List of rollout dictionaries
        top_k: Number of best rollouts to select
        min_len: Minimum length of episode to consider

    Returns:
        List of selected rollout dictionaries
    """
    # Filter by minimum length
    rollouts = list(filter(lambda x: x['length'] > min_len, rollouts))

    # Sort rollouts by total reward (descending)
    sorted_rollouts = sorted(rollouts, key=lambda x: x['total_reward'], reverse=True)

    # Select top k
    selected_rollouts = sorted_rollouts[:top_k]

    print(f"Selected {len(selected_rollouts)} best rollouts from {len(rollouts)} total")
    for i, rollout in enumerate(selected_rollouts[:min(10, len(selected_rollouts))]):
        print(f"Rollout {i+1}: Reward = {rollout['total_reward']:.2f}, Steps = {rollout['length']}")

    return selected_rollouts

def reconstruct_rollout(grids, actions, rewards, infos):
    rollout = {}
    rollout['observations'] = [{"grid":grid} for grid in grids]
    rollout['actions'] = actions
    rollout['rewards'] = rewards
    rollout['infos'] = infos
    return rollout

def extract_promising_actions(rollouts, feasible_actions, k=10):
    sorted_actions = []
    actions_dict = rollouts[0]['infos'][0]['action_mapping']
    idx2name = {v:k for k,v in actions_dict.items()}
    all_actions = [action.tolist()[0]
                   for rollout in rollouts[:k]
                   for action in rollout['actions']
                  ]
    action_names = list(set([idx2name[action] for action in all_actions]))
    for action in feasible_actions:
        for action_realization in action_names:
            if action in action_realization:
                sorted_actions.append(action)
                break
    return sorted_actions

def action_exploration(subtask, config):
    test_vec_env = create_vec_env(subtask, n_envs=rl_config['n_envs'], max_episode_len=rl_config['max_episode_len'],
                             right_placement_reward=rl_config['right_placement_reward'],  action_penalty=rl_config['action_penalty'],
                             repetitive_actions_penalty=rl_config['repetitive_actions_penalty'], seed=42, font_color=rl_config['font_color'],
                             padding=rl_config['padding'], input_pattern=rl_config['input_pattern'], milestones_rewards=rl_config['milestones_rewards'],
                             pad_val=rl_config['pad_val'], reward_approach=rl_config['reward_approach'],
                             feasible_actions=rl_config['feasible_actions'], repr_level=rl_config['repr_level'],
                             observation_space_elements=rl_config['observation_space_elements'])
    agent = create_agent(rl_config=rl_config, vec_env=test_vec_env, model_config=load_PPO_config())
    best_rollouts = rollout_preparation(agent, method="mcts",  n_initial_rollouts=500, top_k=5, mcts_iterations=10)
    promising_actions = extract_promising_actions(best_rollouts, rl_config['feasible_actions'])
    return promising_actions
