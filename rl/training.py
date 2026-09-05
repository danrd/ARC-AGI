import gymnasium as gym
import functools
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3 import PPO
from rl.arc_env import ARCGridWorld, MAX_OBJECTS
from rl.evaluation import evaluate_ARC_policy
from stable_baselines3.common.callbacks import CallbackList
from rl.callbacks import MonitorCallback, ARCLogger
from rl.mcts import rollout_preparation, extract_promising_actions
from utils.utils import seed_everything
from utils.plotting import plot_grid
from data.configs.rl_configs import load_PPO_config

def create_agent(rl_config:dict, vec_env, model_config:dict=None, path_to_pretrained:str=None, agent_init=None):
    """Build the agent to train with: a caller-supplied one, one restored
    from a checkpoint, or a fresh one configured from rl_config.

    Every branch has to either return an agent or say why it can't. A branch
    that falls through returns whatever the name happened to be bound to,
    which for a function-local name is nothing at all - an UnboundLocalError
    naming `agent`, which says nothing about the argument that caused it.
    """
    if agent_init:
        # Pointed at this vec_env, not left holding the one it was built on.
        # train_on_task hands the same agent down the subtasks, each with a
        # freshly built env - and without this the agent kept rolling out in
        # subtask 0 for every one of them while being evaluated on the env it
        # never saw. Five accuracies that looked like five trainings were one
        # policy measured on five grids.
        if vec_env is not None:
            agent_init.set_env(vec_env)
        return agent_init
    if path_to_pretrained:
        # A classmethod on the algorithm, not a method on an instance - there
        # is no agent yet at this point. The env goes in with it: a model
        # loaded without one has no observation/action space to check against
        # and cannot .learn().
        return PPO.load(path_to_pretrained, env=vec_env)
    if rl_config['model_type'] != 'PPO':
        raise ValueError(
            f"Unsupported model_type {rl_config['model_type']!r} - this builds PPO agents. "
            "Pass agent_init=<your agent> to use anything else."
        )
    PPO_config = load_PPO_config()
    if model_config:
        PPO_config.update(model_config)
    policy = PPO_config['policy'] if PPO_config['policy'] != 'default' else "MultiInputPolicy"
    policy_kwargs = {'net_arch':dict(pi=PPO_config['actor_arch'], vf=PPO_config['critic_arch']), 'activation_fn':PPO_config['activation_fn'],
                     'action_heads':PPO_config['action_heads'],
                     'features_extractor_kwargs':{'extr_arch': PPO_config['extr_arch']}}
    # gae_lambda among them: the config has carried a value for it all
    # along and this call dropped it, so every agent ever built here ran on
    # PPO's own default of 0.95 while the config said 0.9.
    return PPO(policy, vec_env, batch_size=PPO_config['batch_size'], n_steps=PPO_config['n_steps'], verbose=PPO_config['verbose'],
               n_epochs=PPO_config['n_epochs'], gamma=PPO_config['gamma'], gae_lambda=PPO_config['gae_lambda'],
               max_grad_norm=PPO_config['max_grad_norm'],
               learning_rate=PPO_config['learning_rate'], clip_range=PPO_config['clip_range'], ent_coef=PPO_config['ent_coef'],
               vf_coef=PPO_config['vf_coef'], use_sde = PPO_config['use_sde'], policy_kwargs=policy_kwargs)


def create_ARC_env(subtask, max_episode_len=50, right_placement_reward=5.0, action_penalty=1.0,
                   repetitive_actions_penalty=1.0, seed=None, font_color=0, padding=False, input_pattern=False,
                   milestones_rewards=(1, 2, 3, 4), pad_val=10, reward_approach=1, repr_level=1,
                   feasible_actions={0:"submit"}, observation_space_elements = ["objects_emb", "relations_emb"],
                   observation_grid_shape=None, max_objects=MAX_OBJECTS,
                  ):
    """Auxiliary function for creating environments to create vectorized environment."""
    gym.envs.register(
     id='ARC-Gridworld-v0',
     entry_point='rl.arc_env:create_env',
     kwargs={})
    env = gym.make('ARC-Gridworld-v0', max_episode_len=max_episode_len, right_placement_reward=right_placement_reward,
                   action_penalty=action_penalty, repetitive_actions_penalty=repetitive_actions_penalty,
                   seed=seed, font_color=font_color, padding=padding, input_pattern=input_pattern, repr_level=repr_level,
                   reward_approach=reward_approach, milestones_rewards=milestones_rewards, pad_val=pad_val,
                   feasible_actions=feasible_actions, observation_space_elements=observation_space_elements,
                   observation_grid_shape=observation_grid_shape,
                   max_objects=max_objects,
                  )
    # .unwrapped: gym.make() wraps env in OrderEnforcing/PassiveEnvChecker,
    # and current gymnasium no longer forwards custom methods through
    # wrappers' __getattr__ - set_subtask only exists on the raw ARCGridWorld.
    env.unwrapped.set_subtask(subtask)
    env.action_space.seed(seed)
    return env

def create_vec_env(subtasks, n_envs:int, max_episode_len=50, right_placement_reward=5.0, action_penalty=1.0,
                   repetitive_actions_penalty=1.0, seed=None, font_color=0, padding=False, input_pattern=False,
                   milestones_rewards=(1, 2, 3, 4), pad_val=10, reward_approach=1, repr_level=1,
                   feasible_actions={0:"submit"}, observation_space_elements = ["objects_emb", "relations_emb"],
                   observation_grid_shape=None, max_objects=MAX_OBJECTS):
    """Auxiliary function for creating vectorized environment."""
    envs = [functools.partial(create_ARC_env, subtask=subtask, max_episode_len=max_episode_len, right_placement_reward=right_placement_reward,
                              action_penalty=action_penalty, repetitive_actions_penalty=repetitive_actions_penalty,
                              seed=seed, font_color=font_color, padding=padding, input_pattern=input_pattern, repr_level=repr_level,
                              reward_approach=reward_approach, milestones_rewards=milestones_rewards, pad_val=pad_val,
                              observation_space_elements=observation_space_elements,
                              observation_grid_shape=observation_grid_shape,
                              max_objects=max_objects,
                              feasible_actions=feasible_actions) for subtask in subtasks for i in range(n_envs)]
    vec_env = VecMonitor(DummyVecEnv(envs))
    return vec_env

def train_on_subtasks(subtasks, rl_config:dict, PPO_config:dict=None, agent_init=None,
                      path_to_pretrained=None, verbose=False, plot_grid_pred=False, debug=False,
                      extra_callback=None):
    """Train one agent on `subtasks` at once: one vec env holding all of
    them, so one rollout buffer carries steps from every subtask and the
    policy update sees the task rather than one of its examples.

    train_on_subtask is this with a single-element list - kept under its own
    name because notebooks call it.

    `extra_callback` runs alongside MonitorCallback rather than instead of
    it - the seam for watching a run without editing the loop, which is how
    the question "does a rollout ever close any of the distance" gets
    asked at all: every held-out score comes from a deterministic
    evaluation and cannot see what the stochastic rollouts did.
    """
    seed = rl_config['seed']
    seed_everything(seed)
    vec_env = create_vec_env(subtasks, n_envs=rl_config['n_envs'], max_episode_len=rl_config['max_episode_len'], repr_level=rl_config['repr_level'],
                             right_placement_reward=rl_config['right_placement_reward'],  action_penalty=rl_config['action_penalty'],
                             repetitive_actions_penalty=rl_config['repetitive_actions_penalty'], seed=seed, font_color=rl_config['font_color'],
                             padding=rl_config['padding'], input_pattern=rl_config['input_pattern'], milestones_rewards=rl_config['milestones_rewards'],
                             pad_val=rl_config['pad_val'], reward_approach=rl_config['reward_approach'],
                             feasible_actions=rl_config['feasible_actions'], observation_space_elements=rl_config['observation_space_elements'],
                             observation_grid_shape=rl_config.get('observation_grid_shape'),
                             max_objects=rl_config.get('max_objects', MAX_OBJECTS))
    # verbose passed through: the callback's own default is True, and a
    # verbose evaluation prints and calls plot_grid - at rl_config's
    # eval_freq that is a figure every few steps of training.
    callback = MonitorCallback(vec_env, eval_freq=rl_config['eval_freq'], n_eval_episodes=rl_config['n_eval_episodes'],
                                   log_path=rl_config['log_path'], debug=debug, verbose=verbose)
    agent = create_agent(rl_config=rl_config, vec_env=vec_env, model_config=PPO_config,
                     path_to_pretrained=path_to_pretrained, agent_init=agent_init)
    metrics_list = ['train/loss', 'train/value_loss', 'train/clip_fraction', 'train/approx_kl', 'train/explained_variance', 'rollout/ep_rew_mean']
    # rl_config's own log_path, not a hardcoded one: '/data/logs/rl' is an
    # absolute path at the filesystem root, which is not writable anywhere
    # this runs. And three arguments, not four - ARCLogger takes
    # (folder, output_formats, metrics_list), and the fourth killed every
    # training run before the first step.
    logger = ARCLogger(rl_config['log_path'], ["stdout", "csv"], metrics_list)
    agent.set_logger(logger)
    callbacks = callback if extra_callback is None else CallbackList([callback, extra_callback])
    agent.learn(rl_config['total_steps'], callback=callbacks)
    acc, mean_len, grid_pred = evaluate_ARC_policy(agent, vec_env, n_eval_episodes=rl_config['n_eval_episodes'])
    if verbose:
        labels = ', '.join(subtask.label for subtask in subtasks)
        print(f'Accuracy for {labels}: {acc}. Mean episode length: {mean_len}')
    if plot_grid_pred:
        plot_grid(grid_pred)
    return acc, mean_len, agent, callback, vec_env

def train_on_subtask(subtask, rl_config:dict, PPO_config:dict=None, agent_init=None,
                     path_to_pretrained=None, verbose=False, plot_grid_pred=False, debug=False,
                     extra_callback=None):
    """One subtask, one env - train_on_subtasks with a single-element list."""
    return train_on_subtasks([subtask], rl_config=rl_config, PPO_config=PPO_config,
                             agent_init=agent_init, path_to_pretrained=path_to_pretrained,
                             verbose=verbose, plot_grid_pred=plot_grid_pred, debug=debug,
                             extra_callback=extra_callback)


def evaluate_on_subtask(agent, subtask, rl_config:dict):
    """Score a trained agent on a subtask it was not trained on.

    An env of its own, built the same way as the training ones and thrown
    away afterwards: evaluate_ARC_policy takes the vec env to step in as an
    argument, so nothing about the agent's own env changes.
    """
    vec_env = create_vec_env([subtask], n_envs=1, max_episode_len=rl_config['max_episode_len'],
                             repr_level=rl_config['repr_level'],
                             right_placement_reward=rl_config['right_placement_reward'],
                             action_penalty=rl_config['action_penalty'],
                             repetitive_actions_penalty=rl_config['repetitive_actions_penalty'],
                             seed=rl_config['seed'], font_color=rl_config['font_color'],
                             padding=rl_config['padding'], input_pattern=rl_config['input_pattern'],
                             milestones_rewards=rl_config['milestones_rewards'],
                             pad_val=rl_config['pad_val'], reward_approach=rl_config['reward_approach'],
                             feasible_actions=rl_config['feasible_actions'],
                             observation_space_elements=rl_config['observation_space_elements'],
                             observation_grid_shape=rl_config.get('observation_grid_shape'),
                             max_objects=rl_config.get('max_objects', MAX_OBJECTS))
    try:
        return evaluate_ARC_policy(agent, vec_env, n_eval_episodes=rl_config['n_eval_episodes'])
    finally:
        vec_env.close()


def distance_to_close(subtask, rl_config:dict):
    """How far this subtask's input starts from its output, in the units
    the accuracy is a fraction of.

    Zero means the input already matches the target, and closed_fraction
    scores such a subtask 1.0 by definition - "nothing to close" rather
    than "solved". That number cost this session a false positive: three of
    the five degenerate subtasks in the whole shape-preserving training
    split landed in one six-task sample, and their 1.0s read as the first
    thing PPO had ever learned. So the spans travel with the accuracies,
    and a caller can tell one kind of 1.0 from the other.
    """
    env = create_ARC_env(subtask, max_episode_len=rl_config['max_episode_len'],
                         repr_level=rl_config['repr_level'],
                         font_color=rl_config['font_color'],
                         padding=rl_config['padding'],
                         input_pattern=rl_config['input_pattern'],
                         pad_val=rl_config['pad_val'],
                         reward_approach=rl_config['reward_approach'],
                         feasible_actions=rl_config['feasible_actions'],
                         observation_space_elements=rl_config['observation_space_elements'],
                         observation_grid_shape=rl_config.get('observation_grid_shape'),
                         max_objects=rl_config.get('max_objects', MAX_OBJECTS))
    try:
        env.reset()
        return int(env.unwrapped.target_int - env.unwrapped.base_int)
    finally:
        env.close()


def check_one_grid_shape(task, observation_grid_shape=None):
    """Refuse a task whose examples are different sizes, and say why.

    Unless `observation_grid_shape` is set, in which case there is nothing
    to refuse: the observation is padded to that common shape for the
    buffer's sake and cropped back before the policy looks at it - see
    ARCGridWorld.observed_grid and rl.features.unpadded_grid_features.

    The observation carries the raw grid, and its Box is sized from the
    subtask the env was built with. So subtasks of different sizes cannot
    share a vec env (mixed) and cannot even be handed to the same agent one
    after another (sequential): stable-baselines3 checks the observation
    space on set_env and refuses. Half the shape-preserving training split
    is like this - 128 of 262 tasks - so this is the common case, not an
    edge one.

    Left as a refusal rather than turned on for everyone, because what
    raises otherwise is "could not broadcast input array from shape (10,10)
    into shape (6,6)" from inside DummyVecEnv, which names neither the task
    nor the reason - and because an observation shape is a setting, not
    something to pick on a caller's behalf.
    """
    if observation_grid_shape:
        return
    shapes = {subtask.train_inp_shape for subtask in task.subtasks}
    shapes.add(task.test_subtask.train_inp_shape)
    if len(shapes) > 1:
        raise ValueError(
            f"{task.label}: examples are {sorted(shapes)} - one agent cannot "
            f"span them while the observation carries a grid of one fixed "
            f"size. Set observation_grid_shape to pad the observation to a "
            f"common shape; the padding is cropped back off in the policy.")


def train_on_task(task, rl_config:dict, PPO_config:dict=None, agent_init=None, verbose=False,
                  plot_grid_pred=False, mode='mixed', extra_callback=None):
    """Train one agent on a whole task, then score it on the held-out pair.

    `mode='mixed'` puts every training subtask in one vec env and trains
    once: the rollout buffer holds steps from all of them, so an update is
    made against the task rather than against one example of it. A task's
    subtasks are the same rule shown several times - a policy fitted to one
    of them has nothing to generalise from.

    `mode='sequential'` is the original walk, one subtask after another,
    carrying the agent forward. Kept because it is a different experiment
    and not a strictly worse one, but note what it is: each subtask
    overwrites the last, which is catastrophic forgetting by construction.

    The score that answers "did it learn the rule" is on task.test_subtask,
    which no mode ever trains on. The per-subtask accuracies are the
    training ones and are reported alongside, not instead.

    total_steps is the budget for the task in both modes, so the sequential
    walk divides it among the subtasks rather than spending it on each. It
    used to spend it on each, which made the modes incomparable at the same
    setting - measured at 50_000, the sequential runs took three to four
    times the steps and the wall clock of the mixed ones, and the collapse
    to submitting on step one that showed up only there is what more
    training does, not what the mode does.
    """
    if mode not in ('mixed', 'sequential'):
        raise ValueError(f"mode={mode!r}: expected 'mixed' or 'sequential'")
    check_one_grid_shape(task, rl_config.get('observation_grid_shape'))
    seed = rl_config['seed']
    seed_everything(seed)
    train_metrics = {}
    accs_for_subtasks = {}
    lens_for_subtasks = {}
    expl_vars = {}
    subtasks = task.subtasks
    if mode == 'mixed':
        _acc, _len, agent, callback, _vec_env = train_on_subtasks(
            subtasks=subtasks, rl_config=rl_config, PPO_config=PPO_config,
            agent_init=agent_init, verbose=verbose, plot_grid_pred=plot_grid_pred,
            extra_callback=extra_callback)
        expl_vars['all'] = round(callback.explained_variances[-1], 3)
        for idx, subtask in enumerate(subtasks):
            acc, mean_len, _grid = evaluate_on_subtask(agent, subtask, rl_config)
            accs_for_subtasks[idx] = acc
            lens_for_subtasks[idx] = mean_len
    else:
        agent = agent_init
        share = dict(rl_config)
        share['total_steps'] = max(1, rl_config['total_steps'] // len(subtasks))
        for idx, subtask in enumerate(subtasks):
            acc, mean_len, agent, callback, _vec_env = train_on_subtask(
                subtask=subtask, rl_config=share, PPO_config=PPO_config,
                agent_init=agent, verbose=verbose, plot_grid_pred=plot_grid_pred,
                extra_callback=extra_callback)
            accs_for_subtasks[idx] = acc
            lens_for_subtasks[idx] = mean_len
            expl_vars[idx] = round(callback.explained_variances[-1], 3)
    test_acc, test_len, test_grid = evaluate_on_subtask(agent, task.test_subtask, rl_config)
    print(f'Explaines variances for subtasks: {expl_vars}')
    train_metrics['expl_vars'] = list(expl_vars.values())
    # Alongside the accuracies, not instead: a subtask whose input already
    # matches its output has nothing to close and is scored 1.0 whatever
    # the policy did.
    train_metrics['spans'] = {idx: distance_to_close(subtask, rl_config)
                              for idx, subtask in enumerate(subtasks)}
    degenerate = [idx for idx, span in train_metrics['spans'].items() if span <= 0]
    if degenerate:
        print(f'Subtasks with nothing to close, scored 1.0 by definition: '
              f'{degenerate}')
    train_metrics['test_acc'] = test_acc
    train_metrics['test_len'] = test_len
    train_metrics['test_grid'] = test_grid
    print(f'Accuracies for task: {list(accs_for_subtasks.values())}, Mean episode lengths for task: {list(lens_for_subtasks.values())}')
    print(f'Held-out accuracy for {task.test_subtask.label}: {test_acc:.3f}')
    return accs_for_subtasks, lens_for_subtasks, agent, train_metrics

def actions_exploration(subtask, rl_config: dict, n_rollouts: int = 500,
                        top_k: int = 5, mcts_iterations: int = 10):
    """Search `subtask`'s action space with MCTS and report which actions the
    best rollouts used.

    No policy involved. MCTS reads the environment directly - it snapshots
    env.grid/env.objects/env.max_int and explores through
    ARCGridWorld.simulate_action, which is cheap enough to try thousands of
    action sequences and drop the ones that go nowhere. That makes this
    usable before any training has happened, and independent of whether a
    policy can even be built.

    So a raw ARCGridWorld, not a vec env and not an agent: rollout_preparation
    calls reset()/step() expecting gymnasium's (obs, info) pair and reaches
    for attributes only ARCGridWorld has, neither of which survives
    DummyVecEnv or a wrapper stack.
    """
    env = ARCGridWorld(max_episode_len=rl_config['max_episode_len'],
                       right_placement_reward=rl_config['right_placement_reward'],
                       action_penalty=rl_config['action_penalty'],
                       repetitive_actions_penalty=rl_config['repetitive_actions_penalty'],
                       seed=42, font_color=rl_config['font_color'], padding=rl_config['padding'],
                       input_pattern=rl_config['input_pattern'],
                       milestones_rewards=rl_config['milestones_rewards'],
                       pad_val=rl_config['pad_val'], reward_approach=rl_config['reward_approach'],
                       repr_level=rl_config['repr_level'],
                       feasible_actions=rl_config['feasible_actions'],
                       observation_space_elements=rl_config['observation_space_elements'])
    env.set_subtask(subtask)
    best_rollouts = rollout_preparation(env, method="mcts", n_initial_rollouts=n_rollouts,
                                        top_k=top_k, mcts_iterations=mcts_iterations)
    if not best_rollouts:
        return []
    return extract_promising_actions(best_rollouts, rl_config['feasible_actions'])
