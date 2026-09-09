import torch.nn as nn
from rl.utils import linear_schedule
from rl.policy import ARCCustomActorCriticPolicy

rl_config = {
    'model_type': 'PPO',
    'total_steps': 1000000,
    'n_eval_episodes': 1,
    'n_envs': 1,
    'seed' : 42,
    'eval_freq': 5,
    'log_path': ".data/logs/rl/",
    'max_episode_len': 25,
    'right_placement_reward': 5.0,
    'action_penalty': 1.0,
    'repetitive_actions_penalty': 1.0,
    'font_color': 0.0,
    'padding': False,
    'input_pattern': 'start',
    'milestones_rewards': [1,2,3,4],
    # 3 pays 0 for a submit that solved nothing, and never less, while
    # acting pays -0.01 to -0.03 a step on the measured tasks: giving up on
    # the first step is the best return this reward offers, and PPO finds
    # it. Of six 30k runs, four ended submitting on 100% of steps,
    # transform-head entropy 0.007 to 0.023 out of ~4.0, and the share of
    # steps closing any distance fell from 17-20% to zero. ent_coef does not
    # hold against a reward that asks for this.
    #
    # 2 was tried for exactly that reason - it charges for a submit that
    # achieved nothing and pays partial credit for milestones reached - and
    # it does stop the collapse: on one action set, one process, one seed,
    # submits went to 0.0% of steps and entropy held at 2.6-2.8 against
    # 0.16. But by the outcome, the fraction of the distance to the target
    # the trained policy actually closes, it was no better on any of three
    # tasks and worse on two: -1.513 against -1.392, and -0.625 against
    # -0.250 with the held-out pair at -0.750 against 0.000. The agent stops
    # giving up and spends the horizon making the grid worse; 0 was better
    # than that. Not giving up is a proxy, and it went the other way from
    # the thing it stands for.
    #
    # So 3 stays, not because it is good - most closed fractions under it
    # are negative too - but because the one alternative measured against it
    # lost. Whatever replaces it should be judged on closed distance, and on
    # more than one seed: the same script on the same tasks moved 178fcbfb
    # from -0.187 to 0.000 between two runs, because search_task is bounded
    # by wall clock and handed the agent 39 actions once and 40 the next.
    'reward_approach': 3,
    'pad_val': 10,
    'feasible_actions': {0:'submit'},
    'repr_level': 1,
    'observation_space_elements': ["objects_emb"], # ["objects_emb", "relations_emb"]
    # None keeps the observation's grid sized to the subtask, which means one
    # agent can only span examples that are all the same size. A shape here
    # (ARC's largest is (30, 30)) pads the observation to it for the buffer's
    # sake and carries the true shape alongside, so the policy crops it back
    # off - see ARCGridWorld.observed_grid.
    'observation_grid_shape': None,
    # Object slots, and so the two object indices of every action. Sized by
    # this rather than by the task, so a slot past the objects a grid has is
    # a legal action that does nothing: on the median shape-preserving task
    # only 3.5% of the (object, object) pairs name two real objects.
    'max_objects': 16,
    }

def load_PPO_config():
    return {
    'verbose': 1,
    'batch_size': 256,
    'n_steps': 2048,
    'n_epochs' : 3,
    'gamma': 0.9,
    'gae_lambda': 0.9,
    'learning_rate': linear_schedule(0.0002),
    'clip_range': 0.2,
    'max_grad_norm': 0.5,
    'ent_coef': 0.01,
    'vf_coef': 0.5,
    'use_sde': False,
    'policy': ARCCustomActorCriticPolicy,
    'actor_arch': [256, 256, 256],
    'critic_arch': [256, 256, 256],
    'activation_fn': nn.ReLU,
    # The function, not a module built from it: a single module here is one
    # set of weights shared by every agent built in the process, and the
    # extractor would have to know to copy it. Naming what to build says
    # "this architecture" rather than "this network".
    'extr_arch': lin,
    'action_heads': 3,
    }

def lin(act_func=nn.ReLU()):
    return nn.Sequential(
              nn.Conv2d(in_channels=10, out_channels=8, kernel_size=3, stride=1, padding=1),
              nn.ReLU(),
              nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
              nn.ReLU(),
              nn.AdaptiveAvgPool2d((1, 1)),  # Output shape: [batch, 16, 1, 1]
              nn.Flatten()                   # Output shape: [batch, 16]
            )
lin_arch = lin()  # kept for notebooks that import it; configs name `lin`
