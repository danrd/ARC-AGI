import torch.nn as nn
from rl.utils import linear_schedule
from rl.policy import ARCCustomActorCriticPolicy

rl_config = {
    'model_type': 'PPO',
    # The budget, and the one setting that decides whether anything else is
    # measurable at all. What matters is not the step count but how many
    # times PPO updates on it: a rollout holds n_steps x n_envs, n_envs is
    # the subtask count in mixed mode, so a task with three examples at
    # n_steps=512 gets total_steps/1536 calls to train().
    #
    # Measured on dc433765, grid-only observation, two seeds, everything
    # else held:
    #
    #    30_000 steps (~8 updates)    closed fraction 0.000, held-out 0.000
    #   100_000 steps (~27 updates)   closed fraction 0.714, held-out 0.500
    #   300_000 steps (~84 updates)   closed fraction 0.714-0.786, held-out 0.500
    #
    # So below roughly 25 updates the policy has not begun to learn, and a
    # sweep run there compares configurations none of which are training -
    # which reads as "the setting does not matter" and is not that. Six grid
    # encoders from 16 to 256 features wide, across three seeds, produced
    # one number per task at 30_000 steps and a held-out of zero on all 108
    # runs; the same encoder at 100_000 solved a task the sweep had called
    # hopeless.
    #
    # The plateau is between 100k and 300k, which matches how this value was
    # chosen originally: more than 300k almost never bought anything, and
    # very small budgets never worked. 100_000 to 150_000 is enough for a
    # trial sweep; 300_000 for a result worth keeping. Economise on how many
    # configurations are compared, never on this.
    #
    # The value stayed at 1_000_000 under that paragraph until the curve was
    # measured - six tasks, three seeds, held-out accuracy every 25k steps of
    # one 600k run each:
    #
    #    25k  50k  75k  100k  150k  200k  300k  400k  500k  600k
    #   .005 .122 .089  .244  .303  .383  .322  .301  .302  .278   (mean)
    #
    # The mean peaks at 200k and falls after it, and the best checkpoint of a
    # run came at a median of 188k (p75 312k). The final policy is what a run
    # returns, and past the peak it wanders off: 4093f84a was at +0.444 on
    # the held-out pair at 100k and at +0.056 by 600k. So a million steps
    # cost five times the time of this and bought a worse policy. At the
    # measured 209 steps/s that is 16 minutes a task instead of 80.
    'total_steps': 200_000,
    'n_eval_episodes': 1,
    'n_envs': 1,
    'seed' : 42,
    # How many times MonitorCallback evaluates during one run, from which
    # the step interval is derived (rl.utils.calculate_eval_freq) - so it
    # scales with total_steps instead of being counted in raw callback
    # calls. It was eval_freq: 5, an evaluation episode every five steps of
    # training, which measured 2.4x the wall clock of the same run with
    # evaluation every 5000 (87 steps/s against 213) for nothing the
    # training uses: these evaluations only feed the plots.
    'evaluations': 20,
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
    #
    # All of the above was measured before the scale was fixed: a submit
    # was paid raw, up to 4.0, while the steps of a whole solve summed to
    # 0.2, and the solve's own submit reward was never paid because a
    # solve ends the episode before any submit. Under 3 that only means a
    # solve paid 0.2 where it now pays 1.0 (an incomplete submit pays 0
    # either way); under 2 the charge for giving up was -4.0 against step
    # rewards a hundredth of that, so the comparison above set 3 against
    # a reward dominated by its submit. See ARCGridWorld.paid_submit_reward
    # - worth measuring again rather than trusting.
    'reward_approach': 3,
    'pad_val': 10,
    # {0: 'submit'} and None below mean "decide per task":
    # rl.rl_job.narrowed_for_task fills them in from the task, and keeps
    # whatever is set to anything else - here or in a notebook - as set.
    # This one is the search's: action types that moved the grid, in every
    # output colour and every direction.
    'feasible_actions': {0:'submit'},
    'repr_level': 1,
    'observation_space_elements': ["objects_emb"], # ["objects_emb", "relations_emb"]
    # None keeps the observation's grid sized to the subtask, which means one
    # agent can only span examples that are all the same size. A shape here
    # (ARC's largest is (30, 30)) pads the observation to it for the buffer's
    # sake and carries the true shape alongside, so the policy crops it back
    # off - see ARCGridWorld.observed_grid.
    'observation_grid_shape': None,
    # What an action names: 'objects' or 'coordinates' (see
    # ARCGridWorld.addressing). None: narrowed_for_task reads it off the
    # agent's label; a run that is not narrowed gets objects.
    'addressing': None,
    # The grid size the coordinate half of the action space spans. None:
    # the task's largest grid.
    'coordinate_shape': None,
    # Object slots, and so the two object indices of every action - a slot
    # past the objects a grid has is a legal action that does nothing. None:
    # the slots the task's grids fill (a median 3 on the shape-preserving
    # tasks, where only 3.5% of the (object, object) pairs of a fixed 16
    # name two real objects); a run that is not narrowed gets
    # rl.arc_env.MAX_OBJECTS.
    'max_objects': None,
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
    # Observations the critic reads and the actor does not. The critic runs
    # only during training - it turns returns into advantages and nothing
    # calls it at inference - so it may read what will not exist at test
    # time, while the actor, which is all that runs on a held-out pair,
    # never sees it. ('target',) alongside 'target' in
    # observation_space_elements is the asymmetric case; empty means both
    # halves see the same observation. 'target' and 'delta_target' are
    # added whenever the observation carries them, whatever this says -
    # see rl.training.ANSWER_KEYS.
    'critic_only_keys': (),
    # Width of the per-object rows ARCCombinedExtractor carries for the
    # pointer heads, or 0 for the Linear object heads that preceded them -
    # which is the control the pointer head is measured against rather than
    # a setting anyone should want. See rl.policy.PointerHead.
    'pointer_dim': 32,
    # The same for coordinate addressing: the width of the per-row and
    # per-column embeddings the four coordinate heads score, one row of the
    # grid (or one column) each. Read only when rl_config's addressing is
    # 'coordinates', where action_heads is five whatever the line above
    # says - the action space decides that, not this file.
    'coordinate_dim': 32,
    # How the coordinate heads choose: 'autoregressive' draws the action,
    # then i1 knowing it, j1 knowing both, and so on (see
    # rl.policy.AutoregressiveCoordinateDistribution); 'independent' draws
    # all five from the state alone, which cannot want one box or another
    # without also wanting the box spanning both. Unmeasured - kept
    # switchable so the two can be compared.
    'coordinate_heads': 'autoregressive',
    # The object branch's architecture, the way extr_arch is the grid's:
    # keyword arguments for ObjectSetProcessor - dropout, self_attention,
    # grouped, cross_attention, use_position. None is what this shipped as.
    'object_arch': None,
    # How 'relations_emb' is read, when observation_space_elements asks for
    # it at all. 'messages' is one round of message passing with weights
    # shared across pairs, merged into the object rows so that it reaches
    # the pointer head - 22.7k parameters however many slots there are.
    # 'flat' is the Flatten-and-two-Linears branch this shipped with, kept
    # so the two can be compared.
    #
    # Measured over 93 runs on six tasks and up to seven seeds, paired
    # against an observation of grid and objects alone:
    #
    #   flat       n=12  median difference -0.006, better on 6      - and it
    #              only ran on four of the six tasks. Its parameter count is
    #              quartic in the slot count: 132.9M at MAX_OBJECTS=16,
    #              702M at 24 slots, 1.14bn at 27, where the optimiser
    #              states alone are 12.7 GB and the run is OOM-killed.
    #              Three to five times slower for a difference of zero.
    #   messages   n=18  median +0.017, better on 9  - on its own, nothing
    #   with the deltas alongside it, n=32, median +0.131, better on 19,
    #   and better on four of the six tasks, worse on one, tied on one.
    #
    # So relations pay next to the deltas rather than by themselves, and
    # only in this form. The effect is smaller than the 0.292 spread
    # between seeds, so it shows up as a consistent sign rather than in any
    # single run.
    'relation_mode': 'messages',
    # The message-passing architecture, the way object_arch is the object
    # branch's: keyword arguments for RelationMessages - endpoints,
    # aggregation, rounds, hidden. None is what was measured.
    'relation_arch': None,
    }

def lin(act_func=nn.ReLU()):
    """The grid encoder: two convolutions and a pool to a 3x3 grid.

    It used to pool to (1, 1), which returns one mean per channel over the
    whole grid - how much of each of 16 textures the grid holds on average,
    and nothing at all about where. Measured with the grid as the only
    observation, at 150_000 steps, six encoders on the three tasks of six
    that respond at that budget, three seeds each:

        arm     pool   width   held-out, averaged over 6 runs
        pool1   1x1       16   +0.000        <- what this was
        deep1   1x1       16   +0.060        <- four convolutions, RF 9x9
        pool2   2x2       64   +0.333
        pool3   3x3      144   +0.417
        deep3   3x3      144   +0.250
        pool4   4x4      256   +0.262

    The robust part is the first row: a global mean scored zero on the
    held-out pair in all six of its runs, where every encoder that keeps
    some notion of where scored above zero somewhere. On dc433765 that is
    0.000 on three seeds against 0.500 on three seeds for anything pooled
    2x2 or finer. Depth without pooling does not substitute: deep1 sees
    9x9 instead of 5x5 and still averages it away, and scores 0.060.

    Which k is best is not resolved - 2, 3 and 4 differ by less than the
    spread across seeds on six runs, and on 253bf280 the 4x4 arm ranges
    from +0.5 to -0.429 across its three seeds. 3x3 has the best held-out
    average of the three and sits in the middle on width, so it is the one
    taken; the finding being acted on is "not a global mean", not "3x3
    exactly".

    `act_func` is accepted and unused, as it was before.
    """
    return nn.Sequential(
              nn.Conv2d(in_channels=10, out_channels=8, kernel_size=3, stride=1, padding=1),
              nn.ReLU(),
              nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
              nn.ReLU(),
              nn.AdaptiveAvgPool2d((3, 3)),  # Output shape: [batch, 16, 3, 3]
              nn.Flatten()                   # Output shape: [batch, 144]
            )
lin_arch = lin()  # kept for notebooks that import it; configs name `lin`
