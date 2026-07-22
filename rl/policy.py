import torch
import torch.nn as nn
from typing import Callable, Dict, List, Optional, Tuple
from stable_baselines3.common.distributions import MultiCategoricalDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from gymnasium import spaces
from rl.features import ARCCombinedExtractor, ARCGNNExtractor, ARCSeparateExtractor

class ARCCustomNetwork(nn.Module):
    """Custom network for policy and value function.

    Receives as input the features extracted by the features extractor.

    Args:
        feature_dim: Dimension of the features extracted by the features
            extractor (e.g. features from a CNN).
        action_dims: List of dimensions for each action space.
        use_sde: Whether to use state dependent exploration.
        net_arch: Network architecture for policy and value networks.
        action_heads: Number of action distribution heads (1, 2, 3, or 5).
    """

    def __init__(
        self,
        feature_dim: int,
        action_dims: list,
        use_sde: bool = False,
        net_arch: dict = {'pi': [64], 'vf': [64]},
        action_heads: int = 1,
    ):
        super().__init__()
        self.action_dims = action_dims
        self.action_heads = action_heads
        self.n_action_dims = len(action_dims)

        # Get network architecture
        policy = net_arch['pi']
        self.latent_dim_pi = policy[-1]
        value = net_arch['vf']
        self.latent_dim_vf = value[-1]

        # Shared network
        shared_net = [nn.Linear(feature_dim, policy[0]), nn.ReLU()]
        for i in range(len(policy)-1):
            shared_net.append(nn.Linear(policy[i], policy[i+1]))
            shared_net.append(nn.ReLU())

        # Value network
        value_net = [nn.Linear(feature_dim, value[0]), nn.ReLU()]
        for i in range(len(value)-1):
            value_net.append(nn.Linear(value[i], value[i+1]))
            value_net.append(nn.ReLU())

        # Policy network
        self.shared_net = nn.Sequential(*shared_net[:-1])  # Remove the last ReLU

        # Value network
        self.value_net = nn.Sequential(*value_net)

        # Create policy networks based on action_heads
        self.policy_nets = nn.ModuleList()

        if action_heads == 1:
            # One head for all 5 dimensions combined
            self.policy_nets.append(nn.Linear(self.latent_dim_pi, sum(action_dims)))
        elif action_heads == 2:
            # First head for action type (first dimension)
            self.policy_nets.append(nn.Linear(self.latent_dim_pi, action_dims[0]))
            # Second head for the rest dimensions
            self.policy_nets.append(nn.Linear(self.latent_dim_pi, sum(action_dims[1:])))
        elif action_heads == 3:
            # First head for action type
            self.policy_nets.append(nn.Linear(self.latent_dim_pi, action_dims[0]))
            # Second head for first two coordinates
            self.policy_nets.append(nn.Linear(self.latent_dim_pi, action_dims[1] + action_dims[2]))
            # Third head for second two coordinates
            self.policy_nets.append(nn.Linear(self.latent_dim_pi, action_dims[3] + action_dims[4]))
        elif action_heads == 5:
            # Separate head for each dimension
            for dim in action_dims:
                self.policy_nets.append(nn.Linear(self.latent_dim_pi, dim))
        else:
            raise ValueError(f"Unsupported number of action heads: {action_heads}")

    def forward(self, features: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Returns:
            List of latent_policy outputs (one for each action head), and
            latent_value.
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> List[torch.Tensor]:
        shared_features = self.shared_net(features)
        return [policy_net(shared_features) for policy_net in self.policy_nets]

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)

class ARCCustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        features_extractor_class=ARCCombinedExtractor,
        features_extractor_kwargs: Optional[Dict] = None,
        action_heads: int = 1,
        *args,
        **kwargs,
    ):
        # Save action_heads before passing to parent class
        self.action_heads = action_heads

        # Disable orthogonal initialization if needed
        kwargs["ortho_init"] = kwargs.get("ortho_init", True)

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        action_dims = self.action_space.nvec.tolist()
        self.mlp_extractor = ARCCustomNetwork(
            self.features_dim,
            action_dims=action_dims,
            net_arch=self.net_arch,
            action_heads=self.action_heads
        )

    def _get_action_dist_from_latent(self, latent_pi: List[torch.Tensor]):
        """Create action distributions based on the number of action heads."""
        action_dims = self.action_space.nvec.tolist()
        logits = torch.hstack(latent_pi)
        distribution = MultiCategoricalDistribution(action_dims)
        distribution.proba_distribution(logits)
        return distribution

    def extract_features(self, obs) -> torch.Tensor:
        """
        Preprocess the observation if needed and extract features.
        """
        # preprocessed_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        return self.features_extractor(obs)

    def forward(self, obs, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through both the actor and critic networks.

        Args:
            obs: Observation.
            deterministic: Whether to sample or use deterministic actions.

        Returns:
            Action, value, and log probability of the action.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)

        # Get latent representations
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # print(f'Value net: {self.value_net}')
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)

        # Get actions and log probabilities
        distribution = self._get_action_dist_from_latent(latent_pi)

        if deterministic:
            actions = distribution.mode()
        else:
            actions = distribution.sample()

        log_prob = distribution.log_prob(actions)
        # Reshape actions to match action space
        actions = actions.reshape((-1, len(self.action_space.nvec)))
        # print(f'Values at the end: {values}')
        return actions, values, log_prob

class ARCGNNPolicy(ARCCustomActorCriticPolicy):
    """Policy using GNN approach"""
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        features_extractor_kwargs = kwargs.pop('features_extractor_kwargs', {})
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            features_extractor_class=ARCGNNExtractor,
            features_extractor_kwargs=kwargs.get('features_extractor_kwargs', {}),
            **kwargs
        )

class ARCSeparatePolicy(ARCCustomActorCriticPolicy):
    """Policy using enhanced separate processing approach"""
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        features_extractor_kwargs = kwargs.pop('features_extractor_kwargs', {})
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            features_extractor_class=ARCSeparateExtractor,
            features_extractor_kwargs=kwargs.get('features_extractor_kwargs', {}),
            **kwargs
        )
