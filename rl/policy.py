import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Callable, Dict, Optional, Tuple

class ARCCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, pos_enc_dim=16, cnn_arch=None):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim=1)
        extractors = {}
        total_concat_size = 0
        self.pos_enc_dim = pos_enc_dim

        if cnn_arch:
            self.cnn =  cnn_arch   
        else:
            self.n_features = [16, 16]
            self.kernel_sizes = [3, 3]
            # zero_dim = observation_space['grid'].shape[0]
            self.cnn = nn.Sequential(
                nn.Conv2d(1, self.n_features[0], kernel_size=self.kernel_sizes[0], stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.n_features[0], self.n_features[1], kernel_size=self.kernel_sizes[1], stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(),
            )
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "agent_position":
                # Run through a simple MLP
                extractors[key] = nn.Sequential(nn.Flatten(), nn.Linear(subspace.shape[0]*subspace.shape[1], self.pos_enc_dim))
                total_concat_size += self.pos_enc_dim
            else:
                # We will just downsample one channel of the image by 4x4 and flatten.
                # Assume the image is single-channel (subspace.shape[0] == 0)  
                extractors[key] = self.cnn
                            # Compute shape by doing one forward pass
                with torch.no_grad():
                    n_flatten = extractors[key](torch.as_tensor(observation_space.spaces[key].sample()[None]).float())
                total_concat_size += n_flatten.shape[-2] * n_flatten.shape[-1]
    
        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size

    def forward(self, observation) -> torch.Tensor:
        encoded_tensor_list = []
        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            if key == "agent_position":
                res = extractor(observation[key])
            else:
                res = extractor(observation[key].unsqueeze(1))
            encoded_tensor_list.append(res)
        return torch.cat(encoded_tensor_list, dim=1)

class ARCCustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
        use_sde:bool = False
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_pi), nn.ReLU()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf), nn.ReLU()
        )

    def forward(self, features: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)


class ARCCustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        # use_sde: bool,
        features_extractor_class=ARCCombinedExtractor,
        features_extractor_kwargs: Optional[Dict] = None,
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = True
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # use_sde,
            features_extractor_class=ARCCombinedExtractor,
            features_extractor_kwargs=features_extractor_kwargs,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    def _build_extractor(self) -> None:
        self.extractor = ARCCombinedExtractor(self.features_dim, **self.features_extractor_kwargs)