from itertools import zip_longest
from typing import Dict, List, Tuple, Type, Union

import gym
import torch as th
from torch import nn

from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.utils import get_device

import sys
sys.path.append('../')
sys.path.append('../Transformer/')

from Transformer.BusEncoder import BusEncoder

class TransformerExtractor(nn.Module):
    """
    Constructs an Transformer network from a ``net_arch`` description.
    It constructed over the HybridMlpExtractor.
    Adapted from Stable Baselines and pytorch nn.module.Transformer.
    """

    def __init__(
        self,
        feature_dim: int,
        net_arch: List[Union[int, Dict[str, List[int]]]],
        activation_fn: Type[nn.Module],
        device: Union[th.device, str] = "auto",
    ):
        super().__init__()
        device = get_device(device)

        layer_dim = 32

        self.latent_dim_pi = 32
        self.latent_dim_vf = 32
        self.shared_net = BusEncoder(feature_dim, layer_dim)
        self.transformer_policy_net = nn.Sequential(nn.Linear(32, 32), activation_fn())
        self.transformer_value_net = nn.Sequential(nn.Linear(32, 32), activation_fn())

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        shared_latent = self.shared_net(features)
        return self.transformer_policy_net(shared_latent), self.transformer_value_net(shared_latent)

    def forward_actor(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.transformer_policy_net(self.shared_net(features))

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.transformer_value_net(self.shared_net(features))