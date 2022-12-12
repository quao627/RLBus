from typing import Dict, Optional, Union, Generator, NamedTuple

import numpy as np
import torch as th
from gym import spaces

from stable_baselines3.common.type_aliases import DictRolloutBufferSamples
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.buffers import RolloutBuffer

from HybridPPO.policies import *


TensorDict = Dict[Union[str, int], th.Tensor]
class HybridRolloutBufferSamples(NamedTuple):
    observations: TensorDict
    actions: spaces.Tuple
    old_values: th.Tensor
    old_log_prob: spaces.Tuple
    advantages: th.Tensor
    returns: th.Tensor
    timesteps: th.Tensor # Ao's addition
    action_masks: th.Tensor # Ao's addition

class HybridRolloutBuffer(RolloutBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space_h: spaces.Space,
        action_space_l: spaces.Space,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super(RolloutBuffer, self).__init__(buffer_size, observation_space, action_space_h, device, n_envs=n_envs)
        self.action_space = action_space_h, action_space_l
        assert isinstance(self.action_space, tuple), "HybridRolloutBuffer must be used with Hybrid action space"
        
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.observations, self.actions, self.rewards, self.advantages = None, None, None, None
        self.returns, self.episode_starts, self.values, self.log_probs = None, None, None, None
        self.generator_ready = False
        self.action_dim_h = get_action_dim(self.action_space[0])
        self.action_dim_l = get_action_dim(self.action_space[1])
        self.action_dim = (self.action_dim_h, self.action_dim_l)
        self.reset()
    
    def reset(self) -> None:
        assert isinstance(self.action_space, tuple), "HybridRolloutBuffer must be used with Hybrid action space"

        self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, 3), dtype=np.float32) # not sure about this
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs, 2), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        # Ao's addition
        if isinstance(self.action_space[0], spaces.Discrete):
            mask_dims = self.action_space[0].n
        elif isinstance(self.action_space[0], spaces.MultiDiscrete):
            mask_dims = sum(self.action_space[0].nvec)
        elif isinstance(self.action_space[0], spaces.MultiBinary):
            mask_dims = 2 * self.action_space[0].n  # One mask per binary outcome
        else:
            raise ValueError(f"Unsupported action space {type(self.action_space[0])}")
        self.mask_dims = mask_dims
        self.timesteps = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.action_masks = np.ones((self.buffer_size, self.n_envs, mask_dims), dtype=np.float32)

        self.generator_ready = False

        super(RolloutBuffer, self).reset()

    def add(
        self,
        obs: np.ndarray,
        action: spaces.Tuple,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: spaces.Tuple,
        timestep: np.ndarray, # Ao's addition
        action_masks: Optional[np.ndarray] = None, # Ao's addition
    ) -> None:  # pytype: disable=signature-mismatch
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        log_prob_h, log_prob_l = log_prob
        # Reshape 0-d tensor to avoid error
        if len(log_prob_h.shape) == 0:
            log_prob_h = log_prob_h.reshape(-1, 1)
        if len(log_prob_l.shape) == 0:
            log_prob_l = log_prob_l.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)
        self.observations[self.pos] = np.array(obs).copy()

        action_h, action_l = action
        self.actions[self.pos,0,0] += action_h.item()
        self.actions[self.pos,0,1] += np.array(action_l)[0,0]
        self.actions[self.pos,0,2] += np.array(action_l)[0,1]
        
        # if action_h.item() == 2:
        #     action_l_real = 0
        # elif action_h.item() == 1:
        #     action_l_real = np.array(action_l)[0,1]
        # elif action_h.item() == 0:
        #     action_l_real = np.array(action_l)[0,0]
        # self.actions[self.pos,0,1] += action_l_real
        self.rewards[self.pos] = np.array(reward).copy()
        # Ao's addition
        self.timesteps[self.pos] = np.array(timestep).copy()
        if action_masks is not None:
            self.action_masks[self.pos] = action_masks.reshape((self.n_envs, self.mask_dims))
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos,0,0] = log_prob_h.clone().cpu().numpy()
        self.log_probs[self.pos,0,1] = log_prob_l.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: Optional[int] = None) -> Generator[HybridRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:

            # Ao's addition
            _tensor_names = ["actions", "values", "log_probs", "advantages", "returns", "timesteps", "action_masks"]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs
        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) ->HybridRolloutBufferSamples:

        return HybridRolloutBufferSamples(
            observations=self.to_torch(self.observations[batch_inds]),
            actions=self.to_torch(self.actions[batch_inds]),
            old_values=self.to_torch(self.values[batch_inds].flatten()),
            old_log_prob=self.to_torch(self.log_probs[batch_inds]),
            advantages=self.to_torch(self.advantages[batch_inds].flatten()),
            returns=self.to_torch(self.returns[batch_inds].flatten()),
            # Ao's addition
            timesteps=self.to_torch(self.timesteps[batch_inds].flatten()),
            action_masks=self.to_torch(self.action_masks[batch_inds].reshape(-1, self.mask_dims)),
        )
    
    # Ao's code for advantage and return calculation
    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        # Convert to numpy
        last_values = last_values.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            # 
            delta = self.rewards[step] + np.exp(-self.gamma * self.timesteps[step]) * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + np.exp(-self.gamma * self.timesteps[step]) * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values

