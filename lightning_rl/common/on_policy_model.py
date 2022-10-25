from typing import Any, Dict, Optional, Tuple, Union

import gym
import numpy as np
import torch
from lightning_rl.common import RLModel
from stable_baselines3.common.vec_env import VecEnv

from lightning_rl.common.buffers import RolloutBuffer, RolloutSamples
from lightning_rl.common.off_policy_model import OffPolicyDataLoader
from lightning_rl.common.utils import clip_actions


class OnPolicyModel(RLModel):
  """
  Base class for on-policy algorithms

  Parameters
  ----------
  RLModel : _type_
      _description_
  """

  def __init__(self,
               env: Union[gym.Env, VecEnv, str],
               n_steps_per_rollout: int,
               n_rollouts_per_epoch: int,
               gamma: float = 0.99,
               gae_lambda: float = 1.0,
               seed: Optional[int] = None) -> None:
    super().__init__(
        env=env,
        support_multi_env=True,
        seed=seed,
    )

    self.n_steps_per_rollout = n_steps_per_rollout
    self.n_rollouts_per_epoch = n_rollouts_per_epoch
    self.batch_size = self.n_steps_per_rollout * self.n_envs
    self.gamma = gamma
    self.gae_lambda = gae_lambda

    self.rollout_buffer = RolloutBuffer(
        buffer_size=n_steps_per_rollout,
        observation_space=self.observation_space,
        action_space=self.action_space,
        gamma=gamma,
        gae_lambda=gae_lambda,
        n_envs=self.n_envs
    )

  def forward(self,
              obs: Union[Tuple, Dict[str, Any], np.ndarray, int]) -> Tuple[torch.distributions.Distribution, torch.Tensor]:
    """
    Override this function with the forward function of your model

    :param obs: The input observations
    :return: The chosen actions
    """
    raise NotImplementedError

  def train_dataloader(self):
    """
    Create the dataloader for our OffPolicyModel
    """
    return OnPolicyDataLoader(self)

  def collect_rollouts(self) -> RolloutSamples:
    """
    Collect rollouts and put them into the RolloutBuffer
    """

    assert self._last_obs is not None, "No previous observation was provided"
    with torch.no_grad():
      self.eval()
      for step in range(self.n_steps_per_rollout):

        # Convert to pytorch tensor, let lightning take care of any GPU transfers
        obs_tensor = torch.as_tensor(self._last_obs).to(
            device=self.device, dtype=torch.float32)
        dist, values = self(obs_tensor)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        clipped_actions = clip_actions(actions, self.action_space)

        new_obs, rewards, dones, infos = self.env.step(clipped_actions)

        if isinstance(self.action_space, gym.spaces.Discrete):
          # Reshape in case of discrete actions
          actions = actions.view(-1, 1)

        if not torch.is_tensor(self._last_dones):
          self._last_dones = torch.as_tensor(
              self._last_dones).to(device=obs_tensor.device)
        rewards = torch.as_tensor(rewards).to(device=obs_tensor.device)
        self.rollout_buffer.add(
            obs_tensor, actions, rewards, self._last_dones, values, log_probs)
        self._last_obs = new_obs
        self._last_dones = dones

      final_obs = torch.as_tensor(new_obs).to(
          device=self.device, dtype=torch.float32)
      dist, final_values = self(final_obs)
      samples = self.rollout_buffer.finalize(
          final_values,
          torch.as_tensor(dones).to(device=obs_tensor.device, dtype=torch.float32))
      self.rollout_buffer.reset()

    self.train()
    return samples


class OnPolicyDataLoader():
  def __init__(self, model: OnPolicyModel):
    self.model = model

  def __iter__(self):
    for _ in range(self.model.n_rollouts_per_epoch):
      experiences = self.model.collect_rollouts()
      observations, actions, old_values, old_log_probs, advantages, returns = experiences
      yield RolloutSamples(
          observations=observations,
          actions=actions,
          old_values=old_values,
          old_log_probs=old_log_probs,
          advantages=advantages,
          returns=returns,
      )
