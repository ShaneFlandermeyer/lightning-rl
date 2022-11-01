from typing import Any, Dict, Optional, Tuple, Union

import gym
import numpy as np
import torch
from lightning_rl.models import RLModel
from lightning_rl.common.buffers import RolloutBuffer, RolloutBatch
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
               env: Union[gym.Env, gym.vector.VectorEnv, str],
               n_steps_per_rollout: int,
               n_rollouts_per_epoch: int,
               gamma: float = 0.99,
               gae_lambda: float = 1.0,
               seed: Optional[int] = None,
               **kwargs) -> None:

    super().__init__(
        env=env,
        support_multi_env=True,
        seed=seed,
        **kwargs,
    )

    self.n_steps_per_rollout = n_steps_per_rollout
    self.n_rollouts_per_epoch = n_rollouts_per_epoch
    self.batch_size = self.n_steps_per_rollout * self.n_envs
    self.gamma = gamma
    self.gae_lambda = gae_lambda

    self.rollout_buffer = RolloutBuffer(
        n_rollout_steps=n_steps_per_rollout,
        gamma=gamma,
        gae_lambda=gae_lambda,
        n_envs=self.n_envs
    )

    # Metrics
    self.total_step_count = 0

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

  @torch.no_grad()
  def collect_rollouts(self) -> RolloutBatch:
    """
    Collect rollouts and put them into the RolloutBuffer
    """

    assert self._last_obs is not None, "No previous observation was provided"

    self.eval()
    self.rollout_buffer.reset()
    while not self.rollout_buffer.full():
      # Convert to pytorch tensor, let lightning take care of any GPU transfers
      obs_tensor = torch.as_tensor(self._last_obs).to(
          device=self.device, dtype=torch.float32)
      # Compute actions and log-probabilities
      dist, values = self.forward(obs_tensor)
      actions = dist.sample()
      log_probs = dist.log_prob(actions)
      # Perform actions and update the environment
      new_obs, rewards, dones, infos = self.env.step(actions.cpu().numpy())
      if isinstance(self.action_space, gym.spaces.Discrete):
        # Reshape in case of discrete actions
        actions = actions.view(-1, 1)
      rewards = torch.as_tensor(rewards).to(
          device=obs_tensor.device, dtype=torch.float32)
      # Store the data in the rollout buffer
      self.rollout_buffer.add(
          obs_tensor,
          actions,
          rewards,
          self._last_dones,
          values,
          log_probs)
      self._last_obs = new_obs
      self._last_dones = torch.as_tensor(
          dones).to(device=obs_tensor.device, dtype=torch.float32)
      self.total_step_count += 1

    final_obs = torch.as_tensor(new_obs).to(
        device=self.device, dtype=torch.float32)
    dist, final_values = self.forward(final_obs)
    samples = self.rollout_buffer.finalize(final_values, self._last_dones)

    self.train()
    return samples


class OnPolicyDataLoader():
  def __init__(self, model: OnPolicyModel):
    self.model = model

  def __iter__(self):
    for _ in range(self.model.n_rollouts_per_epoch):
      experience_batch = self.model.collect_rollouts()
      yield experience_batch
