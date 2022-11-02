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
    # TODO: Hard-coded values incoming
    self.avg_reward_len = 25
    min_episode_reward = -21
    self.total_step_count = 0
    self.episode_reward = 0
    self.episode_count = 0
    self.episode_step_count = 0

    # For the reward computations, assuming the first environment is representative of the agent's performance
    self.total_rewards = min_episode_reward*np.ones((self.avg_reward_len, ))
    self.avg_rewards = np.mean(self.total_rewards[-self.avg_reward_len:])

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

  def collect_rollouts(self) -> RolloutBatch:
    """
    Collect rollouts and put them into the RolloutBuffer
    """

    assert self._last_obs is not None, "No previous observation was provided"

    self.eval()
    self.rollout_buffer.reset()
    while not self.rollout_buffer.full():
      # Convert to pytorch tensor, let lightning take care of any GPU transfers
      with torch.no_grad():
        obs_tensor = torch.as_tensor(self._last_obs).to(
            device=self.device, dtype=torch.float32)
        # Compute actions and log-probabilities
        action_dist, value_tensor = self.forward(obs_tensor)
        action_tensor = action_dist.sample()
        log_prob_tensor = action_dist.log_prob(action_tensor)
      actions = action_tensor.cpu().numpy()

      # Perform actions and update the environment
      new_obs, rewards, new_dones, infos = self.env.step(actions)

      # Convert buffer entries to tensor
      if isinstance(self.action_space, gym.spaces.Discrete):
        # Reshape in case of discrete actions
        action_tensor = action_tensor.view(-1, 1)
      reward_tensor = torch.as_tensor(rewards).to(
          device=obs_tensor.device, dtype=torch.float32)
      done_tensor = torch.as_tensor(
          self._last_dones).to(device=obs_tensor.device, dtype=torch.float32)

      # Store the data in the rollout buffer
      self.rollout_buffer.add(
          obs_tensor,
          action_tensor,
          reward_tensor,
          done_tensor,
          value_tensor,
          log_prob_tensor)
      self._last_obs = new_obs
      self._last_dones = new_dones

      # Update metrics
      self.total_step_count += 1
      if new_dones[0]:
        self.episode_count += 1
        self.total_rewards = np.append(self.total_rewards, self.episode_reward)
        self.avg_rewards = np.mean(self.total_rewards[-self.avg_reward_len:])
        print()
        print("Episode #", self.episode_count)
        print("Episode reward:", self.episode_reward)
        print("Average reward:", self.avg_rewards)
        print("Num. Steps:", self.episode_step_count)
        self.episode_reward = 0
        self.episode_step_count = 0
      else:
        self.episode_reward += rewards[0]
        self.episode_step_count += 4

    with torch.no_grad():
      final_obs_tensor = torch.as_tensor(new_obs).to(
          device=self.device, dtype=torch.float32)
      action_dist, final_value_tensor = self.forward(final_obs_tensor)

    new_done_tensor = torch.as_tensor(new_dones).to(
        device=obs_tensor.device, dtype=torch.float32)
    samples = self.rollout_buffer.finalize(
        final_value_tensor,
        new_done_tensor
    )

    self.train()
    return samples


class OnPolicyDataLoader():
  def __init__(self, model: OnPolicyModel):
    self.model = model

  def __iter__(self):
    # TODO: Add a batch size parameter
    for _ in range(self.model.n_rollouts_per_epoch):
      experience_batch = self.model.collect_rollouts()
      observations, actions, values, log_probs, advantages, returns = experience_batch
      batch_size = self.model.n_steps_per_rollout * self.model.n_envs
      perm = torch.randperm(batch_size)

      yield RolloutBatch(
          observations=observations,
          actions=actions,
          values=values,
          log_probs=log_probs,
          advantages=advantages,
          returns=returns,
      )
