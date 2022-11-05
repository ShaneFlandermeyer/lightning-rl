from typing import Any, Dict, Iterator, Optional, Tuple, Union

import gym
import numpy as np
import torch
from torch.utils.data import DataLoader
from lightning_rl.models import RLModel
from lightning_rl.common.buffers import RolloutBuffer, RolloutBatch, RolloutBufferSamples, RolloutExperience
from lightning_rl.common.utils import clip_actions
from lightning_rl.common.datasets import OnPolicyDataset


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
               batch_size: Optional[int] = None,
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
    if batch_size is None:
      self.batch_size = self.n_steps_per_rollout * self.n_envs
    else:
      self.batch_size = batch_size
    self.gamma = gamma
    self.gae_lambda = gae_lambda

    self.rollout_buffer = RolloutBuffer(
        n_steps_per_rollout,
        gamma=self.gamma,
        gae_lambda=self.gae_lambda,
        n_envs=self.n_envs,
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

  # @torch.no_grad()
  def collect_rollouts(self) -> Iterator[RolloutBufferSamples]:
    """
    Perform rollouts in the environment and return the results

    Yields
    ------
    Iterator[RolloutBufferSamples]
        Metrics from a randomized single time step in the current rollout (from multiple agents if the environment is vectorized). These metrics include:
        - observation tensors
        - actions tensors
        - state-values
        - log-probabilities over actions,
        - Advantages
        - Returns
    """
    assert self._last_obs is not None, "No previous observation was provided"
    for _ in range(self.n_rollouts_per_epoch):
      with torch.no_grad():
        self.eval()
        while not self.rollout_buffer.full():
          # Convert to pytorch tensor, let lightning take care of any GPU transfers
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
          # TODO: Use a gym wrapper for this
          self.total_step_count += 1
          if new_dones[0]:
            self.episode_count += 1
            self.total_rewards = np.append(
                self.total_rewards, self.episode_reward)
            self.avg_rewards = np.mean(
                self.total_rewards[-self.avg_reward_len:])
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

        final_obs_tensor = torch.as_tensor(new_obs).to(
            device=self.device, dtype=torch.float32)
        _, final_value_tensor = self.forward(final_obs_tensor)
        new_done_tensor = torch.as_tensor(new_dones).to(
            device=obs_tensor.device, dtype=torch.float32)
        samples = self.rollout_buffer.finalize(
            final_value_tensor,
            new_done_tensor
        )
        self.rollout_buffer.reset()

        # Return the samples from this rollout in a random order. 
        self.train()
        indices = np.random.permutation(self.n_steps_per_rollout * self.n_envs)
        for idx in indices:
          yield RolloutBufferSamples(
              observations=samples.observations[idx],
              actions=samples.actions[idx],
              old_values=samples.values[idx],
              old_log_probs=samples.log_probs[idx],
              advantages=samples.advantages[idx],
              returns=samples.returns[idx])

    

  def train_dataloader(self):
    """
    Create the dataloader for our OffPolicyModel
    """
    self.dataset = OnPolicyDataset(
        rollout_generator=self.collect_rollouts,
    )
    return DataLoader(dataset=self.dataset, batch_size=self.batch_size)

  def forward(self,
              obs: Union[Tuple, Dict[str, Any], np.ndarray, int]
              ) -> Tuple[torch.distributions.Distribution, torch.Tensor]:
    """
    Override this function with the forward pass through your model

    Parameters
    ----------
    obs : Union[Tuple, Dict[str, Any], np.ndarray, int]
        Input observations

    Returns
    -------
    Tuple[torch.distributions.Distribution, torch.Tensor]
        Action probability distribution from the policy network and the state-value from the value network

    Raises
    ------
    NotImplementedError
    """

    raise NotImplementedError
