from typing import Any, Dict, Iterator, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
from lightning_rl.common.buffers import RolloutBuffer, RolloutSample
from lightning_rl.common.datasets import OnPolicyDataset
from lightning_rl.common.utils import clip_actions
from lightning_rl.models import RLModel
from torch.utils.data import DataLoader


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
               n_gradient_steps: int,
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
    self.n_gradient_steps = n_gradient_steps
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
    self.total_step_count = 0

  def collect_rollouts(self) -> Iterator[RolloutSample]:
    """
    Perform rollouts in the environment and return the results

    Yields
    ------
    Iterator[RolloutSample]
        Metrics from a randomized single time step in the current rollout (from multiple agents if the environment is vectorized). These metrics include:
        - observation tensors
        - actions tensors
        - state-values
        - log-probabilities over actions,
        - Advantages
        - Returns
    """
    assert self._last_obs is not None, "No previous observation was provided"
    with torch.no_grad():
      self.continue_training = True
      for _ in range(self.n_rollouts_per_epoch):
        self.eval()
        while not self.rollout_buffer.full():
          # Convert to pytorch tensor, let lightning take care of any GPU transfers
          obs_tensor = torch.as_tensor(self._last_obs).to(
              device=self.device, dtype=torch.float32)

          # Compute actions, values, and log-probs
          action_tensor, value_tensor = self.forward(obs_tensor)
          log_prob_tensor, _ = self.evaluate_actions(obs_tensor, action_tensor)
          actions = action_tensor.cpu().numpy()
          # Perform actions and update the environment
          new_obs, rewards, terminated, truncated, infos = self.env.step(actions)
          new_dones = terminated
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

          # Update the number of environment steps taken across ALL agents
          self.total_step_count += self.n_envs

        final_obs_tensor = torch.as_tensor(new_obs).to(
            device=self.device, dtype=torch.float32)
        final_value_tensor = self.forward(final_obs_tensor)[1]
        new_done_tensor = torch.as_tensor(new_dones).to(
            device=obs_tensor.device, dtype=torch.float32)
        samples = self.rollout_buffer.finalize(
            final_value_tensor,
            new_done_tensor
        )
        self.rollout_buffer.reset()

        # Train on minibatches from the current rollout. 
        self.train()
        for _ in range(self.n_gradient_steps):
          # Check if the training_step has requested to stop training on the current batch.
          if not self.continue_training:
            break
          indices = np.random.permutation(
              self.n_steps_per_rollout * self.n_envs)
          for idx in indices:
            yield RolloutSample(
                observations=samples.observations[idx],
                actions=samples.actions[idx],
                values=samples.values[idx],
                log_probs=samples.log_probs[idx],
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
              observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the action and value for the given observation

    Parameters
    ----------
    observation : torch.Tensor
        Observation Tensor

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        The action and value for the given observation
    """
    raise NotImplementedError

  def evaluate_actions(self,
                       observations: torch.Tensor,
                       actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the log probability and entropy for the given actions for the current state

    Parameters
    ----------
    observations : torch.Tensor
        Current environment state/observation
    actions : torch.Tensor
        Actions to evaluate

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Tuple containing the log probability and entropy for the given actions
    """
    raise NotImplementedError
