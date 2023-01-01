import time
from typing import Any, Dict, Iterator, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
from lightning_rl.common.buffers import RolloutBatch, RolloutBuffer
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
               n_gradient_steps: int = 1,
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

  def collect_rollouts(self) -> Iterator[RolloutBatch]:
    """
    Perform rollouts in the environment and return the results

    Yields
    ------
    Iterator[RolloutBatch]
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
          done_tensor = torch.as_tensor(
              self._last_dones).to(device=obs_tensor.device, dtype=torch.float32)
          # Compute actions, values, and log-probs
          action, value, log_prob, entropy = self.act(
              obs_tensor)
          # Perform actions and update the environment
          new_obs, rewards, terminated, truncated, infos = self.env.step(
              action.cpu().numpy())
          new_dones = terminated
          # Convert buffer entries to tensor
          reward_tensor = torch.as_tensor(rewards).to(
              device=obs_tensor.device, dtype=torch.float32)
          # Store the data in the rollout buffer
          self.rollout_buffer.add(
              obs=obs_tensor,
              action=action,
              reward=reward_tensor,
              done=done_tensor,
              value=value,
              log_prob=log_prob)
          self._last_obs = torch.as_tensor(new_obs, dtype=torch.float32).to(
              self.device)
          self._last_dones = torch.as_tensor(new_dones).to(self.device)

          # Update the number of environment steps taken across ALL agents
          self.total_step_count += self.n_envs

        # Use GAE to compute the advantage and return
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
        n_samples = self.n_steps_per_rollout * self.n_envs
        for epoch in range(self.n_gradient_steps):
          # Check if the training_step has requested to stop training on the current batch.
          if not self.continue_training:
            break
          indices = np.random.permutation(n_samples)
          for start in range(0, n_samples, self.batch_size):
            end = start + self.batch_size
            minibatch_inds = indices[start:end]
            yield RolloutBatch(
                observations=samples.observations[minibatch_inds],
                actions=samples.actions[minibatch_inds],
                values=samples.values[minibatch_inds],
                log_probs=samples.log_probs[minibatch_inds],
                advantages=samples.advantages[minibatch_inds],
                returns=samples.returns[minibatch_inds],
            )

  def train_dataloader(self):
    """
    Create the dataloader for our OffPolicyModel
    """
    self.dataset = OnPolicyDataset(
        rollout_generator=self.collect_rollouts,
    )
    return DataLoader(dataset=self.dataset, batch_size=None)

  def forward(self,
              observation: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """
    Compute the unprocessed output of the network. These outputs do not have to take a specific form. It is up to the act() function to process the output into the desired form.

    Parameters
    ----------
    observation : torch.Tensor
        Observation Tensor

    Returns
    -------
    Tuple[torch.Tensor, ...]
        The network output
    """
    raise NotImplementedError

  def act(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """
    Compute the processed output of the network.

    Parameters
    ----------
    x : torch.Tensor
        Input observation

    Returns
    -------
    Tuple[torch.Tensor, ...]
        The elements of the output tuple are:
        - action: The action selected by the network
        - value: The value of the input observation
        - log_prob: The log-probability of the selected action
        - entropy: The entropy of the action distribution
    """
    raise NotImplementedError

  def evaluate_actions(self,
                       observations: torch.Tensor,
                       actions: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """
    Compute the log-probability of the input action, along with the entropy of the action distribution and the value of the input observation.

    Parameters
    ----------
    observations : torch.Tensor
        Current environment state/observation
    actions : torch.Tensor
        Actions to evaluate

    Returns
    -------
    Tuple[torch.Tensor, ...]
        The elements of the output tuple are:
        - log_prob: The log-probability of the selected action
        - entropy: The entropy of the action distribution
        - value: The value of the input observation
    """
    raise NotImplementedError
