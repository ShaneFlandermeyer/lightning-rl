from collections import deque
from typing import NamedTuple, Tuple
from gym import spaces
import numpy as np

import torch

from lightning_rl.common.utils import get_action_dim, get_obs_shape


# Off policy experiences
class Experience(NamedTuple):
  state: np.ndarray
  action: np.ndarray
  reward: float
  next_state: np.ndarray
  done: bool


class ExperienceBatch(NamedTuple):
  states: torch.Tensor
  actions: torch.Tensor
  rewards: torch.Tensor
  next_states: torch.Tensor
  dones: torch.Tensor


class RolloutExperience(NamedTuple):
  """
  A single policy gradient rollout experience

  NOTE: This is a single experience FROM EACH ENVIRONMENT. Therefore, each of the values must be tensors rather than scalars
  """
  observation: torch.Tensor
  action: torch.Tensor
  reward: torch.Tensor
  done: torch.Tensor
  value: torch.Tensor
  log_prob: torch.Tensor


class RolloutBatch(NamedTuple):
  observations: torch.Tensor
  actions: torch.Tensor
  values: torch.Tensor
  log_probs: torch.Tensor
  advantages: torch.Tensor
  returns: torch.Tensor


class RolloutBufferSamples(NamedTuple):
  observations: torch.Tensor
  actions: torch.Tensor
  old_values: torch.Tensor
  old_log_probs: torch.Tensor
  advantages: torch.Tensor
  returns: torch.Tensor


class ReplayBuffer():
  """
  Basic buffer for storing an experience from a single time step
  """

  def __init__(
      self,
      capacity: int,
      n_envs: int = 1,
  ) -> None:
    assert n_envs == 1, "Replay buffer currently only supports single environments"

    self.capacity = capacity
    self.n_envs = 1
    self.buffer = deque(maxlen=capacity)

  def __len__(self) -> int:
    return len(self.buffer)

  def append(self,
             state: np.ndarray,
             action: np.ndarray,
             reward: np.ndarray,
             next_state: np.ndarray,
             done: np.ndarray) -> None:
    """
    Add an experience to the buffer
    """
    self.buffer.append(Experience(state, action, reward, next_state, done))

  def sample(self, batch_size: int) -> ExperienceBatch:
    """
    Return a mini-batch of experiences

    Parameters
    ----------
    batch_size : int
        Number of experiences in the batch

    Returns
    -------
    ExperienceBatch
        A named tuple containing torch tensors for each of the experience components
    """
    indices = np.random.choice(
        len(self.buffer), size=min(batch_size, len(self.buffer)), replace=False)
    states, actions, rewards, next_states, dones = zip(
        *(self.buffer[idx] for idx in indices))

    return ExperienceBatch(
        states=torch.as_tensor(np.array(states)[:, 0, :]),
        actions=torch.as_tensor(np.array(actions)[:, 0, :]),
        rewards=torch.as_tensor(np.array(rewards)),
        next_states=torch.as_tensor(np.array(next_states)[:, 0, :]),
        dones=torch.as_tensor(np.array(dones))
    )


class RolloutBuffer():
  """
  Buffer used for on-policy algorithms where an n-step rollout is used to estimate the value/advantage of each state

  Parameters
  ----------

  n_rollout_steps: (int) 
    Max number of element in the buffer
  device: (torch.device)
    Device containing all experience tensors
  gae_lambda: (float) 
    Factor for trade-off of bias vs variance for Generalized Advantage Estimator. Equivalent to classic advantage when set to 1.
  gamma: (float) 
    Discount factor
  n_envs: (int) 
    Number of parallel environments
  """

  def __init__(
      self,
      n_rollout_steps: int,
      gamma: float = 0.99,
      gae_lambda: float = 1.0,
      n_envs: int = 1,
  ) -> None:
    self.n_rollout_steps = n_rollout_steps
    self.gae_lambda = gae_lambda
    self.gamma = gamma
    self.n_envs = n_envs
    self.buffer = deque(maxlen=n_rollout_steps)

  def __len__(self) -> int:
    return len(self.buffer)

  def full(self) -> bool:
    return len(self.buffer) == self.n_rollout_steps

  def reset(self):
    self.buffer.clear()

  def add(
      self,
      obs: torch.Tensor,
      action: torch.Tensor,
      reward: torch.Tensor,
      done: torch.Tensor,
      value: torch.Tensor,
      log_prob: torch.Tensor
  ) -> None:
    """
    Add a new experience to the buffer.

    Parameters
    ----------
    obs: (torch.tensor) 
      Observation tensor
    action: (torch.tensor) 
      Action tensor
    reward: (torch.tensor)
    done: (torch.tensor) 
      End of episode signal.
    value: (torch.Tensor) 
      estimated value of the current state following the current policy.
    log_prob: (torch.Tensor) 
      log probability of the action following the current policy.
    """
    experience = RolloutExperience(
        observation=obs,
        action=action,
        reward=reward,
        done=done,
        value=value,
        log_prob=log_prob,
    )
    self.buffer.append(experience)

  def finalize(self,
               last_values: torch.Tensor,
               last_dones: torch.Tensor) -> RolloutBatch:
    """
    Finalize and compute the returns (sum of discounted rewards) and GAE advantage.
    Adapted from Stable-Baselines PPO2.

    Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
    to compute the advantage. To obtain vanilla advantage (A(s) = R - V(S))
    where R is the discounted reward with value bootstrap,
    set ``gae_lambda=1.0`` during initialization.

    Parameters
    ----------
      last_values: (torch.Tensor) estimated value of the current state
        following the current policy.
      last_dones: (torch.Tensor) 
        End of episode signal.
    """
    assert self.full(), "Can only finalize RolloutBuffer when RolloutBuffer is full"

    # Stack experiences from the buffer into tensors
    observations = torch.stack(
        [experience.observation for experience in self.buffer], dim=0)
    actions = torch.stack(
        [experience.action for experience in self.buffer], dim=0)
    rewards = torch.stack(
        [experience.reward for experience in self.buffer], dim=0)
    dones = torch.stack([experience.done for experience in self.buffer], dim=0)
    values = torch.stack(
        [experience.value for experience in self.buffer], dim=0)
    log_probs = torch.stack(
        [experience.log_prob for experience in self.buffer], dim=0)

    assert last_values.device == values.device, 'All value function outputs must be on same device'

    # Compute advantages and returns
    advantages = torch.zeros_like(rewards)
    last_advantage_estimate = 0
    for step in reversed(range(self.n_rollout_steps)):
      if step == self.n_rollout_steps - 1:
        next_non_terminal = 1.0 - last_dones
        next_values = last_values
      else:
        next_non_terminal = 1.0 - dones[step + 1]
        next_values = values[step + 1]
      delta = rewards[step] + self.gamma * \
          next_values * next_non_terminal - values[step]
      advantages[step] = delta + self.gamma * \
          self.gae_lambda * next_non_terminal * last_advantage_estimate
      last_advantage_estimate = advantages[step]
    returns = advantages + values

    # Reshape experience tensors
    observations = observations.view(
        (-1, *observations.shape[2:])).detach()
    actions = actions.view((-1, *actions.shape[2:])).detach()
    rewards = rewards.flatten().detach()
    values = values.flatten().detach()
    log_probs = log_probs.flatten().detach()
    advantages = advantages.flatten().detach()
    returns = returns.flatten().detach()

    # Return a batch of experiences
    return RolloutBatch(
        observations=observations,
        actions=actions,
        values=values,
        log_probs=log_probs,
        advantages=advantages,
        returns=returns)