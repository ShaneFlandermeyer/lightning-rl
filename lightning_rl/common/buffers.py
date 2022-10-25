from collections import deque
from typing import NamedTuple
from gym import spaces
import numpy as np

import torch

from lightning_rl.common.utils import get_action_dim, get_obs_shape


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


class RolloutSamples(NamedTuple):
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
  Rollout buffer used in on-policy algorithms like A2C/PPO.

  :param buffer_size: (int) Max number of element in the buffer
  :param observation_space: (spaces.Space) Observation space
  :param action_space: (spaces.Space) Action space
  :param device: (torch.device)
  :param gae_lambda: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
      Equivalent to classic advantage when set to 1.
  :param gamma: (float) Discount factor
  :param n_envs: (int) Number of parallel environments
  """

  def __init__(
      self,
      buffer_size: int,
      observation_space: spaces.Space,
      action_space: spaces.Space,
      gamma: float = 0.99,
      gae_lambda: float = 1.0,
      n_envs: int = 1,
  ) -> None:
    self.buffer_size = buffer_size
    self.observation_space = observation_space
    self.action_space = action_space
    self.obs_shape = get_obs_shape(observation_space)
    self.action_dim = get_action_dim(action_space)
    self.gae_lambda = gae_lambda
    self.gamma = gamma
    self.n_envs = n_envs
    self.pos = 0
    self.full = False

    self.reset()

  def size(self) -> int:
    """
    :return: (int) The current size of the buffer
    """
    if self.full:
      return self.buffer_size
    return self.pos

  def reset(self):
    self.pos = 0
    self.initialized = False

  def add(
      self,
      obs: torch.Tensor,
      action: torch.Tensor,
      reward: torch.Tensor,
      done: torch.Tensor,
      value: torch.Tensor,
      log_prob: torch.Tensor,
  ) -> None:
    """
    Add experiences to the rollout buffer

    Parameters
    ----------
    obs : torch.Tensor
        Observation
    action : torch.Tensor
        Action
    reward : torch.Tensor
        Reward
    done : torch.Tensor
        End of episode signal
    value : torch.Tensor
        Estimated value of the current state, following the current policy
    log_prob : torch.Tensor
        Log probability of the actions following the current policy
    """
    if not self.initialized:
      self.observations = torch.zeros(
          (self.buffer_size, self.n_envs) + self.obs_shape, device=obs.device)
      self.actions = torch.zeros(
          (self.buffer_size, self.n_envs, self.action_dim), device=action.device)
      self.rewards = torch.zeros(
          (self.buffer_size, self.n_envs), device=reward.device)
      self.dones = torch.zeros(
          (self.buffer_size, self.n_envs), device=done.device)
      self.values = torch.zeros(
          (self.buffer_size, self.n_envs), device=value.device)
      self.log_probs = torch.zeros(
          (self.buffer_size, self.n_envs), device=log_prob.device)
      self.initialized = True

    self.observations[self.pos] = obs
    self.actions[self.pos] = action
    self.rewards[self.pos] = reward
    self.dones[self.pos] = done
    self.values[self.pos] = value
    self.log_probs[self.pos] = log_prob
    self.pos += 1
    if self.pos == self.buffer_size:
        self.full = True

  def finalize(self, last_values: torch.Tensor, last_dones: torch.Tensor) -> None:
    """
    Finalize and compute the returns (sum of discounted rewards) and GAE advantage.

    Uses generalized advantage estimation to compute the advantage. To obtain vanilla advantage (A(s) = R - V(s)) where R is the discounted reward with value bootstrap, set ```gae_lambda=1.0`` during initialization.

    Parameters
    ----------
    last_values : torch.Tensor
        Estimated value of the current state following the current policy.
    last_dones : torch.Tensor
        End of episode signal
    """
    assert self.full, "Buffer must be full before finalizing"

    assert last_values.device == self.values.device, "Values function outputs must be on the same device"

    last_gae_lam = 0
    advantages = torch.zeros_like(self.rewards)
    for step in reversed(range(self.buffer_size)):
      if step == self.buffer_size - 1:
        next_non_terminal = 1.0 - last_dones
        next_values = last_values
      else:
        next_non_terminal = 1.0 - self.dones[step + 1]
        next_values = self.values[step + 1]
      delta = self.rewards[step] + self.gamma * \
          next_values * next_non_terminal - self.values[step]
      last_gae_lam = delta + self.gamma * \
          self.gae_lambda * next_non_terminal * last_gae_lam
    returns = advantages + self.values

    self.observations = self.observations.view(
        (-1, *self.observations.shape[2:]))
    self.actions = self.actions.view((-1, *self.actions.shape[2:]))
    self.rewards = self.rewards.flatten()
    self.values = self.values.flatten()
    self.log_probs = self.log_probs.flatten()
    advantages = advantages.flatten()
    returns = returns.flatten()

    return RolloutSamples(self.observations, self.actions, self.values, self.log_probs, advantages, returns)
