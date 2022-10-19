from collections import deque
from typing import NamedTuple
from gym import spaces
import numpy as np

import torch


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
    
