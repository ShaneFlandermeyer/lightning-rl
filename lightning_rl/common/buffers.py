from collections import deque, namedtuple
from typing import NamedTuple, Optional, Tuple
from gymnasium import spaces
import numpy as np

import torch
from lightning_rl.common.advantage import estimate_advantage

from lightning_rl.common.utils import get_action_dim, get_obs_shape

class ReplayBuffer():
  """
  A generalized replay buffer class for storing arbitrary tensors
  """
  def __init__(self, capacity: int, n_envs: int = 1):
    self.capacity = capacity
    self.n_envs = n_envs
    # Store the names of all tensors that are created
    self.items = set()
    
    self.index = 0
    self.full = False
  
  def __len__(self):
    return self.capacity if self.full else self.index
  
  def create_tensor(self,
                    name: str,
                    size: Tuple[int],
                    dtype: Optional[np.dtype] = None):
    """
    Allocate memory for a new tensor in the buffer

    Parameters
    ----------
    name : str
        Property name of the tensor
    size : Tuple[int]
        Size of individual samples in the tensor
    dtype : Optional[np.dtype], optional
        Data type of tensor elements, by default None
    """
    self.items.add(name)
    if dtype is None:
      dtype = np.float32
    setattr(self, name, np.zeros((self.capacity, *size), dtype=dtype))
  
  def add(self, **data: np.ndarray):
    """
    Add new samples to the buffer

    Raises
    ------
    ValueError
        If the name of a tensor does not match any of the names in the buffer
    """
    assert len(data) == len(self.items), "All tensors must be updated at the same time"
    
    istart = self.index
    iend = (self.index + self.n_envs) % self.capacity
    
    for name, value in data.items():
      # Without this, the code below will fail for 1D arrays since buffer values are assigned to row slices, which must be 2D column vectors.
      if isinstance(value, np.ndarray) and value.ndim == 1:
        value = value[:, np.newaxis]
      
      if name in self.items:
        if istart > iend:
          getattr(self, name)[istart:] = value[:self.capacity - istart]
          getattr(self, name)[:iend] = value[self.capacity - istart:]
        else:
          getattr(self, name)[istart:iend] = value
      else:
        raise ValueError(f"kwarg '{name}' not found in buffer")
       
    # Update the buffer index
    if iend == self.capacity or istart > iend:
      self.full = True
    self.index = (self.index + self.n_envs) % self.capacity
        
  def sample(self, batch_size: int, replace=False):
    """
    Sample a batch of experiences from the buffer

    Parameters
    ----------
    batch_size : int
        Number of elements to sample
    replace : bool, optional
        If true, samples with replacement, by default False

    Returns
    -------
    _type_
        _description_
    """
    indices = np.random.choice(len(self), 
                               size=min(batch_size, len(self)), 
                               replace=replace)
    # Create the batch of samples
    Batch = namedtuple('Batch', self.items)
    batch = Batch(*[getattr(self, name)[indices] for name in self.items])
    return batch
  
  def reset(self):
    """
    Reset the state of the buffer. 
    
    This does not delete any elements. It simply resets the index to 0.
    """
    self.index = 0
    self.full = False
  
if __name__ == '__main__':
  # TODO: Move this to a test file
  rb = ReplayBuffer(10)
  rb.create_tensor('obs', (16,), dtype=np.float32)
  rb.add(obs=np.random.randn(16,))
  rb.sample(1)
  print(rb.items)
  print(rb.obs)


# # Off policy experiences
# class OffPolicyExperience(NamedTuple):
#   """
#   An single off-policy experience
#   """
#   state: np.ndarray
#   action: np.ndarray
#   reward: float
#   next_state: np.ndarray
#   done: bool


# class OffPolicyExperienceBatch(NamedTuple):
#   """
#   A batch of off-policy experiences
#   """
#   states: torch.Tensor
#   actions: torch.Tensor
#   rewards: torch.Tensor
#   next_states: torch.Tensor
#   dones: torch.Tensor

# # On-policy experiences
# class RolloutExperience(NamedTuple):
#   """
#   A single policy gradient rollout experience (potentially from multiple environments)
#   """
#   observation: torch.Tensor
#   action: torch.Tensor
#   reward: torch.Tensor
#   done: torch.Tensor
#   value: torch.Tensor
#   log_prob: torch.Tensor

# class RolloutBatch(NamedTuple):
#   """
#   A batch of complete rollout experiences.
#   """
#   observations: torch.Tensor
#   actions: torch.Tensor
#   values: torch.Tensor
#   log_probs: torch.Tensor
#   advantages: torch.Tensor
#   returns: torch.Tensor

# class RecurrentRolloutBatch(NamedTuple):
#   """
#   A batch of complete rollout experiences.
#   """
#   observations: torch.Tensor
#   actions: torch.Tensor
#   dones: torch.Tensor
#   values: torch.Tensor
#   log_probs: torch.Tensor
#   advantages: torch.Tensor
#   returns: torch.Tensor
#   hidden_states: torch.Tensor



# class ReplayBuffer():
#   """
#   Basic buffer for storing an experience from a single time step
#   """

#   def __init__(
#       self,
#       capacity: int,
#       n_envs: int = 1,
#   ) -> None:
#     assert n_envs == 1, "Replay buffer currently only supports single environments"

#     self.capacity = capacity
#     self.n_envs = 1
#     self.buffer = deque(maxlen=capacity)

#   def __len__(self) -> int:
#     return len(self.buffer)

#   def append(self,
#              state: np.ndarray,
#              action: np.ndarray,
#              reward: np.ndarray,
#              next_state: np.ndarray,
#              done: np.ndarray) -> None:
#     """
#     Add an experience to the buffer
#     """
#     self.buffer.append(OffPolicyExperience(state, action, reward, next_state, done))

#   def sample(self, batch_size: int) -> OffPolicyExperienceBatch:
#     """
#     Return a mini-batch of experiences

#     Parameters
#     ----------
#     batch_size : int
#         Number of experiences in the batch

#     Returns
#     -------
#     ExperienceBatch
#         A named tuple containing torch tensors for each of the experience components
#     """
#     indices = np.random.choice(
#         len(self.buffer), size=min(batch_size, len(self.buffer)), replace=False)
#     states, actions, rewards, next_states, dones = zip(
#         *(self.buffer[idx] for idx in indices))

#     return OffPolicyExperienceBatch(
#         states=torch.as_tensor(np.array(states)[:, 0, :]),
#         actions=torch.as_tensor(np.array(actions)[:, 0, :]),
#         rewards=torch.as_tensor(np.array(rewards)),
#         next_states=torch.as_tensor(np.array(next_states)[:, 0, :]),
#         dones=torch.as_tensor(np.array(dones))
#     )


# class RolloutBuffer():
#   """
#   Buffer used for on-policy algorithms where an n-step rollout is used to estimate the value/advantage of each state

#   Parameters
#   ----------

#   n_rollout_steps: (int)
#     Max number of element in the buffer
#   device: (torch.device)
#     Device containing all experience tensors
#   gae_lambda: (float)
#     Factor for trade-off of bias vs variance for Generalized Advantage Estimator. Equivalent to classic advantage when set to 1.
#   gamma: (float)
#     Discount factor
#   n_envs: (int)
#     Number of parallel environments
#   """

#   def __init__(
#       self,
#       n_rollout_steps: int,
#       gamma: float = 0.99,
#       gae_lambda: float = 1.0,
#       n_envs: int = 1,
#   ) -> None:
#     self.n_rollout_steps = n_rollout_steps
#     self.gae_lambda = gae_lambda
#     self.gamma = gamma
#     self.n_envs = n_envs
#     self.buffer = deque(maxlen=n_rollout_steps)

#   def __len__(self) -> int:
#     return len(self.buffer)

#   def full(self) -> bool:
#     return len(self.buffer) == self.n_rollout_steps

#   def reset(self):
#     self.buffer.clear()

#   def add(
#       self,
#       obs: torch.Tensor,
#       action: torch.Tensor,
#       reward: torch.Tensor,
#       done: torch.Tensor,
#       value: torch.Tensor,
#       log_prob: torch.Tensor
#   ) -> None:
#     """
#     Add a new experience to the buffer.

#     Parameters
#     ----------
#     obs: (torch.tensor)
#       Observation tensor
#     action: (torch.tensor)
#       Action tensor
#     reward: (torch.tensor)
#     done: (torch.tensor)
#       End of episode signal.
#     value: (torch.Tensor)
#       estimated value of the current state following the current policy.
#     log_prob: (torch.Tensor)
#       log probability of the action following the current policy.
#     """
#     experience = RolloutExperience(
#         observation=obs,
#         action=action,
#         reward=reward,
#         done=done,
#         value=value,
#         log_prob=log_prob,
#     )
#     self.buffer.append(experience)

#   def finalize(self,
#                last_values: torch.Tensor,
#                last_dones: torch.Tensor,
#                normalize_returns: bool = False) -> RolloutBatch:
#     """
#     Finalize and compute the returns (sum of discounted rewards) and GAE advantage.
#     Adapted from Stable-Baselines PPO2.

#     Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
#     to compute the advantage. To obtain vanilla advantage (A(s) = R - V(S))
#     where R is the discounted reward with value bootstrap,
#     set ``gae_lambda=1.0`` during initialization.

#     Parameters
#     ----------
#       last_values: (torch.Tensor) estimated value of the current state
#         following the current policy.
#       last_dones: (torch.Tensor)
#         End of episode signal.
#     """
#     assert self.full(), "Can only finalize RolloutBuffer when RolloutBuffer is full"

#     # Stack experiences from the buffer into tensors
#     observations = torch.stack(
#         [experience.observation for experience in self.buffer], dim=0)
#     actions = torch.stack(
#         [experience.action for experience in self.buffer], dim=0)
#     rewards = torch.stack(
#         [experience.reward for experience in self.buffer], dim=0)
#     dones = torch.stack([experience.done for experience in self.buffer], dim=0)
#     values = torch.stack(
#         [experience.value for experience in self.buffer], dim=0)
#     log_probs = torch.stack(
#         [experience.log_prob for experience in self.buffer], dim=0)

#     assert last_values.device == values.device, 'All value function outputs must be on same device'

#     # Compute advantages and returns
#     advantages, returns = estimate_advantage(rewards,
#                                              values,
#                                              dones,
#                                              last_values,
#                                              last_dones,
#                                              self.n_rollout_steps,
#                                              self.gamma,
#                                              self.gae_lambda)
#     if normalize_returns:
#       returns = (returns - returns.mean()) / (returns.std() + 1e-8)
#     # Reshape experience tensors
#     observations = observations.view(
#         (-1, *observations.shape[2:]))
#     actions = actions.view((-1, *actions.shape[2:]))
#     rewards = rewards.view((-1, *rewards.shape[2:]))
#     values = values.view((-1, *values.shape[2:]))
#     log_probs = log_probs.view(-1, *log_probs.shape[2:])
#     advantages = advantages.view((-1, *advantages.shape[2:]))
#     if log_probs.ndim > 1:
#       # For multi-action spaces
#       advantages.unsqueeze_(-1)
#     returns = returns.view((-1, *returns.shape[2:]))

#     # Return a batch of experiences
#     return RolloutBatch(
#         observations=observations,
#         actions=actions,
#         values=values,
#         log_probs=log_probs,
#         advantages=advantages,
#         returns=returns)


# class RecurrentRolloutBuffer():
#   """
#   Buffer used for on-policy algorithms where an n-step rollout is used to estimate the value/advantage of each state

#   Parameters
#   ----------

#   n_rollout_steps: (int)
#     Max number of element in the buffer
#   device: (torch.device)
#     Device containing all experience tensors
#   gae_lambda: (float)
#     Factor for trade-off of bias vs variance for Generalized Advantage Estimator. Equivalent to classic advantage when set to 1.
#   gamma: (float)
#     Discount factor
#   n_envs: (int)
#     Number of parallel environments
#   """

#   def __init__(
#       self,
#       n_rollout_steps: int,
#       gamma: float = 0.99,
#       gae_lambda: float = 1.0,
#       n_envs: int = 1,
#   ) -> None:
#     self.n_rollout_steps = n_rollout_steps
#     self.gae_lambda = gae_lambda
#     self.gamma = gamma
#     self.n_envs = n_envs
#     self.buffer = deque(maxlen=n_rollout_steps)

#   def __len__(self) -> int:
#     return len(self.buffer)

#   def full(self) -> bool:
#     return len(self.buffer) == self.n_rollout_steps

#   def reset(self):
#     self.buffer.clear()

#   def add(
#       self,
#       obs: torch.Tensor,
#       action: torch.Tensor,
#       reward: torch.Tensor,
#       done: torch.Tensor,
#       value: torch.Tensor,
#       log_prob: torch.Tensor,
#   ) -> None:
#     """
#     Add a new experience to the buffer.

#     Parameters
#     ----------
#     obs: (torch.tensor)
#       Observation tensor
#     action: (torch.tensor)
#       Action tensor
#     reward: (torch.tensor)
#     done: (torch.tensor)
#       End of episode signal.
#     value: (torch.Tensor)
#       estimated value of the current state following the current policy.
#     log_prob: (torch.Tensor)
#       log probability of the action following the current policy.
#     """
#     experience = RolloutExperience(
#         observation=obs,
#         action=action,
#         reward=reward,
#         done=done,
#         value=value,
#         log_prob=log_prob,
#     )
#     self.buffer.append(experience)

#   def finalize(self,
#                last_values: torch.Tensor,
#                last_dones: torch.Tensor,
#                normalize_returns: bool = False) -> RolloutBatch:
#     """
#     Finalize and compute the returns (sum of discounted rewards) and GAE advantage.
#     Adapted from Stable-Baselines PPO2.

#     Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
#     to compute the advantage. To obtain vanilla advantage (A(s) = R - V(S))
#     where R is the discounted reward with value bootstrap,
#     set ``gae_lambda=1.0`` during initialization.

#     Parameters
#     ----------
#       last_values: (torch.Tensor) estimated value of the current state
#         following the current policy.
#       last_dones: (torch.Tensor)
#         End of episode signal.
#     """
#     assert self.full(), "Can only finalize RolloutBuffer when RolloutBuffer is full"

#     # Stack experiences from the buffer into tensors
#     observations = torch.stack(
#         [experience.observation for experience in self.buffer], dim=0)
#     actions = torch.stack(
#         [experience.action for experience in self.buffer], dim=0)
#     rewards = torch.stack(
#         [experience.reward for experience in self.buffer], dim=0)
#     dones = torch.stack([experience.done for experience in self.buffer], dim=0)
#     values = torch.stack(
#         [experience.value for experience in self.buffer], dim=0)
#     log_probs = torch.stack(
#         [experience.log_prob for experience in self.buffer], dim=0)
#     assert last_values.device == values.device, 'All value function outputs must be on same device'

#     # Compute advantages and returns
#     advantages, returns = estimate_advantage(rewards,
#                                              values,
#                                              dones,
#                                              last_values,
#                                              last_dones.float(),
#                                              self.n_rollout_steps,
#                                              self.gamma,
#                                              self.gae_lambda)

#     # Reshape experience tensors
#     observations = observations.view(
#         (-1, *observations.shape[2:]))
#     actions = actions.view((-1, *actions.shape[2:]))
#     rewards = rewards.view((-1, *rewards.shape[2:]))
#     dones = dones.view((-1, *dones.shape[2:]))
#     values = values.view((-1, *values.shape[2:]))
#     log_probs = log_probs.view(-1, *log_probs.shape[2:])
#     advantages = advantages.view((-1, *advantages.shape[2:]))
#     if log_probs.ndim > 1:
#       # For multi-action spaces
#       advantages.unsqueeze_(-1)
#     returns = returns.view((-1, *returns.shape[2:]))

#     # Return a batch of experiences
#     return RecurrentRolloutBatch(
#         observations=observations,
#         actions=actions,
#         dones=dones,
#         values=values,
#         log_probs=log_probs,
#         advantages=advantages,
#         returns=returns,
#         hidden_states=None)
