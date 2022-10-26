from gym import spaces
from typing import Tuple
import numpy as np
import torch
import gym


def get_obs_shape(observation_space: spaces.Space) -> Tuple[int, ...]:
  """
  Get the shape of the observation (useful for the buffers).

  :param observation_space: (spaces.Space)
  :return: (Tuple[int, ...])
  """
  if isinstance(observation_space, spaces.Box):
    return observation_space.shape
  elif isinstance(observation_space, spaces.Discrete):
    # Observation is an int
    return (1,)
  elif isinstance(observation_space, spaces.MultiDiscrete):
    # Number of discrete features
    return (int(len(observation_space.nvec)),)
  elif isinstance(observation_space, spaces.MultiBinary):
    # Number of binary features
    return (int(observation_space.n),)
  else:
    raise NotImplementedError()


def get_action_dim(action_space: spaces.Space) -> int:
  """
  Get the dimension of the action space.

  :param action_space: (spaces.Space)
  :return: (int)
  """
  if isinstance(action_space, spaces.Box):
    return int(np.prod(action_space.shape))
  elif isinstance(action_space, spaces.Discrete):
    # Action is an int
    return 1
  elif isinstance(action_space, spaces.MultiDiscrete):
    # Number of discrete actions
    return int(len(action_space.nvec))
  elif isinstance(action_space, spaces.MultiBinary):
    # Number of binary actions
    return int(action_space.n)
  else:
    raise NotImplementedError()

# From stable baselines


def explained_variance(y_pred: torch.tensor, y_true: torch.tensor) -> np.ndarray:
  """
  Computes fraction of variance that ypred explains about y.
  Returns 1 - Var[y-ypred] / Var[y]

  interpretation:
      ev=0  =>  might as well have predicted zero
      ev=1  =>  perfect prediction
      ev<0  =>  worse than just predicting zero

  :param y_pred: (np.ndarray) the prediction
  :param y_true: (np.ndarray) the expected value
  :return: (float) explained variance of ypred and y
  """
  assert y_true.ndim == 1 and y_pred.ndim == 1
  var_y = torch.var(y_true)
  return torch.nan if var_y == 0 else 1 - torch.var(y_true - y_pred) / var_y


def clip_actions(actions: torch.Tensor, action_space: spaces.Space) -> np.ndarray:
  """
  Clip actions to stay in the bounds of the action space

  Parameters
  ----------
  actions : torch.Tensor
      Action tensor
  action_space : spaces.Space
      Action space object

  Returns
  -------
  torch.Tensor
      _description_
  """
  # Rescale and perform action
  clipped_actions = actions.cpu().numpy()
  # Clip actions to avoid out of bound errors
  if isinstance(action_space, gym.spaces.Box):
    clipped_actions = np.clip(clipped_actions,
                              action_space.low,
                              action_space.high)
  elif isinstance(action_space, gym.spaces.Discrete):
    clipped_actions = clipped_actions.astype(np.int32)
    
  return clipped_actions