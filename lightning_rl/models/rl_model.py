import inspect
import warnings
import pytorch_lightning as pl
import gym
from typing import List, Union, Optional, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn


class RLModel(pl.LightningModule):
  """
    Base class for all RL algorithms

    Parameters
    ----------
    env : Union[gym.Env, VecEnv, str]
        The environment to learn from
        (if registered in Gym, can be str. Can be None for loading trained models)
    support_multi_env : bool, optional
        Whether the algorithm supports training
      with multiple environments in parallel, by default False
    seed : Optional[int], optional
        Seed for the pseudo random number generators, by default None

    Raises
    ------
    ValueError
        _description_
  """

  def __init__(
      self,
      env: Union[str, gym.Env, gym.vector.VectorEnv],
      support_multi_env: bool = False,
      seed: Optional[int] = None
  ) -> None:
    super().__init__()

    if isinstance(env, str):
      env = gym.make(env)

    # The data collection loops assume the environment is vectorized. If this is not the case, wrap the environment in a SyncVectorEnv with 1 environment.
    is_vector_env = getattr(env, "is_vector_env", False)
    if not is_vector_env:
      self.env = gym.vector.SyncVectorEnv([lambda: self.env])
    else:
      self.env = env

    self.observation_space = self.env.single_observation_space
    self.action_space = self.env.single_action_space
    self.n_envs = self.env.num_envs

    if seed:
      self.seed = seed
      self.set_random_seed(self.seed)
    if not support_multi_env and self.n_envs > 1:
      raise ValueError(
          "Error: the model does not support multiple envs; it requires " "a single vectorized environment."
      )

    self.reset()

  def act(
      self,
      state: np.ndarray,
      deterministic: bool = False
  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform an action based on the current state of the environment

    Parameters
    ----------
    state : np.ndarray
        Input state
    deterministic : bool, optional
        If true, samples the action deterministically, by default False

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The action to perform
    """
    with torch.no_grad():
      state = torch.tensor(state).to(self.device)
      action = self.predict(state, deterministic=deterministic)

    if isinstance(self.action_space, gym.spaces.Box):
      action = np.clip(action, self.action_space.low, self.action_space.high)
    elif isinstance(self.action_space, (gym.spaces.Discrete,
                                        gym.spaces.MultiDiscrete,
                                        gym.spaces.MultiBinary)):
      action = action.astype(np.int32)

    return action
  
  def predict(self,
              obs: Union[Tuple, Dict[str, Any], np.ndarray, int], deterministic: bool = False) -> np.ndarray:
    """
    Override this function with the predict function of your own mode

    Parameters
    ----------
    obs : Union[Tuple, Dict[str, Any], np.ndarray, int]
        The input observations
    deterministic : bool, optional
        If true, samples the action deterministically, by default False

    Returns
    -------
    np.ndarray
        The chosen action

    Raises
    ------
    NotImplementedError
    """
    raise NotImplementedError

  def reset(self) -> None:
    """
    Reset the environment
    """
    self._last_obs = self.env.reset(seed=self.seed)[0]
    self._last_dones = np.zeros((self.env.num_envs,), dtype=np.bool)

  def save_hyperparameters(self, frame=None, exclude=['env', 'eval_env']):
    """
    Utility function to save the hyperparameters of the model.
    This function behaves identically to LightningModule.save_hyperparameters, but will by default exclude the Gym environments
    #lightningmodule-hyperparameters for more details
    See https://pytorch-lightning.readthedocs.io/en/latest/hyperparameters.html
    """
    if not frame:
      frame = inspect.currentframe().f_back
    if not exclude:
      return super().save_hyperparameters(frame=frame)
    if isinstance(exclude, str):
      exclude = (exclude, )
    init_args = pl.utilities.parsing.get_init_args(frame)
    include = [k for k in init_args.keys() if k not in exclude]
    if len(include) > 0:
      super().save_hyperparameters(*include, frame=frame)
      
  def set_random_seed(self, seed: int) -> None:
    """
    Set the seed for all RNGs

    Parameters
    ----------
    seed : int
        Random seed
    """
    pl.seed_everything(seed)
    if self.env:
      self.action_space.seed(seed)

  def _get_output_shape(self, network: nn.Module) -> Tuple[int]:
    """
    Compute the size of the output of a network for a single example

    Parameters
    ----------
    network : nn.Module
        A Pytorch ANN module

    Returns
    -------
    Tuple[int]
        The output size of the network
    """
    o = network(torch.zeros(1, *self.observation_space.shape))
    return o.shape
