import inspect
import time
import warnings
import pytorch_lightning as pl
import gymnasium as gym
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
      seed: Optional[int] = None,
      **kwargs
  ) -> None:
    super().__init__()

    if isinstance(env, str):
      env = gym.make(env)

    # The data collection loops assume the environment is vectorized. If this is not the case, wrap the environment in a SyncVectorEnv with 1 environment.
    is_vector_env = getattr(env, "is_vector_env", False)
    if not is_vector_env:
      self.env = gym.vector.SyncVectorEnv([lambda: env])
    else:
      self.env = env

    self.observation_space = self.env.single_observation_space
    self.action_space = self.env.single_action_space
    self.n_envs = self.env.num_envs

    if seed:
      self.seed = seed
      self.set_random_seed(self.seed)
    else:
      self.seed = None
    if not support_multi_env and self.n_envs > 1:
      raise ValueError(
          "Error: the model does not support multiple envs; it requires " "a single vectorized environment."
      )

    self.start_time = time.time()
    self.reset()
    
  def reset(self) -> None:
    """
    Reset the environment
    """
    if self.seed is not None:
      self._last_obs = self.env.reset()[0]
    else:
      self._last_obs = self.env.reset(seed=self.seed)[0]
    self._last_dones = np.zeros((self.env.num_envs,), dtype=np.uint8)

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
  
  def on_train_epoch_end(self) -> None:
    # Log episode statistics at the end of each epoch if the environment has the wrapper
    if hasattr(self.env, "return_queue") and hasattr(self.env, "length_queue"):
      if self.env.return_queue and self.env.length_queue:
        self.log_dict({
            'train/mean_episode_reward': np.mean(self.env.return_queue),
            'train/mean_episode_length': np.mean(self.env.length_queue),
            'train/total_step_count': float(self.total_step_count),
        },
            prog_bar=True, logger=True)
