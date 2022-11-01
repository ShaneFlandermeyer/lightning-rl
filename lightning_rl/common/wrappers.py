import gym
from lightning_rl.models.icm import ICM
from typing import Union
import numpy as np
import cv2
import torch

class CuriosityWrapper(gym.vector.VectorEnvWrapper):
  """
  Generates an additional intrinsic reward using an intrinsic curiosity module and adds it to the extrinsic reward from the environment.

  See: https://arxiv.org/abs/1705.05363
  """

  def __init__(self,
               env: Union[gym.Env, gym.vector.VectorEnv, str],
               icm: ICM) -> None:
    super().__init__(env)
    
    # Create the gym environment
    if isinstance(env, str):
      self.env = gym.make(env)
    else:
      self.env = env
    
    self.state = self.env.reset()
    self.icm = icm
    
    
    

  def step(self, action):
    next_state, extrinsic_reward, done, info = self.env.step(action)

    action_tensor = torch.as_tensor(
        action, dtype=torch.int64).to(self.icm.device)
    state_tensor = torch.as_tensor(
        self.state, dtype=torch.float32).to(self.icm.device)
    next_state_tensor = torch.as_tensor(
        next_state, dtype=torch.float32).to(self.icm.device)

    intrinsic_reward = self.icm.compute_intrinsic_reward(
        state_tensor, next_state_tensor, action_tensor)
    reward = extrinsic_reward + intrinsic_reward
    self.state = next_state
    return next_state, reward, done, info


class ImageToPytorch(gym.ObservationWrapper):
  """
  Convert the observation image from gym's (H, W, C) format to Pytorch's (C, H, W) format.
  
  This function will fail if the image does not have a channel dimension.
  """

  def __init__(self, env):
    super().__init__(env)
    old_shape = self.observation_space.shape
    new_shape = (old_shape[-1], old_shape[0], old_shape[1])
    self.observation_space = gym.spaces.Box(
        low=0.0, high=1.0, shape=new_shape, dtype=np.float32)

  def observation(self, observation: np.ndarray) -> np.ndarray:
    return np.moveaxis(observation, 2, 0)
  
