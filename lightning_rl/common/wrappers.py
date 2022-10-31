import gym
from lightning_rl.models.icm import ICM
from typing import Union
import numpy as np
import cv2
import torch


class CuriosityRewardWrapper(gym.Wrapper):
  """
  Generates an additional intrinsic reward using an intrinsic curiosity module and adds it to the extrinsic reward from the environment.

  See: https://arxiv.org/abs/1705.05363
  """

  def __init__(self,
               env: Union[gym.Env, gym.vector.VectorEnv, str]) -> None:
    super().__init__(env)
    
    # Create the gym environment
    if isinstance(env, str):
      self.env = gym.make(env)
    else:
      self.env = env

    # Make the environment a vector environment
    if not isinstance(self.env, gym.vector.VectorEnv):
      self.env = gym.vector.SyncVectorEnv([lambda: self.env])
    
    self.state = self.env.reset()

  def step(self, action, icm: ICM):
    next_state, extrinsic_reward, done, info = self.env.step(action)

    action_tensor = torch.as_tensor(
        action, dtype=torch.int64).to(icm.device)
    state_tensor = torch.as_tensor(
        self.state, dtype=torch.float32).to(icm.device)
    next_state_tensor = torch.as_tensor(
        next_state, dtype=torch.float32).to(icm.device)

    intrinsic_reward = icm.compute_intrinsic_reward(
        state_tensor, next_state_tensor, action_tensor)
    reward = extrinsic_reward + intrinsic_reward
    self.state = next_state
    return next_state, reward, done, info


class ImageToPytorch(gym.ObservationWrapper):
  """
  Convert the observation image from gym's (H, W, C) format to Pytorch's (C, H, W) format.
  """

  def __init__(self, env):
    super().__init__(env)
    old_shape = self.observation_space.shape
    new_shape = (old_shape[-1], old_shape[0], old_shape[1])
    self.observation_space = gym.spaces.Box(
        low=0.0, high=1.0, shape=new_shape, dtype=np.float32)

  def observation(self, observation):
    return np.moveaxis(observation, 2, 0)

class ProcessFrame84(gym.ObservationWrapper):
  """
  Downsample the image to 84x84 pixels and convert it to a grayscale image.
  """

  def __init__(self, env):
    super().__init__(env)
    self.observation_space = gym.spaces.Box(
        low=0, high=255, shape=(1, 84, 84), dtype=np.uint8)

  def observation(self, frame):
    frame = frame[34:34+160, :160]
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
    return frame[:, :, None]