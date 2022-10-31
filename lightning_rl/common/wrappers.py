import gym
from lightning_rl.models import ICM
import numpy as np
import torch


class CuriosityWrapper(gym.Wrapper):
  """
  Generates an additional intrinsic reward using an intrinsic curiosity module and adds it to the extrinsic reward from the environment.

  See: https://arxiv.org/abs/1705.05363
  """

  def __init__(self,
               env: gym.Env,
               icm: ICM) -> None:
    super().__init__(env)
    self.icm = icm
    self.state = torch.as_tensor(
        self.env.reset(), dtype=torch.float32).to(self.icm.device)

  def step(self, action):
    next_state, extrinsic_reward, done, info = self.env.step(action)

    action = torch.as_tensor(action, dtype=torch.float32).to(self.icm.device)
    state = torch.as_tensor(
        self.state, dtype=torch.int64).to(self.icm.device)
    next_state = torch.as_tensor(
        next_state, dtype=torch.float32).to(self.icm.device)
    
    intrinsic_reward = self.icm.compute_intrinsic_reward(
        state, next_state, action)
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
