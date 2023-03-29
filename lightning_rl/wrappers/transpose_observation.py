from typing import List
import gymnasium as gym
import numpy as np

class TransposeObservation(gym.ObservationWrapper):
  
  def __init__(self, env: gym.Env, axes: List[int]):
    super().__init__(env)
    self.axes = axes
    new_shape = np.array(self.observation_space.shape)[axes]
    self.observation_space = gym.spaces.Box(
      low=self.observation_space.low.transpose(axes),
      high=self.observation_space.high.transpose(axes),
      shape=new_shape,
      dtype=self.observation_space.dtype
    )
    
  def observation(self, observation: np.ndarray) -> np.ndarray:
    return observation.transpose(self.axes)