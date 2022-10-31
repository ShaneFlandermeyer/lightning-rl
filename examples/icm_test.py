from typing import Tuple
import gym
import numpy as np

import torch
from lightning_rl.common.wrappers import CuriosityRewardWrapper, ImageToPytorch, ProcessFrame84
from lightning_rl.models import ICM
from torch import nn


class ICMModel(ICM):
  """
  Implements the ICM architecture from Pathak2017

  NOTE: Assumes inputs are already batched (true for vectorized environment)
  """

  def __init__(self,
               observation_space: gym.Space,
               action_space: gym.Space,
               **kwargs):
    super().__init__(
        observation_space=observation_space,
        action_space=action_space,
        **kwargs)

    # Embedded feature space transformation
    self.feature_net = nn.Sequential(
        nn.Conv2d(
            in_channels=observation_space.shape[0],
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
        ),
        nn.ELU(),
        nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
        ),
        nn.ELU(),
        nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
        ),
        nn.ELU(),
        nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
        ),
        nn.ELU(),
        nn.Flatten(),
    )
    # Dimensionality of the embedded feature space
    feature_shape = self._get_output_shape(self.feature_net)

    # Takes the concatenation of the current state (in the embedded feature space) and the action and tries to predict the (embedded) next state
    self.forward_net = nn.Sequential(
        nn.Linear(feature_shape[1] + action_space.n, 256),
        nn.Linear(256, feature_shape[1]),
    )

    # Takes the concatenation of the current and next state (in the embedded feature space) and tries to predict the action
    self.inverse_net = nn.Sequential(
        nn.Linear(2*feature_shape[1], 256),
        nn.Linear(256, action_space.n),
    )

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

  def forward(self, inputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
    state, next_state, action = inputs

    encoded_state = self.feature_net(state)
    encoded_next_state = self.feature_net(next_state)

    # Predict the action causing the state transition using the inverse dynamics model
    predicted_action = torch.cat((encoded_state, encoded_next_state), 1)
    predicted_action = self.inverse_net(predicted_action)

    predicted_next_state_feature = torch.cat((encoded_state, action), 1)
    predicted_next_state_feature = self.forward_net(
        predicted_next_state_feature)

    actual_next_state_feature = encoded_next_state
    return actual_next_state_feature, predicted_next_state_feature, predicted_action

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
    return optimizer


if __name__ == '__main__':
  env = gym.make('PongNoFrameskip-v4')
#   env = ProcessFrame84(env)
  env = ImageToPytorch(env)
  env = gym.vector.SyncVectorEnv([lambda: env])

  env = CuriosityRewardWrapper(env)
  icm = ICMModel(env.single_observation_space, env.single_action_space)
  # Vectorize the environment

  for i in range(1000):
    next_state, reward, done, info = env.step(env.action_space.sample(), icm)
    if reward > 1:
        print(next_state.shape, reward, done.shape, info)
  print(icm.feature_net)
  print(icm.forward_net)
  print(icm.inverse_net)
