from typing import Tuple
import gym
import numpy as np

import torch
from lightning_rl.common.wrappers import ImageToPytorch, CuriosityWrapper
from lightning_rl.common.atari_wrappers import *
from lightning_rl.models import ICM
from lightning_rl.common.callbacks import EvalCallback
import pytorch_lightning as pl
from torch import nn
from lightning_rl.common.layers import NoisyLinear
from lightning_rl.models.on_policy_models import A2C
from torch import distributions
from lightning_rl.common.atari_wrappers import *
# from lightning_rl.models.on_policy_models.on_policy_model import OnPolicyDataLoader
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
from stable_baselines3.common.env_util import make_vec_env, make_atari_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage

# class ICMModel(ICM):
#   """
#   Implements the ICM architecture from Pathak2017

#   NOTE: Assumes inputs are already batched (true for vectorized environment)
#   """

#   def __init__(self,
#                env: gym.Env,
#                **kwargs):
#     super().__init__(
#         env=env,
#         **kwargs)

#     # Embedded feature space transformation
#     self.feature_net = nn.Sequential(
#         nn.Conv2d(
#             in_channels=self.observation_space.shape[0],
#             out_channels=32,
#             kernel_size=3,
#             stride=2,
#             padding=1,
#         ),
#         nn.ELU(),
#         nn.Conv2d(
#             in_channels=32,
#             out_channels=32,
#             kernel_size=3,
#             stride=2,
#             padding=1,
#         ),
#         nn.ELU(),
#         nn.Conv2d(
#             in_channels=32,
#             out_channels=32,
#             kernel_size=3,
#             stride=2,
#             padding=1,
#         ),
#         nn.ELU(),
#         nn.Conv2d(
#             in_channels=32,
#             out_channels=32,
#             kernel_size=3,
#             stride=2,
#             padding=1,
#         ),
#         nn.ELU(),
#         nn.Flatten(),
#     )
#     # Dimensionality of the embedded feature space
#     feature_shape = self._get_output_shape(self.feature_net)

#     # Takes the concatenation of the current state (in the embedded feature space) and the action and tries to predict the (embedded) next state
#     self.forward_net = nn.Sequential(
#         nn.Linear(feature_shape[1] + self.action_space.n, 256),
#         nn.Linear(256, feature_shape[1]),
#     )

#     # Takes the concatenation of the current and next state (in the embedded feature space) and tries to predict the action
#     self.inverse_net = nn.Sequential(
#         nn.Linear(2*feature_shape[1], 256),
#         nn.Linear(256, self.action_space.n),
#     )

#   def forward(self, inputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
#     state, next_state, action = inputs

#     encoded_state = self.feature_net(state)
#     encoded_next_state = self.feature_net(next_state)

#     # Predict the action causing the state transition using the inverse dynamics model
#     predicted_action = torch.cat((encoded_state, encoded_next_state), 1)
#     predicted_action = self.inverse_net(predicted_action)

#     predicted_next_state_feature = torch.cat((encoded_state, action), 1)
#     predicted_next_state_feature = self.forward_net(
#         predicted_next_state_feature)

#     actual_next_state_feature = encoded_next_state

#     return (actual_next_state_feature,
#             predicted_next_state_feature,
#             predicted_action)

#   def configure_optimizers(self):
#     optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
#     return optimizer


class AtariA2C(A2C):
  def __init__(self,
               env: gym.Env,
               **kwargs):
    # **kwargs will pass our arguments on to A2C
    super().__init__(env=env,
                     **kwargs)
    self.feature_net = nn.Sequential(
        nn.Conv2d(
            self.observation_space.shape[0], 32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.LazyLinear(512),
        nn.ReLU(),
    )

    self.actor = nn.Sequential(
        nn.Linear(512, env.action_space.n),
        nn.Softmax(dim=1),
    )

    self.critic = nn.Sequential(
        nn.Linear(512, 1)
    )

    self.save_hyperparameters()

  def forward(self, x: torch.Tensor):
    fx = x.float() / 256
    features = self.feature_net(fx)
    action_probabilities = self.actor(features)
    dist = distributions.Categorical(probs=action_probabilities)
    return dist, self.critic(features).flatten()

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, eps=1e-3)
    # optimizer = RMSpropTFLike(self.parameters(), lr=7e-4, eps=1e-5)
    return optimizer


if __name__ == '__main__':
  # Unwrapped environments
  env_id = "PongNoFrameskip-v4"
  seed = np.random.randint(0, 2**32 - 1)

  # Vectorize the environment
  n_env = 50
  # TODO: Replace these with standard gym vector envs
  env = make_atari_env(
      env_id=env_id,
      n_envs=n_env,
      seed=seed,
      vec_env_cls=SubprocVecEnv

  )
  env = VecFrameStack(env, n_stack=4)
  env = VecTransposeImage(env)

  a2c = AtariA2C(env=env,
                 n_rollouts_per_epoch=100,
                 n_steps_per_rollout=5,
                 gae_lambda=1.0,
                 seed=seed,
                 normalize_advantage=True,
                 entropy_coef=0.02,
                 value_coef=0.25,
                 batch_size=64
                 )

  trainer = pl.Trainer(
      max_time="00:05:00:00",
      gradient_clip_val=0.1,
      accelerator='gpu',
      devices=1,
  )

  trainer.fit(a2c)
