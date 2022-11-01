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
from lightning_rl.models.on_policy_models.on_policy_model import OnPolicyDataLoader


class ICMModel(ICM):
  """
  Implements the ICM architecture from Pathak2017

  NOTE: Assumes inputs are already batched (true for vectorized environment)
  """

  def __init__(self,
               env: gym.Env,
               **kwargs):
    super().__init__(
        env=env,
        **kwargs)

    # Embedded feature space transformation
    self.feature_net = nn.Sequential(
        nn.Conv2d(
            in_channels=self.observation_space.shape[0],
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
        nn.Linear(feature_shape[1] + self.action_space.n, 256),
        nn.Linear(256, feature_shape[1]),
    )

    # Takes the concatenation of the current and next state (in the embedded feature space) and tries to predict the action
    self.inverse_net = nn.Sequential(
        nn.Linear(2*feature_shape[1], 256),
        nn.Linear(256, self.action_space.n),
    )

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

    return (actual_next_state_feature,
            predicted_next_state_feature,
            predicted_action)

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
    return optimizer


class A2CModel(A2C):
  def __init__(self,
               env: gym.Env,
               **kwargs):
    # **kwargs will pass our arguments on to A2C
    super().__init__(env=env,
                     **kwargs)
    self.feature_net = nn.Sequential(
        nn.Conv2d(
            in_channels=self.observation_space.shape[0],
            out_channels=32,
            kernel_size=8,
            stride=4
        ),
        nn.ReLU(),
        nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=4,
            stride=2
        ),
        nn.ReLU(),
        nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1
        ),
        nn.Flatten(),
        # nn.ReLU(),
    )
    # Dimensionality of the embedded feature space
    feature_shape = self._get_output_shape(self.feature_net)
    self.actor = nn.Sequential(
        nn.Linear(feature_shape[1], 512),
        nn.ReLU(),
        nn.Linear(512, self.action_space.n),
        nn.Softmax(dim=1),
    )
    self.critic = nn.Sequential(
        nn.Linear(feature_shape[1], 512),
        nn.ReLU(),
        nn.Linear(512, 1),
    )

    self.save_hyperparameters()

  def forward(self, x: torch.Tensor):
    x = self.feature_net(x)
    out = self.actor(x)
    dist = distributions.Categorical(probs=out)
    return dist, self.critic(x).flatten()

  def predict(self, x: torch.Tensor, deterministic: bool = False, **kwargs):
    x = self.feature_net(x)
    out = self.actor(x)
    if deterministic:
      out = torch.max(out, dim=1)[1]
    else:
      out = distributions.Categorical(probs=out).sample()
    return out.cpu().numpy()

  def configure_optimizers(self):
    optimizer = torch.optim.RMSprop(self.parameters(), lr=1e-4, eps=1e-5)
    return optimizer


if __name__ == '__main__':
  # Unwrapped environments
  env_id = "PongNoFrameskip-v4"

  # Preprocessing wrappers
  env = make_atari(env_id)
  env = wrap_deepmind(env, scale=True, frame_stack=True)
  env = wrap_pytorch(env)
  eval_env = env

  # Vectorize the environment
  n_env = 16
  env = gym.vector.AsyncVectorEnv([lambda: env] * n_env)

  a2c = A2CModel(env=env,
                 n_rollouts_per_epoch=300,
                 n_steps_per_rollout=5,
                 gae_lambda=1.0,
                 seed=np.random.randint(0, 100000),
                 normalize_advantage=True,
                 entropy_coef=0.01,
                 value_coef=0.25,
                 )

  # Create evaluation callback
  eval_callback = EvalCallback(eval_env, n_eval_episodes=3)
  
  env.reset()
  eval_env.reset()

  trainer = pl.Trainer(
      max_time="00:03:00:00",
      gradient_clip_val=0.5,
      accelerator='gpu',
      devices=1,
      callbacks=[eval_callback],
  )

  trainer.fit(a2c)
