from typing import Tuple
import gym
import numpy as np

import torch
import pytorch_lightning as pl
from torch import nn
from lightning_rl.models.on_policy_models import A2C
from torch import distributions


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
        nn.Linear(512, self.action_space.n),
        nn.Softmax(dim=1),
    )

    self.critic = nn.Sequential(
        nn.Linear(512, 1)
    )

    self.save_hyperparameters()

  def forward(self, x: torch.Tensor):
    features = self.feature_net(x)
    action_probabilities = self.actor(features)
    dist = distributions.Categorical(probs=action_probabilities)
    return dist, self.critic(features).flatten()

  def configure_optimizers(self):
    # optimizer = RMSpropTFLike(self.parameters(), lr=7e-4, eps=1e-5)
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, eps=1e-3)
    return optimizer

  def on_train_epoch_end(self) -> None:
    if self.env.return_queue and self.env.length_queue:
      self.log_dict({
          'train/mean_episode_reward': np.mean(self.env.return_queue),
          'train/mean_episode_length': np.mean(self.env.length_queue),
          'train/total_step_count': float(self.total_step_count),
      },
          prog_bar=True, logger=True)


if __name__ == '__main__':
  # Unwrapped environments
  env_id = "PongNoFrameskip-v4"
  seed = np.random.randint(0, 2**32 - 1)

  # Vectorize the environment
  n_env = 50

  env = gym.make(env_id)
  env = gym.wrappers.AtariPreprocessing(
      env=env,
      frame_skip=4,
      screen_size=84,
      grayscale_obs=True,
      grayscale_newaxis=False,
      scale_obs=True,
  )
  env = gym.wrappers.FrameStack(
      env=env,
      num_stack=4,
  )
  env = gym.vector.AsyncVectorEnv([lambda: env]*n_env)
  env = gym.wrappers.RecordEpisodeStatistics(env=env, deque_size=20)

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
      max_time="00:03:00:00",
      gradient_clip_val=0.1,
      accelerator='gpu',
      devices=1,
  )

  trainer.fit(a2c)
