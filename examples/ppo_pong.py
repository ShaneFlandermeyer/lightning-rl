from typing import Tuple
import gym
import numpy as np

import torch
import pytorch_lightning as pl
from torch import nn
from lightning_rl.models.on_policy_models import PPO
from torch import distributions


class AtariPPO(PPO):
  def __init__(self,
               env: gym.Env,
               **kwargs):
    # **kwargs will pass our arguments on to PPO
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
    actions = dist.sample()
    values = self.critic(features).flatten()
    return actions, values

  def evaluate_actions(self, observations: torch.Tensor, actions: torch.Tensor):
    features = self.feature_net(observations)
    action_probabilities = self.actor(features)
    dist = distributions.Categorical(probs=action_probabilities)
    log_prob = dist.log_prob(actions)
    entropy = dist.entropy()
    return log_prob, entropy

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=2.5e-4, eps=1e-5)
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
  n_env = 8

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

  ppo = AtariPPO(env=env,
                 n_rollouts_per_epoch=10,
                 n_steps_per_rollout=128,
                 n_gradient_steps=10,
                 batch_size=256,
                 gamma=0.99,
                 gae_lambda=0.95,
                 value_coef=1,
                 entropy_coef=0.01,
                 seed=seed,
                 normalize_advantage=True,
                 policy_clip_range=0.1,
                 )

  trainer = pl.Trainer(
      max_time="00:03:00:00",
      gradient_clip_val=0.5,
      accelerator='gpu',
      devices=1,
  )

  trainer.fit(ppo)
