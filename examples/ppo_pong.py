from typing import Tuple
import gym
import numpy as np

import torch
import pytorch_lightning as pl
from torch import nn
from lightning_rl.models.on_policy_models import PPO
from torch import distributions
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
  torch.nn.init.orthogonal_(layer.weight, std)
  torch.nn.init.constant_(layer.bias, bias_const)
  return layer


class AtariPPO(PPO):
  def __init__(self,
               env: gym.Env,
               **kwargs):
    # **kwargs will pass our arguments on to PPO
    super().__init__(env=env,
                     **kwargs)
    self.feature_net = nn.Sequential(
        layer_init(nn.Conv2d(
            self.observation_space.shape[0], 32, kernel_size=8, stride=4)),
        nn.ReLU(),
        layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
        nn.ReLU(),
        layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
        nn.ReLU(),
        nn.Flatten(start_dim=1, end_dim=-1),
        layer_init(nn.Linear(64*7*7, 512)),
        nn.ReLU(),
    )
    self.actor = layer_init(nn.Linear(512, self.action_space.n), std=0.01)
    self.critic = layer_init(nn.Linear(512, 1), std=1)

    self.save_hyperparameters()

  def forward(self, x: torch.Tensor):
    features = self.feature_net(x)
    action_logits = self.actor(features)
    values = self.critic(features).flatten()
    return action_logits, values

  def act(self, x: torch.Tensor):
    action_logits, value = self.forward(x)
    action_dist = Categorical(logits=action_logits)
    action = action_dist.sample()
    return action, value, action_dist.log_prob(action), action_dist.entropy() 

  def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
    action_logits, value = self.forward(obs)
    action_dist = Categorical(logits=action_logits)
    return action_dist.log_prob(actions), action_dist.entropy(), value

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=2.5e-4, eps=1e-5)
    return optimizer


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
