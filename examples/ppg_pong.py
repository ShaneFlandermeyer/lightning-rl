from typing import Tuple
import gym
import numpy as np

import torch
import pytorch_lightning as pl
from torch import nn
from lightning_rl.models.on_policy_models.ppg import PPG
from torch import distributions
from torch.distributions.categorical import Categorical
from lightning_rl.common.layer_init import ortho_init, norm_init
import torch.nn.functional as F


def make_env(env_id, seed, idx):

  def thunk():
    env = gym.make(env_id)
    env = gym.wrappers.AtariPreprocessing(
        env, screen_size=84, grayscale_obs=True, grayscale_newaxis=False)
    env = gym.wrappers.FrameStack(env, num_stack=1)

    return env
  return thunk

class AtariPPG(PPG):
  def __init__(self,
               env: gym.Env,
               **kwargs):
    # **kwargs will pass our arguments on to PPO
    super().__init__(env=env,
                     **kwargs)
    self.feature_net = nn.Sequential(
        ortho_init(nn.Conv2d(
            self.observation_space.shape[0], 32, kernel_size=8, stride=4)),
        nn.ReLU(),
        ortho_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
        nn.ReLU(),
        ortho_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
        nn.ReLU(),
        nn.Flatten(start_dim=1, end_dim=-1),
        ortho_init(nn.Linear(64*7*7, 512)),
        nn.ReLU(),
    )
    self.actor = ortho_init(nn.Linear(512, self.action_space.n), std=0.01)
    self.aux_critic = ortho_init(nn.Linear(512, 1), std=1)
    self.critic = ortho_init(nn.Linear(512, 1), std=1)

    self.save_hyperparameters()

  def forward(self, x: torch.Tensor):
    features = self.feature_net(x / 255.0)
    action_logits = self.actor(features)
    aux_value = self.aux_critic(features).flatten()
    value = self.critic(features).flatten()
    return action_logits, aux_value, value

  def act(self, x: torch.Tensor):
    action_logits, aux_value, value = self.forward(x)
    action_dist = Categorical(logits=action_logits)
    action = action_dist.sample()
    return action, value, action_dist.log_prob(action), action_dist.entropy(), action_dist, aux_value

  def logits_to_action_dist(self, logits: torch.Tensor) -> torch.distributions.Distribution:
    """
    Return the action distribution from the output logits. This is needed to compute the KL divergence term in the joint loss on page 3 of the PPG paper (Cobbe2020).

    Parameters
    ----------
    logits : torch.Tensor
        Logits from the output of the actor network

    Returns
    -------
    torch.distributions.Distribution
        Action distribution
    """
    action_dist = Categorical(logits=logits)
    return action_dist

  def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
    action_logits, _, value = self.forward(obs)
    action_dist = Categorical(logits=action_logits)
    return action_dist.log_prob(actions), action_dist.entropy(), value

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=5e-4, eps=1e-5)
    return optimizer


if __name__ == '__main__':
  # Unwrapped environments
  env_id = "PongNoFrameskip-v4"
  seed = np.random.randint(0, 2**32 - 1)

  # Vectorize the environment
  n_env = 16
  env = gym.vector.AsyncVectorEnv(
      [make_env(env_id, seed, i) for i in range(n_env)])
  env = gym.wrappers.RecordEpisodeStatistics(env=env, deque_size=20)

  ppg = AtariPPG(env=env,
                 n_rollouts_per_epoch=1,
                 n_steps_per_rollout=256,
                 shared_arch=True,
                 gamma=0.99,
                 gae_lambda=0.95,
                 value_coef=1,
                 entropy_coef=0.01,
                 seed=seed,
                 normalize_advantage=True,
                 policy_clip_range=0.1,
                 policy_minibatch_size=256,
                 # PPG parameters
                 aux_minibatch_size=16,
                 n_policy_steps=16,
                 n_policy_epochs=6,
                 n_value_epochs=1,
                 n_aux_epochs=6,
                 beta_clone=1.0,
                 )

  trainer = pl.Trainer(
      max_time="00:03:00:00",
      gradient_clip_val=0.5,
      accelerator='gpu',
      devices=1,
      # strategy='ddp',
  )

  trainer.fit(ppg)
