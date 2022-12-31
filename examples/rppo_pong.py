from typing import Tuple
import gym
import numpy as np

import torch
import pytorch_lightning as pl
from torch import nn
from lightning_rl.common.buffers import RolloutSample
from lightning_rl.models.on_policy_models import PPO
from torch import distributions
from torch.distributions.categorical import Categorical

from lightning_rl.models.on_policy_models.recurrent_ppo import RecurrentPPO


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
  torch.nn.init.orthogonal_(layer.weight, std)
  torch.nn.init.constant_(layer.bias, bias_const)
  return layer


def make_env(env_id, seed, idx):

  def thunk():
    env = gym.make(env_id)
    env = gym.wrappers.AtariPreprocessing(
        env, screen_size=84, grayscale_obs=True, grayscale_newaxis=False)
    env = gym.wrappers.FrameStack(env, num_stack=1)

    return env
  return thunk


class AtariPPO(RecurrentPPO):
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
    # TODO: Refactor this to be batch first
    self.lstm = nn.LSTM(512, 128)
    for name, param in self.lstm.named_parameters():
      if "bias" in name:
        nn.init.constant_(param, 0)
      elif "weight" in name:
        nn.init.orthogonal_(param, 1.0)
    self.actor = layer_init(nn.Linear(128, self.action_space.n), std=0.01)
    self.critic = layer_init(nn.Linear(128, 1), std=1)

    self.save_hyperparameters()

  def get_states(self,
                 x: torch.Tensor,
                 hidden_state: Tuple[torch.Tensor],
                 done: torch.Tensor):
    hidden = self.feature_net(x / 255.0)

    if hidden_state is None:
      hidden_state = (
          torch.zeros(self.lstm.num_layers, self.n_envs,
                      self.lstm.hidden_size).to(self.device),
          torch.zeros(self.lstm.num_layers, self.n_envs,
                      self.lstm.hidden_size).to(self.device),
      )
    # Also need to initialize these on the first episode. I'm not a big fan of doing this here, but I can't think of anywhere else to put it.
    if self._last_hidden_state is None:
      self._last_hidden_state = hidden_state
    if self.initial_hidden_state is None:
      self.initial_hidden_state = (
          self._last_hidden_state[0].clone(),
          self._last_hidden_state[1].clone())

    # LSTM logic
    batch_size = hidden_state[0].shape[1]
    hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
    done = done.reshape((-1, batch_size)).to(self.device)
    new_hidden = []
    for h, d in zip(hidden, done):
      h, hidden_state = self.lstm(
          h.unsqueeze(0),
          (
              (1.0 - d).view(1, -1, 1) * hidden_state[0],
              (1.0 - d).view(1, -1, 1) * hidden_state[1],
          )
      )
      new_hidden += [h]
    new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
    return new_hidden, hidden_state

  def forward(self,
              x: torch.Tensor,
              hidden_state: torch.Tensor,
              done: torch.Tensor):
    hidden, hidden_state = self.get_states(x, hidden_state, done)
    action_logits = self.actor(hidden)
    value = self.critic(hidden).flatten()
    return action_logits, value, hidden_state

  def act(self,
          observation: torch.Tensor,
          hidden_state: torch.Tensor,
          done: torch.Tensor):
    action_logits, values, hidden_state = self.forward(
        observation, hidden_state, done)
    action_probs = Categorical(logits=action_logits)
    action = action_probs.sample()
    return action, values, action_probs.log_prob(action), action_probs.entropy(), hidden_state

  def evaluate_actions(self,
                       observations: torch.Tensor,
                       actions: torch.Tensor,
                       hidden_states: torch.Tensor,
                       dones: torch.Tensor,
                       ):
    action_logits, value, hidden_states = self.forward(
        observations, hidden_states, dones)
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
  env = gym.vector.AsyncVectorEnv(
      [make_env(env_id, seed + i, i) for i in range(n_env)])
  env = gym.wrappers.RecordEpisodeStatistics(env=env, deque_size=20)

  ppo = AtariPPO(env=env,
                 n_rollouts_per_epoch=5,
                 n_steps_per_rollout=128,
                 n_gradient_steps=4,
                 n_minibatch=4,
                 gamma=0.99,
                 gae_lambda=0.95,
                 value_coef=0.5,
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
