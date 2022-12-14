from copy import deepcopy

import gym
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from lightning_rl.common.layers import NoisyLinear
from lightning_rl.models.on_policy_models import A2C
from torch import distributions
from lightning_rl.common.callbacks import EvalCallback


class Model(A2C):
  def __init__(self, **kwargs):
    # **kwargs will pass our arguments on to A2C
    super(Model, self).__init__(**kwargs)

    self.actor = nn.Sequential(
        nn.Linear(self.observation_space.shape[0], 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        NoisyLinear(64, self.action_space.n),
        nn.Softmax(dim=1))

    self.critic = nn.Sequential(
        nn.Linear(self.observation_space.shape[0], 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        NoisyLinear(64, 1),
    )

    self.save_hyperparameters()

  # This is for training the model
  # Returns the distribution and the corresponding value
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
    optimizer = torch.optim.Adam(self.parameters(), lr=7e-4, eps=1e-5)
    return optimizer


if __name__ == '__main__':
  env_id = "CartPole-v1"
  n_env = 8  # Number of processes to use

  # vec_env = make_vec_env(env_id, n_envs=num_cpu, vec_env_cls=DummyVecEnv)
  vec_env = gym.vector.AsyncVectorEnv([lambda: gym.make(env_id)] * n_env)
  eval_env = gym.make(env_id)

  vec_env.reset()
  eval_env.reset()

  model = Model(env=vec_env,
                n_rollouts_per_epoch=100,
                n_steps_per_rollout=16,
                gae_lambda=0.98,
                seed=np.random.randint(0, 100000),
                normalize_advantage=False,
                )
  trainer = pl.Trainer(max_epochs=10,
                       gradient_clip_val=0.5,
                       accelerator='gpu',
                       enable_progress_bar=True,
                       reload_dataloaders_every_n_epochs=1,
                       devices=1,
                       callbacks=[EvalCallback(eval_env)],)
  trainer.fit(model)
