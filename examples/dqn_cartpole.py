import gym
import numpy as np
import torch
from lightning_rl.models import DQN
import torch.nn as nn
import copy
import pytorch_lightning as pl


class Model(DQN):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    self.qnet = nn.Sequential(
        nn.Linear(self.observation_space.shape[0], 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, self.action_space.n))

    self.qnet_target = copy.deepcopy(self.qnet)
    self.eps = 1.0
    self.eps_init = 1.0
    self.eps_decay = 7500
    self.eps_final = 0.04
    self.save_hyperparameters()

  # This is for running the model, returns the Q values given our observation
  def forward(self, x):
    return self.qnet(x)

  # This is for running the target Q network
  def forward_target(self, x):
    return self.qnet_target(x)

  # This is for updating the target Q network
  def update_target(self):
    self.qnet_target.load_state_dict(self.qnet.state_dict())

  def on_step(self):
    super().on_step()
    k = max(self.eps_decay - self.total_step_count, 0) / self.eps_decay
    self.eps = self.eps_final + k * (self.eps_init - self.eps_final)

  # This is for inference and evaluation of our model, returns the action
  def predict(self, x, deterministic=True):
    out = self.qnet(x)
    if deterministic:
      out = torch.max(out, dim=1)[1]
    else:
      eps = torch.rand_like(out[:, 0])
      eps = (eps < self.eps).float()
      out = eps * torch.rand_like(out).max(dim=1)[1] +\
          (1 - eps) * out.max(dim=1)[1]
    return out.long().cpu().numpy()

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
    return optimizer
  
if __name__ == '__main__':
  model = Model(env='CartPole-v1')
  
  trainer = pl.Trainer(max_epochs=30, gradient_clip_val=0.5, accelerator='gpu')
  trainer.fit(model)