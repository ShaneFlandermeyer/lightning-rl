import gym
import numpy as np
import torch
from lightning_rl.common.layers import NoisyLinear
from lightning_rl.models import DQN
import torch.nn as nn
import copy
import pytorch_lightning as pl

# NOTE: This example currently does not work!!!
# I will update it when I get the chance to make the off-policy dataset class.


class Model(DQN):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    self.qnet = nn.Sequential(
        nn.Linear(self.observation_space.shape[0], 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        NoisyLinear(256, self.action_space.n))

    self.qnet_target = copy.deepcopy(self.qnet)
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

  # This is for inference and evaluation of our model, returns the action
  def predict(self, x, deterministic=True):
    out = self.qnet(x)
    out = torch.max(out, dim=1)[1]
    return out.long().cpu().numpy()

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
    return optimizer


if __name__ == '__main__':
  model = Model(env='CartPole-v1')

  trainer = pl.Trainer(max_epochs=10,
                       gradient_clip_val=0.5,
                       accelerator='gpu',
                       devices=1,
                       strategy='ddp')
  trainer.fit(model)
