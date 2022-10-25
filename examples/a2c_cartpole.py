import gym
from lightning_rl.models.a2c import A2C
import torch.nn as nn
from torch import distributions
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
import torch
import pytorch_lightning as pl

class Model(A2C):
    def __init__(self, **kwargs):
        # **kwargs will pass our arguments on to A2C
        super(Model, self).__init__(**kwargs)

        self.actor = nn.Sequential(
            nn.Linear(self.observation_space.shape[0], 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, self.action_space.n),
            nn.Softmax(dim=1))

        self.critic = nn.Sequential(
            nn.Linear(self.observation_space.shape[0], 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1))

        self.save_hyperparameters()

    # This is for training the model
    # Returns the distribution and the corresponding value
    def forward(self, x):
        out = self.actor(x)
        dist = distributions.Categorical(probs=out)
        return dist, self.critic(x).flatten()

    # This is for inference and evaluation of our model, returns the action
    def predict(self, x, deterministic=True):
        out = self.actor(x)
        if deterministic:
            out = torch.max(out, dim=1)[1]
        else:
            out = distributions.Categorical(probs=out).sample()
        return out.cpu().numpy()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        return optimizer
      
if __name__ == '__main__':
    env = make_vec_env('CartPole-v1', 
                       n_envs=8, 
                       vec_env_cls=SubprocVecEnv)
    model = Model(env=env,
                  n_steps_per_update=10,)

    trainer = pl.Trainer(max_epochs=20, gradient_clip_val=0.5)
    trainer.fit(model)