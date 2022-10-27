from copy import deepcopy
import gym
import numpy as np
from lightning_rl.models.on_policy_models import A2C
import torch.nn as nn
from torch import distributions
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
import torch
import pytorch_lightning as pl
from stable_baselines3.common.evaluation import evaluate_policy


class Model(A2C):
    def __init__(self, **kwargs):
        # **kwargs will pass our arguments on to A2C
        super(Model, self).__init__(**kwargs)

        self.actor = nn.Sequential(
            nn.Linear(self.observation_space.shape[0], 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, self.action_space.n),
            nn.Softmax(dim=1))

        self.critic = nn.Sequential(
            nn.Linear(self.observation_space.shape[0], 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1))

        self.save_hyperparameters()

    # This is for training the model
    # Returns the distribution and the corresponding value
    def forward(self, x):
        out = self.actor(x)
        dist = distributions.Categorical(probs=out)
        return dist, self.critic(x).flatten()

    # This is for inference and evaluation of our model, returns the action
    def predict(self, x, deterministic=True, **kwargs):
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
                       n_envs=1, 
                       vec_env_cls=SubprocVecEnv)
    eval_env = gym.make('CartPole-v1')
    model = Model(env=env,
                  n_rollouts_per_epoch=100,
                  n_steps_per_rollout=8,
                  entropy_coef=0.0)


    import time

    start = time.time()
    trainer = pl.Trainer(max_epochs=15, 
                         gradient_clip_val=0.5,
                         accelerator='gpu',
                         devices=1,
                         strategy='ddp',)
    trainer.fit(model)
        
    end = time.time()
    print(end - start)
    
    rewards = model.evaluate(eval_env, n_eval_episodes=20)
    print(np.mean(rewards))
