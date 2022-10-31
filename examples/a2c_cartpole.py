from copy import deepcopy

import gym
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from lightning_rl.common.layers import NoisyLinear
from lightning_rl.models.on_policy_models import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from torch import distributions


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
            nn.Linear(64, 1),
            )

        self.save_hyperparameters()

    # This is for training the model
    # Returns the distribution and the corresponding value
    def forward(self, x):
        out = self.actor(x)
        dist = distributions.Categorical(probs=out)
        return dist, self.critic(x).flatten()

    # This is for inference and evaluation of our model, returns the action
    def predict(self, x, deterministic=False, **kwargs):
        out = self.actor(x)
        if deterministic:
            out = torch.max(out, dim=1)[1]
        else:
            out = distributions.Categorical(probs=out).sample()
        return out.cpu().numpy()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=7e-4, eps=1e-5)
        return optimizer
    
      
if __name__ == '__main__':
    env_id = "CartPole-v1"
    num_cpu = 8  # Number of processes to use
    
    vec_env = make_vec_env(env_id, n_envs=num_cpu, vec_env_cls=DummyVecEnv)
    eval_env = gym.make(env_id)
    for i in range(100):
        print("Iteration: ", i)
        vec_env.reset()
        eval_env.reset()    
        
        # TODO: For some reason, the validation rewards after each epoch only seem to update if n_steps_per_rollout > 10 (ish)
        model = Model(env=vec_env,
                    eval_env=eval_env,
                    n_rollouts_per_epoch=100,
                    n_steps_per_rollout=16,
                    n_eval_episodes=5,
                    gae_lambda=0.98,
                    seed=np.random.randint(0, 100000),
                    normalize_advantage=False,
                    )
        
        trainer = pl.Trainer(max_epochs=5, 
                             gradient_clip_val=1,
                            accelerator='gpu',
                            enable_progress_bar=True,
                            reload_dataloaders_every_n_epochs=1,
                            devices=1)
        trainer.fit(model)
    
    