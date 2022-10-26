from typing import Optional, Tuple, Union

import gym
import torch
import torch.nn.functional as F
from lightning_rl.models.on_policy_models import OnPolicyModel
from stable_baselines3.common.vec_env import VecEnv
from torch import distributions

from lightning_rl.common.utils import explained_variance


class A2C(OnPolicyModel):
  """
    Advantage Actor Critic (A2C)

    Paper: https://arxiv.org/abs/1602.01783
    Code: This implementation borrows code from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    and Stable Baselines 3 (https://github.com/DLR-RM/stable-baselines3)

    Introduction to A2C: https://hackernoon.com/intuitive-rl-intro-to-advantage-actor-critic-a2c-4ff545978752

    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param buffer_length: (int) Length of the buffer and the number of steps to run for each environment per update
    :param n_rollouts_per_epoch: Number of rollouts to do per PyTorch Lightning epoch. This does not affect any training dynamic,
        just how often we evaluate the model since evaluation happens at the end of each Lightning epoch
    :param batch_size: Minibatch size for each gradient update
    :param n_epochs_per_rollout: Number of epochs to optimise the loss for
    :param gamma: (float) Discount factor
    :param gae_lambda: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param value_coef: Value function coefficient for the loss calculation
    :param entropy_coef: Entropy coefficient for the loss calculation
    :param seed: Seed for the pseudo random generators
    """

  def __init__(self,
               env: Union[gym.Env, VecEnv, str],
               n_steps_per_rollout: int = 10,
               n_rollouts_per_epoch: int = 100,
               gamma: float = 0.99,
               gae_lambda: float = 1.0,
               value_coef: float = 0.5,
               entropy_coef: float = 0.0,
               seed: Optional[int] = None):
    super().__init__(
        env=env,
        n_steps_per_rollout=n_steps_per_rollout,
        n_rollouts_per_epoch=n_rollouts_per_epoch,
        gamma=gamma,
        gae_lambda=gae_lambda,
        seed=seed
    )
    self.value_coef = value_coef
    self.entropy_coef = entropy_coef

  def forward(self,
              x: torch.Tensor) -> Tuple[distributions.Distribution, torch.Tensor]:
    """
    Run the actor and critic network on the input observations

    Parameters
    ----------
    x : torch.Tensor
        Input observations

    Returns
    -------
    Tuple[distributions.Distribution, torch.Tensor]
        The deterministic action of the actor
    """
    raise NotImplementedError

  def training_step(self, batch, batch_idx):
    """
    Update step for A2C.
    """
    dist, values = self(batch.observations)
    log_probs = dist.log_prob(batch.actions)
    values = values.flatten()

    advantages = batch.advantages.detach()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    policy_loss = -(advantages * log_probs).mean()
    value_loss = F.mse_loss(batch.returns.detach(), values)
    entropy_loss = -dist.entropy().mean()

    loss = policy_loss + self.value_coef * \
        value_loss + self.entropy_coef * entropy_loss

    with torch.no_grad():
      explained_var = explained_variance(values, batch.returns)
    self.log_dict({
        'train_loss': loss,
        'policy_loss': policy_loss,
        'value_loss': value_loss,
        'entropy_loss': entropy_loss,
        'explained_variance': explained_var},
        prog_bar=False, logger=True
    )

    return loss