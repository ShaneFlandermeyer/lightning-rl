import os
from typing import Optional, Tuple
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch


class SAC(nn.Module):
  def __init__(self,
               actor: nn.Module,
               critic: nn.Module,
               critic_target: nn.Module,
               device: torch.device,
               obs_shape: Tuple[int, ...],
               action_shape: Tuple[int, ...],
               gamma: float = 0.99,
               tau: float = 0.005,
               init_temperature: float = 0,
               actor_lr: float = 1e-3,
               critic_lr: float = 3e-4,
               alpha_lr: float = 3e-4,
               ) -> None:
    super().__init__()
    self.actor = actor
    self.critic = critic
    self.critic_target = critic_target
    self.device = device
    self.action_shape = action_shape
    self.obs_shape = obs_shape
    self.gamma = gamma
    self.tau = tau

    self.log_alpha = torch.tensor(
        np.log(init_temperature), requires_grad=True, device=device)
    self.target_entropy = -np.prod(action_shape).item()

    # Optimizers
    self.actor_optimizer = torch.optim.Adam(
        self.actor.parameters(), lr=actor_lr)

    self.critic_optimizer = torch.optim.Adam(
        self.critic.parameters(), lr=critic_lr)
    self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

    self.train()
    self.critic_target.train()

  def train(self, training: bool = True):
    self.training = training
    self.actor.train(training)
    self.critic.train(training)

  @property
  def alpha(self):
    return self.log_alpha.exp().item()

  def act(self, obs: torch.Tensor, deterministic: bool = False):
    action, logprobs, mean = self.actor(obs)
    if deterministic:
      action = mean
    return action, logprobs

  def update_critic(self,
                    obs: torch.Tensor,
                    actions: torch.Tensor,
                    rewards: torch.Tensor,
                    next_obs: torch.Tensor,
                    dones: torch.Tensor):
    with torch.no_grad():
      next_action, next_logprobs = self.act(next_obs, deterministic=False)
      target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
      min_q_next_target = torch.min(
          target_Q1, target_Q2) - self.alpha * next_logprobs
      next_q_value = rewards.flatten() + (1 - dones.flatten()) * \
          self.gamma * min_q_next_target.view(-1)

    # TODO: Add a flag to detach encoder here
    current_Q1, current_Q2 = self.critic(obs, actions)
    Q1_loss = F.mse_loss(current_Q1.view(-1), next_q_value)
    Q2_loss = F.mse_loss(current_Q2.view(-1), next_q_value)
    critic_loss = Q1_loss + Q2_loss

    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()

    info = {
        "loss/Q1_loss": Q1_loss.item(),
        "loss/Q2_loss": Q2_loss.item(),
        "train/Q1_values": current_Q1.mean().item(),
        "train/Q2_values": current_Q2.mean().item(),
    }
    return info

  def update_actor(self, obs: torch.Tensor):
    # TODO: Detach encoders here if they exist (for both actor and critic)
    # Encoder should only be updated in critic update step
    actions, logprobs = self.act(obs, deterministic=False)
    Q1, Q2 = self.critic(obs, actions)
    Q = torch.min(Q1, Q2).view(-1)
    actor_loss = (self.alpha * logprobs - Q).mean()

    self.actor_optimizer.zero_grad()
    actor_loss.backward()
    self.actor_optimizer.step()

    info = {"loss/actor_loss": actor_loss.item()}
    return info

  def update_alpha(self, obs: torch.Tensor):
    with torch.no_grad():
      _, logprobs = self.act(obs, deterministic=False)
    alpha_loss = (-self.log_alpha * (logprobs + self.target_entropy)).mean()

    self.log_alpha_optimizer.zero_grad()
    alpha_loss.backward()
    self.log_alpha_optimizer.step()

    info = {
        "loss/alpha_loss": alpha_loss.item(),
        "train/alpha": self.alpha,
    }
    return info

  def soft_target_update(self):
    for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
      target_param.data.copy_(self.tau * param.data +
                              (1 - self.tau) * target_param.data)

  def save_model(self, path: str, filename: str):
    torch.save(self.state_dict(), os.path.join(path, f'{filename}.pt'))


def squashed_gaussian_action(mean: torch.Tensor,
                             std: torch.Tensor,
                             action_scale: float,
                             action_bias: float
                             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """
  Sample an action from a Gaussian distribution, then squash it to the desired range.

  Parameters
  ----------
  mean : torch.Tensor
      Mean vector
  std : torch.Tensor
      Standard deviation vector (covariance diagonals)
  action_scale : float
      Span of the action space
  action_bias : float
      Minimum value of the action space

  Returns
  -------
  Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
      A tuple containing:
        - The sampled action
        - The log probability of the action
        - The squashed and scaled mean of the Gaussian distribution
  """
  # Sample an action from the Gaussian distribution, then squash/scale it
  normal = torch.distributions.Normal(mean, std)
  z = normal.rsample()
  squashed_action = torch.tanh(z)
  action = squashed_action * action_scale + action_bias
  # Compute the log probabilities using the change-of-variables formula described here: https://arxiv.org/pdf/1812.05905.pdf
  log_prob = normal.log_prob(z)
  log_prob -= torch.log(action_scale * (1 - squashed_action.pow(2)) + 1e-6)
  log_prob = log_prob.sum(1, keepdim=True)
  # Recompute the mean as well, which is useful if you want a deterministic policy after training
  mean = torch.tanh(mean) * action_scale + action_bias
  return action, log_prob, mean