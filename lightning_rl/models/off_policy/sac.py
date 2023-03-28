import os
from typing import Optional, Tuple
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
# from lightning_rl.modules.encoder import *


class SAC(nn.Module):
  def __init__(self,
               actor,
               critic,
               critic_target,
               device,
               action_shape,
               tau: float = 0.005,
               gamma: float = 0.99,
               init_temperature: float = 0.1,
               actor_lr: float = 1e-3,
               critic_lr: float = 1e-3,
               alpha_lr: float = 1e-4,
               ) -> None:
    super().__init__()
    self.actor = actor
    self.critic = critic
    self.critic_target = critic_target
    self.device = device
    self.tau = tau
    self.gamma = gamma

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
    return self.log_alpha.exp()

  def act(self, obs: np.ndarray, deterministic: bool = False):
    with torch.no_grad():
      obs = torch.FloatTensor(obs).to(self.device)
      obs = obs.unsqueeze(0)
      action, _, mean = self.actor(obs)
      if deterministic:
        action = mean
      return action.cpu().numpy().flatten()

  def update_critic(self,
                    obs: torch.Tensor,
                    actions: torch.Tensor,
                    rewards: torch.Tensor,
                    next_obs: torch.Tensor,
                    dones: torch.Tensor):
    with torch.no_grad():
      next_action, next_logprobs, _ = self.actor(next_obs)
      target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
      min_q_next_target = torch.min(
          target_Q1, target_Q2) - self.alpha * next_logprobs
      next_q_value = rewards.flatten() + (1 - dones.flatten()) * \
          self.gamma * min_q_next_target.view(-1)

    # TODO: Add a flag to detach encoder here
    current_Q1, current_Q2 = self.critic(obs, actions)
    Q1_loss = F.mse_loss(current_Q1, next_q_value)
    Q2_loss = F.mse_loss(current_Q2, next_q_value)
    critic_loss = Q1_loss + Q2_loss

    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()

    info = {
        "losses/Q1_loss": Q1_loss.item(),
        "losses/Q2_loss": Q2_loss.item(),
        "train/Q1_values": current_Q1.mean().item(),
        "train/Q2_values": current_Q2.mean().item(),
    }
    return info

  def update_actor(self, obs: torch.Tensor):
    # TODO: Detach encoders here if they exist (for both actor and critic)
    # Encoder should only be updated in critic update step
    actions, logprobs, _ = self.actor(obs)
    actor_Q1, actor_Q2 = self.critic(obs, actions)
    actor_Q = torch.min(actor_Q1, actor_Q2)
    actor_loss = (self.alpha.detach() * logprobs - actor_Q).mean()

    self.actor_optimizer.zero_grad()
    actor_loss.backward()
    self.actor_optimizer.step()

    info = {"losses/actor_loss": actor_loss.item()}
    return info

  def update_alpha(self, logprobs: torch.Tensor):
    alpha_loss = (self.alpha*(-logprobs - self.target_entropy).detach()).mean()

    self.log_alpha_optimizer.zero_grad()
    alpha_loss.backward()
    self.log_alpha_optimizer.step()

    info = {
        "losses/alpha_loss": alpha_loss.item(),
        "train/alpha": self.alpha.item()
    }
    return info

  def soft_target_update(self):
    for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
      target_param.data.copy_(self.tau * param.data +
                              (1 - self.tau) * target_param.data)

  def save_model(self, path: str, filename: str):
    torch.save(self.state_dict(), os.path.join(path, f'{filename}.pt'))


class Actor(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(4, 256)

  def forward(self, x):
    action = torch.rand((1,))
    logprob = torch.rand((1,))
    mean = torch.rand((1,))
    return action, logprob, mean


class Critic(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(4, 256)

  def forward(self, x, action):
    q1 = torch.rand((1,))
    q2 = torch.rand((1,))
    return q1, q2


if __name__ == '__main__':
  actor = Actor()
  critic = Critic()
  critic_target = Critic()
  model = SAC(actor=actor,
              critic=critic,
              critic_target=critic_target,
              device=torch.device('cpu'),
              action_shape=(1,))
  # model.update_critic(obs=torch.rand((1, 4)), actions=torch.rand((1, 1)), rewards=torch.rand(
  #     (1, 1)), next_obs=torch.rand((1, 4)), dones=torch.rand((1, 1)))
  model.update_actor(obs=torch.rand((1, 4)))
# class SAC():
#   def __init__(self,
#                actor: nn.Module,
#                q1: nn.Module,
#                q2: nn.Module,
#                q1_target: nn.Module,
#                q2_target: nn.Module,
#                actor_optimizer: torch.optim.Optimizer,
#                q_optimizer: torch.optim.Optimizer,
#                gamma: float = 0.99,
#                tau: float = 0.005,
#                ) -> None:
#     self.actor = actor
#     self.actor_optimizer = actor_optimizer

#     self.q1 = q1
#     self.q2 = q2
#     self.q1_target = q1_target
#     self.q2_target = q2_target
#     self.q_optimizer = q_optimizer
#     self.gamma = gamma
#     self.tau = tau

#   def train_critic(self,
#                    obs: torch.Tensor,
#                    next_obs: torch.Tensor,
#                    actions: torch.Tensor,
#                    next_actions: torch.Tensor,
#                    next_logprobs: torch.Tensor,
#                    rewards: torch.Tensor,
#                    dones: torch.Tensor,
#                    alpha: float
#                    ) -> dict:
#     with torch.no_grad():
#       q1_next_target = self.q1_target(next_obs, next_actions)
#       q2_next_target = self.q2_target(next_obs, next_actions)
#       min_q_next_target = torch.min(
#           q1_next_target, q2_next_target) - alpha * next_logprobs
#       next_q_value = rewards.flatten() + (1 - dones.flatten()) * \
#           self.gamma * min_q_next_target.view(-1)

#     q1_action_values = self.q1(obs, actions).view(-1)
#     q2_action_values = self.q2(obs, actions).view(-1)
#     q1_loss = F.mse_loss(q1_action_values, next_q_value)
#     q2_loss = F.mse_loss(q2_action_values, next_q_value)
#     q_loss = q1_loss + q2_loss

#     self.q_optimizer.zero_grad()
#     q_loss.backward()
#     self.q_optimizer.step()

#     info = {
#         "losses/q1_loss": q1_loss.item(),
#         "losses/q2_loss": q2_loss.item(),
#         "losses/q1_values": q1_action_values.mean().item(),
#         "losses/q2_values": q2_action_values.mean().item(),
#     }
#     return info

#   def train_actor(self,
#                   obs: torch.Tensor,
#                   actions: torch.Tensor,
#                   logprobs: torch.Tensor,
#                   alpha: float) -> dict:
#     q1 = self.q1(obs, actions)
#     q2 = self.q2(obs, actions)
#     min_q = torch.min(q1, q2).view(-1)
#     actor_loss = (alpha * logprobs - min_q).mean()

#     self.actor_optimizer.zero_grad()
#     actor_loss.backward()
#     self.actor_optimizer.step()

#     info = {
#         "losses/actor_loss": actor_loss.item(),
#     }

#     return info

#   def update_target_networks(self, tau: Optional[float] = None):
#     if tau is None:
#       tau = self.tau

#     for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
#       target_param.data.copy_(tau * param.data +
#                               (1 - tau) * target_param.data)
#     for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
#       target_param.data.copy_(tau * param.data +
#                               (1 - tau) * target_param.data)


# class VisualSAC(nn.Module):
#   def __init__(self, obs_shape, action_shape, hidden_dim, encoder_feature_dim, logstd_min, logstd_max, device):
#     super().__init__()

#     shared_cnn = NatureCNN(*obs_shape)

#     actor_proj = NormedProjection(shared_cnn.out_dim, encoder_feature_dim)
#     actor_encoder = Encoder(cnn=shared_cnn,
#                             projection=actor_proj)

#     critic_proj = NormedProjection(shared_cnn.out_dim, encoder_feature_dim)
#     critic_encoder = Encoder(cnn=shared_cnn,
#                              projection=critic_proj)

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
  log_prob = normal.log_prob(z) - torch.log(1 - squashed_action.pow(2) + 1e-6)
  log_prob = log_prob.sum(1, keepdim=True)
  # Recompute the mean as well, which is useful if you want a deterministic policy after training
  mean = torch.tanh(mean) * action_scale + action_bias
  return action, log_prob, mean
