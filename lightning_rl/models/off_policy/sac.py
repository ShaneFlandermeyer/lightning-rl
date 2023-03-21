from typing import Optional, Tuple
import torch.nn as nn
import torch.nn.functional as F
import torch


class SAC():
  def __init__(self,
               actor: nn.Module,
               q1: nn.Module,
               q2: nn.Module,
               q1_target: nn.Module,
               q2_target: nn.Module,
               actor_optimizer: torch.optim.Optimizer,
               q_optimizer: torch.optim.Optimizer,
               gamma: float = 0.99,
               tau: float = 0.005,
               ) -> None:
    self.actor = actor
    self.actor_optimizer = actor_optimizer

    self.q1 = q1
    self.q2 = q2
    self.q1_target = q1_target
    self.q2_target = q2_target
    self.q_optimizer = q_optimizer
    self.gamma = gamma
    self.tau = tau

  def train_critic(self,
                   obs: torch.Tensor,
                   next_obs: torch.Tensor,
                   actions: torch.Tensor,
                   next_actions: torch.Tensor,
                   next_logprobs: torch.Tensor,
                   rewards: torch.Tensor,
                   dones: torch.Tensor,
                   alpha: float
                   ) -> dict:
    with torch.no_grad():
      q1_next_target = self.q1_target(next_obs, next_actions)
      q2_next_target = self.q2_target(next_obs, next_actions)
      min_q_next_target = torch.min(
          q1_next_target, q2_next_target) - alpha * next_logprobs
      next_q_value = rewards.flatten() + (1 - dones.flatten()) * \
          self.gamma * min_q_next_target.view(-1)

    q1_action_values = self.q1(obs, actions).view(-1)
    q2_action_values = self.q2(obs, actions).view(-1)
    q1_loss = F.mse_loss(q1_action_values, next_q_value)
    q2_loss = F.mse_loss(q2_action_values, next_q_value)
    q_loss = q1_loss + q2_loss

    self.q_optimizer.zero_grad()
    q_loss.backward()
    self.q_optimizer.step()

    info = {
        "losses/q1_loss": q1_loss.item(),
        "losses/q2_loss": q2_loss.item(),
        "losses/q1_values": q1_action_values.mean().item(),
        "losses/q2_values": q2_action_values.mean().item(),
    }
    return info

  def train_actor(self,
                  obs: torch.Tensor,
                  actions: torch.Tensor,
                  logprobs: torch.Tensor,
                  alpha: float) -> dict:
    q1 = self.q1(obs, actions)
    q2 = self.q2(obs, actions)
    min_q = torch.min(q1, q2).view(-1)
    actor_loss = (alpha * logprobs - min_q).mean()

    self.actor_optimizer.zero_grad()
    actor_loss.backward()
    self.actor_optimizer.step()

    info = {
        "losses/actor_loss": actor_loss.item(),
    }

    return info

  def update_target_networks(self, tau: Optional[float] = None):
    if tau is None:
      tau = self.tau

    for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
      target_param.data.copy_(tau * param.data +
                              (1 - tau) * target_param.data)
    for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
      target_param.data.copy_(tau * param.data +
                              (1 - tau) * target_param.data)


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
