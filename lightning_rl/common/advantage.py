from typing import Tuple
import torch


@torch.no_grad()
def estimate_advantage(rewards: torch.Tensor,
                       values: torch.Tensor,
                       dones: torch.Tensor,
                       last_value: torch.Tensor,
                       last_done: torch.Tensor,
                       n_steps: int,
                       gamma: float,
                       gae_lambda: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
  """
  Estimate the advantage from an n-step rollout using generalized advantage estimation (GAE)

  Parameters
  ----------
  rewards : torch.Tensor
      _description_
  dones : torch.Tensor
      _description_
  values : torch.Tensor
      _description_
  last_done : torch.Tensor
      _description_
  last_value : torch.Tensor
      _description_
  n_steps : int
      _description_
  gamma : float
      _description_
  lam : float
      _description_

  Returns
  -------
  _type_
      _description_
  """
  advantages = torch.zeros_like(rewards)
  last_advantage_estimate = 0
  for t in reversed(range(n_steps)):
    if t == n_steps - 1:
      next_non_terminal = 1.0 - last_done
      next_values = last_value
    else:
      next_non_terminal = 1.0 - dones[t + 1]
      next_values = values[t + 1]
    delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
    advantages[t] = last_advantage_estimate = delta + gamma * \
        gae_lambda * next_non_terminal * last_advantage_estimate
  returns = advantages + values

  return advantages, returns


if __name__ == '__main__':
  n_steps = 512
  gamma = 0.99
  lam = 0.95
  torch.manual_seed(0)
  rewards = torch.rand(n_steps)
  dones = torch.rand(n_steps)
  values = torch.ones(n_steps)
  last_done = torch.rand(1)
  last_value = torch.rand(1)
  a, r = estimate_advantage(rewards, dones, values, last_done,
                            last_value, n_steps, gamma, lam)
  print(a)
