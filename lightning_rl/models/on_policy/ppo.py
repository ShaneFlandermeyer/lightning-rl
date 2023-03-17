import time
import torch
from torch import Tensor
from lightning_rl.common.buffers import RolloutBatch
from lightning_rl.common.utils import explained_variance
from lightning_rl.models.on_policy import OnPolicyModel
import gym
from typing import Tuple, Union, Optional
from torch import distributions
import torch.nn.functional as F


def ppo_loss(
        # Batch data
        batch_advantages: Tensor,
        batch_log_probs: Tensor,
        batch_values: Tensor,
        batch_returns: Tensor,
        # Network outputs
        new_log_probs: Tensor,
        new_values: Tensor,
        entropy: Tensor,
        # Configuration parameters
        policy_clip_range: float = 0.2,
        value_clip_range: Optional[float] = None,
        value_coef: float = 1,
        entropy_coef: float = 0.0,
        normalize_advantage: bool = True,
):
  # Compute the policy loss
  if normalize_advantage:
    batch_advantages = (batch_advantages - batch_advantages.mean()) / \
        (batch_advantages.std() + 1e-8)

  ratio = torch.exp(new_log_probs - batch_log_probs)
  policy_loss_1 = batch_advantages * ratio
  policy_loss_2 = batch_advantages * torch.clamp(ratio,
                                                 1 - policy_clip_range,
                                                 1 + policy_clip_range)
  policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

  if value_clip_range is None:
    # Directly use the latest value network output to compute the loss
    value_loss = F.mse_loss(new_values, batch_returns)
  else:
    # Clip the difference between the old and new value functions
    values_clipped = batch_values + torch.clamp(new_values - batch_values,
                                                -value_clip_range,
                                                value_clip_range)
    # Select the maximum loss between the clipped and unclipped 
    value_loss_unclipped = (new_values - batch_returns)**2
    value_loss_clipped = (values_clipped - batch_returns)**2
    value_loss_max = torch.max(value_loss_unclipped, value_loss_clipped)
    value_loss = value_loss_max.mean()

  # Entropy regularizer
  if entropy is None:
    entropy_loss = -torch.mean(-new_log_probs)
  else:
    entropy_loss = -torch.mean(entropy)

  # Total loss is the sum of all losses
  loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

  # Compute metrics
  with torch.no_grad():
    clip_fraction = torch.mean(
        (torch.abs(ratio - 1) > policy_clip_range).float())
    approx_kl = torch.mean(batch_log_probs - new_log_probs)
    explained_var = explained_variance(batch_values, batch_returns)

  # Compute info dictionary
  info = {
      'loss/value_loss': value_loss.item(),
      'loss/policy_loss': policy_loss.item(),
      'loss/total_loss': loss.item(),
      'loss/entropy_loss': entropy_loss.item(),
      'metrics/approx_kl': approx_kl.item(),
      'metrics/explained_variance': explained_var.item(),
      'metrics/clip_fraction': clip_fraction.item(),
  }

  return loss, info


class PPO(OnPolicyModel):
  """
  Proximal policy optimization (PPO) algorithm

  Parameters
  ----------
  OnPolicyModel : _type_
      _description_
  """

  def __init__(self,
               env: Union[gym.Env, gym.vector.VectorEnv, str],
               n_steps_per_rollout: int = 10,
               n_rollouts_per_epoch: int = 100,
               n_gradient_steps: int = 10,
               batch_size: int = 64,
               gamma: float = 0.99,
               gae_lambda: float = 1.0,
               policy_clip_range: float = 0.2,
               value_clip_range: Optional[float] = None,
               target_kl: Optional[float] = None,
               value_coef: float = 0.5,
               entropy_coef: float = 0.0,
               normalize_advantage: bool = True,
               seed: Optional[int] = None,
               **kwargs,
               ) -> None:
    super().__init__(
        env=env,
        n_steps_per_rollout=n_steps_per_rollout,
        n_rollouts_per_epoch=n_rollouts_per_epoch,
        n_gradient_steps=n_gradient_steps,
        batch_size=batch_size,
        gamma=gamma,
        gae_lambda=gae_lambda,
        seed=seed,
        **kwargs,
    )
    self.policy_clip_range = policy_clip_range
    self.value_clip_range = value_clip_range
    self.target_kl = target_kl
    self.value_coef = value_coef
    self.entropy_coef = entropy_coef
    self.normalize_advantage = normalize_advantage

  def training_step(self, batch: RolloutBatch, batch_idx: int) -> float:
    """
    Perform the PPO update step

    Parameters
    ----------
    batch : RolloutBatch
        Minibatch from the current rollout
    batch_idx : int
        Batch index

    Returns
    -------
    float
        Total loss = policy loss + value loss + entropy_loss
    """
    log_probs, entropy, values = self.evaluate_actions(
        batch.observations, batch.actions)

    advantages = batch.advantages
    if self.normalize_advantage:
      advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Ratio between old and new policy. Should be one at the first iteration.
    ratio = torch.exp(log_probs - batch.log_probs)

    # Compute the clipped surrogate loss
    policy_loss_1 = advantages * ratio
    policy_loss_2 = advantages * torch.clamp(ratio,
                                             1 - self.policy_clip_range,
                                             1 + self.policy_clip_range)
    policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

    if self.value_clip_range is None:
      # Directly use the latest value network output to compute the loss
      values_pred = values
    else:
      # Clip the difference between the old and new value functions
      values_pred = batch.values + torch.clamp(values - batch.values,
                                               -self.value_clip_range,
                                               self.value_clip_range)
    # Value loss using the TD(gae_lambda) target
    value_loss = 0.5*F.mse_loss(values_pred, batch.returns)

    # Use entropy to discourage collapse into a determinsitic policy
    if entropy is None:
      entropy_loss = -torch.mean(-log_probs)
    else:
      entropy_loss = -torch.mean(entropy)
    # Total loss is the sum of all losses
    loss = policy_loss + self.value_coef * \
        value_loss + self.entropy_coef * entropy_loss

    with torch.no_grad():
      clip_fraction = torch.mean(
          (torch.abs(ratio - 1) > self.policy_clip_range).float())
      approx_kl = torch.mean(batch.log_probs - log_probs)
      explained_var = explained_variance(batch.values, batch.returns)

    if self.target_kl is not None and approx_kl > 1.5 * self.target_kl:
      self.continue_training = False

    self.log_dict({
        'train/total_loss': loss,
        'train/policy_loss': policy_loss,
        'train/value_loss': value_loss,
        'train/entropy_loss': entropy_loss,
        'train/clip_fraction': clip_fraction,
        'train/approx_kl': approx_kl,
        'train/explained_variance': explained_var,
        'train/FPS': self.total_step_count / (time.time() - self.start_time)
    },
        prog_bar=False, logger=True
    )
    return loss
