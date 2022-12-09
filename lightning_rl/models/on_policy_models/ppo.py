import torch
from lightning_rl.common.buffers import RolloutBatch
from lightning_rl.common.utils import explained_variance
from lightning_rl.models.on_policy_models import OnPolicyModel
import gym
from typing import Tuple, Union, Optional
from torch import distributions
import torch.nn.functional as F


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
    dist, values = self.forward(batch.observations)
    log_probs = dist.log_prob(batch.actions)
    values = values.flatten()

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
    value_loss = F.mse_loss(batch.returns, values_pred)

    # Use entropy to discourage collapse into a determinsitic policy
    entropy_loss = -dist.entropy().mean()
    # Total loss is the sum of all losses
    loss = policy_loss + self.value_coef * \
        value_loss + self.entropy_coef * entropy_loss

    with torch.no_grad():
      clip_fraction = torch.mean(
          (torch.abs(ratio - 1) > self.policy_clip_range).float())
      approx_kl = torch.mean(batch.log_probs - log_probs)
      explained_var = explained_variance(batch.values, batch.returns)

    # TODO: Implement early stopping logic when the approximate KL exceeds the target KL parameter

    self.log_dict({
        'train/total_loss': loss,
        'train/policy_loss': policy_loss,
        'train/value_loss': value_loss,
        'train/entropy_loss': entropy_loss,
        'train/clip_fraction': clip_fraction,
        'train/approx_kl': approx_kl,
        'train/explained_variance': explained_var,
    },
        prog_bar=False, logger=True
    )
    return loss

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
