from typing import Optional, Union
import gym
import torch
from lightning_rl.common import OffPolicyModel
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from gym import spaces
import torch.nn.functional as F

from lightning_rl.common.buffers import ExperienceBatch


class DQN(OffPolicyModel):
  def __init__(
      self,
      env: Union[gym.Env, VecEnv, str],
      batch_size: int = 64,
      replay_buffer_size: int = int(1e6),
      n_warmup_steps: int = 1000,
      n_rollouts_per_epoch: int = 8,
      train_freq: int = 256,
      n_gradient_steps: int = 128,
      target_update_interval: int = 10,
      gamma: float = 0.99,
      seed: Optional[int] = None,
  ) -> None:

    super().__init__(
        env=env,
        batch_size=batch_size,
        replay_buffer_size=replay_buffer_size,
        n_warmup_steps=n_warmup_steps,
        n_rollouts_per_epoch=n_rollouts_per_epoch,
        train_freq=train_freq,
        n_gradient_steps=n_gradient_steps,
        gamma=gamma,
        seed=seed,
    )

    assert isinstance(
        self.action_space, spaces.Discrete), "DQN only supports environments with Discrete action spaces"

    self.target_update_interval = target_update_interval
    self.target_update_counter = 0

  def on_step(self):
    """
    Perform these actions on every environment step
    """
    self.target_update_counter += 1
    if self.target_update_counter == self.target_update_interval:
      self.update_target()
      self.target_update_counter = 0

  def reset(self):
    """
    Reset the environment and the counter that tracks the target network updates
    """
    super().reset()
    self.target_update_counter = 0

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Q-network forward pass

    Override this function with your own

    Parameters
    ----------
    x : torch.Tensor

    Returns
    -------
    torch.Tensor
        Q-values for each action
    """
    raise NotImplementedError

  def forward_target(self, x: torch.Tensor) -> torch.Tensor:
    """
    Q-network forward pass

    Override this function with your own

    Parameters
    ----------
    x : torch.Tensor

    Returns
    -------
    torch.Tensor
        Q-values for each action
    """
    raise NotImplementedError

  def update_target(self) -> None:
    """
    Function to update the target Q network periodically.
    Override this function with your own.
    """
    raise NotImplementedError

  def training_step(self, batch: ExperienceBatch, batch_idx: int) -> torch.Tensor:
    """
    Compute the loss for each example in the batch using a 1-step TD update target

    Parameters
    ----------
    batch : ExperienceBatch
        Batch of experiences
    batch_idx : int
        Batch index
    """
    # Not enough experience to compute the loss
    if self.total_step_count < self.n_warmup_steps:
      return

    # Compute the TD target
    with torch.no_grad():
      target_q = self.forward_target(batch.next_states)
      target_q = torch.max(target_q, dim=1, keepdims=True)[0]
      target_q = batch.rewards + self.gamma * target_q
      target_q[batch.dones] = 0
      
    # Compute the Q-values estimated by the network
    current_q = self.forward(batch.states)
    current_q = torch.gather(current_q, dim=1, index=batch.actions.long())
    
    loss = F.smooth_l1_loss(current_q, target_q)
    return loss
    
