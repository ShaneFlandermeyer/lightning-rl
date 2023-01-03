from enum import Enum
from typing import Iterator, NamedTuple, Optional, Union

import gymnasium as gym
import numpy as np
import torch

from lightning_rl.common.buffers import RolloutBatch
from lightning_rl.models import on_policy_models
from lightning_rl.models.on_policy_models.ppo import PPO
import torch.nn.functional as F


class AuxiliaryBatch(NamedTuple):
  observations: torch.Tensor
  returns: torch.Tensor
  old_policy: torch.distributions.Distribution


class PPG(PPO):
  def __init__(self,
               env: Union[gym.Env, gym.vector.VectorEnv, str],
               seed: Optional[int] = None,
               n_rollouts_per_epoch: int = 100,
               n_steps_per_rollout: int = 256,
               shared_arch: bool = False,
               # PPO parameters
               policy_minibatch_size: int = 256,
               gamma: float = 0.99,
               gae_lambda: float = 1.0,
               policy_clip_range: float = 0.2,
               value_clip_range: Optional[float] = None,
               target_kl: Optional[float] = None,
               value_coef: float = 0.5,
               entropy_coef: float = 0.0,
               normalize_advantage: bool = True,
               # PPG parameters
               aux_minibatch_size: int = 16,
               n_policy_steps: int = 32,
               n_policy_epochs: int = 1,
               n_value_epochs: int = 1,
               n_aux_epochs: int = 6,
               beta_clone: float = 1.0,
               **kwargs,
               ) -> None:
    super().__init__(env=env,
                     seed=seed,
                     batch_size=policy_minibatch_size,
                     n_rollouts_per_epoch=n_rollouts_per_epoch,
                     n_steps_per_rollout=n_steps_per_rollout,
                     gamma=gamma,
                     gae_lambda=gae_lambda,
                     policy_clip_range=policy_clip_range,
                     value_clip_range=value_clip_range,
                     target_kl=target_kl,
                     value_coef=value_coef,
                     entropy_coef=entropy_coef,
                     normalize_advantage=normalize_advantage,
                     **kwargs)
    self.shared_arch = shared_arch
    # PPG parameters
    self.aux_minibatch_size = aux_minibatch_size
    self.n_policy_steps = n_policy_steps
    self.n_policy_epochs = n_policy_epochs
    self.n_value_epochs = n_value_epochs
    self.n_aux_epochs = n_aux_epochs
    self.beta_clone = beta_clone

    # Auxiliary buffer
    self.aux_buffer_size = int(self.n_envs * self.n_policy_steps)
    self.aux_obs = torch.zeros(
        (self.n_steps_per_rollout, self.aux_buffer_size) +
        self.observation_space.shape
    )
    self.aux_returns = torch.zeros(
        (self.n_steps_per_rollout, self.aux_buffer_size))

  def training_step(self, batch: Union[RolloutBatch, AuxiliaryBatch], batch_idx: int) -> float:
    if isinstance(batch, RolloutBatch):
      if self.shared_arch:
        # If the actor and critic branches share a common feature space, only the policy is updated here. The value function is updated in the auxiliary phase. See section 3.6 of the PPG paper.
        batch = batch._replace(values=batch.values.detach())
      # In the policy phase, compute the standard PPO loss.
      return PPO.training_step(self, batch, batch_idx)

    elif isinstance(batch, AuxiliaryBatch):
      # Compute the joint loss from section 2 of the paper
      _, new_values, _, _, new_policy, new_aux_values = self.act(
          batch.observations)
      kl_loss = torch.distributions.kl_divergence(
          batch.old_policy, new_policy).mean()
      aux_value_loss = F.mse_loss(new_aux_values, batch.returns)
      joint_loss = aux_value_loss + self.beta_clone * kl_loss

      # Compute the "standard" value loss like in regular PPO.
      real_value_loss = F.mse_loss(new_values, batch.returns)
      loss = joint_loss + real_value_loss
      return loss

  def collect_rollouts(self) -> Union[Iterator[RolloutBatch], Iterator[AuxiliaryBatch]]:
    """
    Perform rollouts in the environment and return the results

    Yields
    ------
    Iterator[RolloutBatch]
        Metrics from a randomized single time step in the current rollout (from multiple agents if the environment is vectorized). These metrics include:
        - observation tensors
        - actions tensors
        - state-values
        - log-probabilities over actions,
        - Advantages
        - Returns
    """
    assert self._last_obs is not None, "No previous observation was provided"
    with torch.no_grad():
      self.continue_training = True

      # POLICY PHASE
      for i_policy_step in range(self.n_policy_steps):
        self.eval()
        while not self.rollout_buffer.full():
          # Convert to pytorch tensor, let lightning take care of any GPU transfers
          obs_tensor = torch.as_tensor(self._last_obs, dtype=torch.float32).to(
              device=self.device)
          done_tensor = torch.as_tensor(
              self._last_dones, dtype=torch.float32).to(device=obs_tensor.device)
          # Compute actions, values, and log-probs
          action, value, log_prob = self.act(obs_tensor)[:3]
          # Perform actions and update the environment
          new_obs, rewards, terminated, truncated, infos = self.env.step(
              action.cpu().numpy())
          new_dones = terminated
          # Convert buffer entries to tensor
          reward_tensor = torch.as_tensor(rewards).to(
              device=obs_tensor.device, dtype=torch.float32)
          # Store the data in the rollout buffer
          self.rollout_buffer.add(
              obs=obs_tensor,
              action=action,
              reward=reward_tensor,
              done=done_tensor,
              value=value,
              log_prob=log_prob)
          self._last_obs = torch.as_tensor(new_obs, dtype=torch.float32).to(
              self.device)
          self._last_dones = torch.as_tensor(
              new_dones, dtype=torch.float32).to(self.device)

          # Update the number of environment steps taken across ALL agents
          self.total_step_count += self.n_envs

        # Use GAE to compute the advantage and return
        final_value_tensor = self.act(self._last_obs)[1]
        samples = self.rollout_buffer.finalize(
            final_value_tensor,
            self._last_dones
        )
        self.rollout_buffer.reset()

        # Train on minibatches from the current rollout.
        self.train()
        n_samples = self.n_steps_per_rollout * self.n_envs
        indices = np.arange(n_samples)
        for epoch in range(self.n_policy_epochs):
          # Check if the training_step has requested to stop training on the current batch.
          if not self.continue_training:
            break
          np.random.shuffle(indices)
          for start in range(0, n_samples, self.batch_size):
            end = start + self.batch_size
            minibatch_inds = indices[start:end]
            yield RolloutBatch(
                observations=samples.observations[minibatch_inds],
                actions=samples.actions[minibatch_inds],
                values=samples.values[minibatch_inds],
                log_probs=samples.log_probs[minibatch_inds],
                advantages=samples.advantages[minibatch_inds],
                returns=samples.returns[minibatch_inds],
            )

        # Add the current rollout states and returns to the auxiliary buffer
        storage_slice = slice(self.n_envs * i_policy_step,
                              self.n_envs * (i_policy_step + 1))
        obs = samples.observations.view_as(self.aux_obs[:, storage_slice])
        returns = samples.returns.view_as(self.aux_returns[:, storage_slice])
        self.aux_obs[:, storage_slice] = obs.cpu().clone()
        self.aux_returns[:, storage_slice] = returns.cpu().clone()

      # Compute and store the current policy for all states in the aux buffer
      aux_inds = np.arange(self.aux_buffer_size)
      action_shape = self.action_space.shape or self.action_space.n
      if isinstance(self.action_space.n, int):
        action_shape = [self.action_space.n]
      aux_action_logits = torch.zeros(
          (self.n_steps_per_rollout, self.aux_buffer_size, *action_shape))
      for start in range(0, self.aux_buffer_size, self.aux_minibatch_size):
        end = start + self.aux_minibatch_size
        aux_minibatch_inds = aux_inds[start:end]
        # Flatten the first two dimensions of the observation tensor so that it can be passed into the action network
        aux_obs = self.aux_obs[:, aux_minibatch_inds].to(
            torch.float32).to(self.device)
        aux_obs_shape = aux_obs.shape
        aux_obs = aux_obs.view(
            (-1, *aux_obs_shape[2:]))
        # Compute the action logits and store them in (step, minibatch, *action dim form)
        with torch.no_grad():
          action_logits = self.forward(aux_obs)[0].cpu().clone()
        aux_action_logits[:, aux_minibatch_inds] = action_logits.view(
            *aux_obs_shape[:2], -1)
        del aux_obs

      # AUXILIARY PHASE
      for epoch in range(self.n_aux_epochs):
        np.random.shuffle(aux_inds)
        for start in range(0, self.aux_buffer_size, self.aux_minibatch_size):
          end = start + self.aux_minibatch_size
          aux_minibatch_ind = aux_inds[start:end]
          # Get the observations and flatten them so that they can be passed into the actor network
          aux_obs = self.aux_obs[:, aux_minibatch_ind].to(self.device)
          aux_obs_shape = aux_obs.shape
          aux_obs = aux_obs.view((-1, *aux_obs_shape[2:]))
          # Get the returns directly from the policy phase
          aux_returns = self.aux_returns[:, aux_minibatch_ind].to(self.device)
          aux_returns = aux_returns.view(-1, *self.aux_returns.shape[2:])
          # Compute the "old" policy from the action logits
          old_action_logits = aux_action_logits[:, aux_minibatch_ind].to(
              self.device).view(-1, *aux_action_logits.shape[2:])
          old_policy = self.logits_to_action_dist(old_action_logits)
          yield AuxiliaryBatch(
              observations=aux_obs,
              returns=aux_returns,
              old_policy=old_policy
          )


if __name__ == '__main__':
  ppg = PPG(gym.make('CartPole-v1'))
