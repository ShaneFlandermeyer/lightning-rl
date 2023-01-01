from enum import Enum
from typing import Iterator, Optional, Union

import gymnasium as gym
import numpy as np
import torch

from lightning_rl.common.buffers import RolloutBatch
from lightning_rl.models import on_policy_models
from lightning_rl.models.on_policy_models.ppo import PPO


class TrainingPhase(Enum):
  """
  Enum to indicate which phase of training we are in. This is switched in collect_rollouts() and used in training_step().
  """
  POLICY = 1
  AUXILIARY = 2


class PPG(PPO):
  def __init__(self,
               env: Union[gym.Env, gym.vector.VectorEnv, str],
               seed: Optional[int] = None,
               n_rollouts_per_epoch: int = 100,
               n_steps_per_rollout: int = 256,
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
               aux_minibatch_size: int = 256,
               n_policy_steps: int = 1,
               n_policy_epochs: int = 1,
               n_value_epochs: int = 1,
               n_aux_epochs: int = 1,
               beta_clone: float = 1.0,
               **kwargs,
               ) -> None:
    super(PPO, self).__init__(env=env,
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
    # PPG parameters
    self.aux_minibatch_size = aux_minibatch_size
    self.n_policy_steps = n_policy_steps
    self.n_policy_epochs = n_policy_epochs
    self.n_value_epochs = n_value_epochs
    self.n_aux_epochs = n_aux_epochs
    self.beta_clone = beta_clone

    # Buffers for
    self.aux_buffer_size = int(self.n_envs * self.n_policy_steps)
    self.aux_obs = torch.zeros(
        (self.n_steps_per_rollout, self.aux_buffer_size) +
        self.observation_space.shape
    )
    self.aux_returns = torch.zeros(
        (self.n_steps_per_rollout, self.aux_buffer_size))

  def training_step(self, batch: RolloutBatch, batch_idx: int) -> float:
    if self.phase == TrainingPhase.POLICY:
      # Compute PPO loss
      return super(PPO, self).training_step(batch, batch_idx)
    elif self.phase == TrainingPhase.AUXILIARY:
      # TODO: Compute auxiliary losses
      pass

  def collect_rollouts(self) -> Iterator[RolloutBatch]:
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

      # Policy phase
      for i_policy_step in range(self.n_policy_steps):
        self.eval()
        while not self.rollout_buffer.full():
          # Convert to pytorch tensor, let lightning take care of any GPU transfers
          obs_tensor = torch.as_tensor(self._last_obs).to(
              device=self.device, dtype=torch.float32)
          done_tensor = torch.as_tensor(
              self._last_dones).to(device=obs_tensor.device, dtype=torch.float32)
          # Compute actions, values, and log-probs
          action, value, log_prob, entropy = self.act(
              obs_tensor)
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
          self._last_dones = torch.as_tensor(new_dones).to(self.device)

          # Update the number of environment steps taken across ALL agents
          self.total_step_count += self.n_envs

        # Use GAE to compute the advantage and return
        final_obs_tensor = torch.as_tensor(new_obs).to(
            device=self.device, dtype=torch.float32)
        final_value_tensor = self.forward(final_obs_tensor)[1]
        new_done_tensor = torch.as_tensor(new_dones).to(
            device=obs_tensor.device, dtype=torch.float32)
        samples = self.rollout_buffer.finalize(
            final_value_tensor,
            new_done_tensor
        )
        self.rollout_buffer.reset()

        # Train on minibatches from the current rollout.
        self.train()
        n_samples = self.n_steps_per_rollout * self.n_envs
        indices = np.arange(n_samples)
        self.phase = TrainingPhase.POLICY
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

        # Save rollouts for PPG
        storage_slice = slice(self.n_envs * i_policy_step,
                              self.n_envs * (i_policy_step + 1))
        # TODO: Make these a deque?
        self.aux_obs[:, storage_slice] = samples.observations.cpu(
        ).clone()
        self.aux_returns[:, storage_slice] = samples.returns.cpu().clone()

      # Auxiliary phase
      aux_inds = np.arange(self.aux_buffer_size)

      # TODO: Compute and store the current policy for all states in the aux buffer
      action_shape = self.action_space.shape or self.action_space.n
      aux_pi = torch.zeros(
          (self.n_steps_per_rollout, self.aux_buffer_size) + action_shape)
      for i, start in enumerate(range(0, self.aux_buffer_size, self.aux_minibatch_size)):
        end = start + self.aux_minibatch_size
        aux_minibatch_inds = aux_inds[start:end]
        minibatch_aux_obs = self.aux_obs[:, aux_minibatch_inds].to(
            torch.float32).to(self.device)
        minibatch_obs_shape = minibatch_aux_obs.shape
        # TODO: Might have to flatten the obs
        # TODO: Compute and store the logits for the current action

      self.phase = TrainingPhase.AUXILIARY
      for i_aux_epoch in range(self.n_aux_epochs):
        np.random.suffle(aux_inds)


if __name__ == '__main__':
  ppg = PPG(gym.make('CartPole-v1'))
