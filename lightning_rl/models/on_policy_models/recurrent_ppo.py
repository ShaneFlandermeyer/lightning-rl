import numpy as np
import torch
from lightning_rl.common.buffers import RolloutBatch, RolloutSample
from lightning_rl.common.utils import explained_variance
from lightning_rl.models.on_policy_models import PPO
import gym
from typing import Iterator, Tuple, Union, Optional
from torch import distributions
import torch.nn.functional as F
import torch.nn as nn


class RecurrentPPO(PPO):
  """TODO: This should probably subclass from a RecurrentOnPolicyModel class instead"""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._last_lstm_states = None

  def collect_rollouts(self) -> Iterator[RolloutSample]:
    """
    Perform rollouts in the environment and return the results

    Yields
    ------
    Iterator[RolloutSample]
        Metrics from a randomized single time step in the current rollout (from multiple agents if the environment is vectorized). These metrics include:
        - observation tensors
        - actions tensors
        - state-values
        - log-probabilities over actions,
        - Advantages
        - Returns
    """
    assert self._last_obs is not None, "No previous observation was provided"
    if self._last_lstm_states is None:
      hidden_state_shape = (self.lstm.num_layers,
                            self.n_envs, self.lstm.hidden_size)
      self._last_lstm_states = (
          torch.zeros(hidden_state_shape, device=self.device,
                      dtype=torch.float32),
          torch.zeros(hidden_state_shape, device=self.device,
                      dtype=torch.float32),
      )

    with torch.no_grad():
      self.continue_training = True
      for _ in range(self.n_rollouts_per_epoch):
        self.eval()
        lstm_states = self._last_lstm_states
        while not self.rollout_buffer.full():
          # Convert to pytorch tensor, let lightning take care of any GPU transfers
          obs_tensor = torch.as_tensor(self._last_obs).to(
              device=self.device, dtype=torch.float32)
          done_tensor = torch.as_tensor(
              self._last_dones).to(device=obs_tensor.device, dtype=torch.float32)

          # Compute actions, values, and log-probs
          action_tensor, value_tensor, lstm_states = self.forward(
              obs_tensor, lstm_states, done_tensor)
          log_prob_tensor, _ = self.evaluate_actions(
              obs_tensor, action_tensor, lstm_states, done_tensor)
          actions = action_tensor.cpu().numpy()
          # Perform actions and update the environment
          new_obs, rewards, terminated, truncated, infos = self.env.step(
              actions)
          new_dones = terminated
          # Convert buffer entries to tensor
          if isinstance(self.action_space, gym.spaces.Discrete):
            # Reshape in case of discrete actions
            action_tensor = action_tensor.view(-1, 1)
          reward_tensor = torch.as_tensor(rewards).to(
              device=obs_tensor.device, dtype=torch.float32)

          # Store the data in the rollout buffer
          self.rollout_buffer.add(
              obs_tensor,
              action_tensor,
              reward_tensor,
              done_tensor,
              value_tensor,
              log_prob_tensor)
          self._last_obs = new_obs
          self._last_dones = new_dones
          self._last_lstm_states = lstm_states

          # Update the number of environment steps taken across ALL agents
          self.total_step_count += self.n_envs

        final_obs_tensor = torch.as_tensor(new_obs).to(
            device=self.device, dtype=torch.float32)
        new_done_tensor = torch.as_tensor(new_dones).to(
            device=obs_tensor.device, dtype=torch.float32)
        final_value_tensor = self.forward(
            final_obs_tensor, lstm_states, new_done_tensor)[1]

        samples = self.rollout_buffer.finalize(
            final_value_tensor,
            new_done_tensor
        )
        self.rollout_buffer.reset()

        # Train on minibatches from the current rollout.
        self.train()
        for _ in range(self.n_gradient_steps):
          # Check if the training_step has requested to stop training on the current batch.
          if not self.continue_training:
            break
          indices = np.random.permutation(
              self.n_steps_per_rollout * self.n_envs)
          for idx in indices:
            yield RolloutSample(
                observations=samples.observations[idx],
                actions=samples.actions[idx],
                values=samples.values[idx],
                log_probs=samples.log_probs[idx],
                advantages=samples.advantages[idx],
                returns=samples.returns[idx])

  @staticmethod
  def _process_sequence(
      features: torch.Tensor,
      lstm_states: Tuple[torch.Tensor, torch.Tensor],
      episode_starts: torch.Tensor,
      lstm: nn.LSTM,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Do a forward pass in the LSTM network.
    :param features: Input tensor
    :param lstm_states: previous cell and hidden states of the LSTM
    :param episode_starts: Indicates when a new episode starts,
        in that case, we need to reset LSTM states.
    :param lstm: LSTM object.
    :return: LSTM output and updated LSTM states.
    """
    # LSTM logic
    # (sequence length, batch size, features dim)
    # (batch size = n_envs for data collection or n_seq when doing gradient update)
    n_seq = lstm_states[0].shape[1]
    # Batch to sequence
    # (padded batch size, features_dim) -> (n_seq, max length, features_dim) -> (max length, n_seq, features_dim)
    # note: max length (max sequence length) is always 1 during data collection
    features_sequence = features.reshape(
        (n_seq, -1, lstm.input_size)).swapaxes(0, 1)
    episode_starts = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)

    # If we don't have to reset the state in the middle of a sequence
    # we can avoid the for loop, which speeds up things
    if torch.all(episode_starts == 0.0):
      lstm_output, lstm_states = lstm(features_sequence, lstm_states)
      lstm_output = torch.flatten(
          lstm_output.transpose(0, 1), start_dim=0, end_dim=1)
      return lstm_output, lstm_states

    lstm_output = []
    # Iterate over the sequence
    for features, episode_start in zip(features_sequence, episode_starts, strict=True):
      hidden, lstm_states = lstm(
          features.unsqueeze(dim=0),
          (
              # Reset the states at the beginning of a new episode
              (1.0 - episode_start).view(1, n_seq, 1) * lstm_states[0],
              (1.0 - episode_start).view(1, n_seq, 1) * lstm_states[1],
          ),
      )
      lstm_output += [hidden]
    # Sequence to batch
    # (sequence length, n_seq, lstm_out_dim) -> (batch_size, lstm_out_dim)
    lstm_output = torch.flatten(
        torch.cat(lstm_output).transpose(0, 1), start_dim=0, end_dim=1)
    return lstm_output, lstm_states

  def forward(self,
              x: torch.Tensor,
              lstm_states: Tuple[torch.Tensor],
              dones: torch.Tensor):
    raise NotImplementedError

  def evaluate_actions(self,
                       observations: torch.Tensor,
                       actions: torch.Tensor,
                       lstm_states: Tuple[torch.Tensor],
                       dones: torch.Tensor):
    raise NotImplementedError
