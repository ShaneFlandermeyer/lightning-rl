import numpy as np
import torch
from lightning_rl.common.buffers import RecurrentRolloutBatch, RecurrentRolloutBuffer
from lightning_rl.common.datasets import OnPolicyDataset
from lightning_rl.common.utils import explained_variance
from lightning_rl.models.on_policy_models import PPO
import gym
from typing import Iterator, Tuple, Union, Optional
from torch import distributions
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader


from lightning_rl.models.rl_model import RLModel


class RecurrentOnPolicyModel(RLModel):
  """TODO: This should probably subclass from a RecurrentOnPolicyModel class instead"""

  def __init__(self,
               env: Union[gym.Env, gym.vector.VectorEnv, str],
               n_steps_per_rollout: int,
               n_rollouts_per_epoch: int,
               n_gradient_steps: int,
               n_minibatch: int = 4,
               gamma: float = 0.99,
               gae_lambda: float = 1.0,
               seed: Optional[int] = None,
               **kwargs) -> None:

    super().__init__(
        env=env,
        support_multi_env=True,
        seed=seed,
        **kwargs,
    )
    assert self.n_envs % n_minibatch == 0, "n_envs must be divisible by n_minibatch"
    self.n_steps_per_rollout = n_steps_per_rollout
    self.n_rollouts_per_epoch = n_rollouts_per_epoch
    self.n_gradient_steps = n_gradient_steps
    self.n_minibatch = n_minibatch
    self.batch_size = self.n_steps_per_rollout * self.n_envs
    self.gamma = gamma
    self.gae_lambda = gae_lambda

    self.rollout_buffer = RecurrentRolloutBuffer(
        n_steps_per_rollout,
        gamma=self.gamma,
        gae_lambda=self.gae_lambda,
        n_envs=self.n_envs,
    )
    self._last_hidden_state = None
    self.initial_hidden_state = None

    # Metrics
    self.total_step_count = 0

  def collect_rollouts(self) -> Iterator[RecurrentRolloutBatch]:
    assert self._last_obs is not None, "No previous observation was provided"
    with torch.no_grad():
      self.continue_training = True

      for _ in range(self.n_rollouts_per_epoch):
        self.eval()
        # The initial hidden state gets used in the training step so we can reconstruct the probability distribution in the rollout. See LSTM implementation detail #5 here: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
        if self._last_hidden_state is not None:
          self.initial_hidden_state = (
              self._last_hidden_state[0].clone(),
              self._last_hidden_state[1].clone())
        while not self.rollout_buffer.full():
          # Convert to pytorch tensor, let lightning take care of any GPU transfers
          obs_tensor = torch.as_tensor(self._last_obs).to(
              device=self.device, dtype=torch.float32)
          done_tensor = torch.as_tensor(
              self._last_dones).to(device=obs_tensor.device, dtype=torch.float32)
          # Compute actions, values, and log-probs
          action, value, log_prob, entropy, hidden_state = self.act(
              obs_tensor, self._last_hidden_state, done_tensor)
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
          self._last_hidden_state = hidden_state

          # Update the number of environment steps taken across ALL agents
          self.total_step_count += self.n_envs

        # Use GAE to compute the advantage and return
        final_value_tensor = self.act(
            self._last_obs, self._last_hidden_state, self._last_dones.float())[-2]
        samples = self.rollout_buffer.finalize(
            final_value_tensor,
            self._last_dones
        )
        self.rollout_buffer.reset()

        # Train on minibatches from the current rollout.
        self.train()
        envs_per_batch = self.n_envs // self.n_minibatch
        env_inds = np.arange(self.n_envs)
        flat_inds = np.arange(self.n_steps_per_rollout*self.n_envs).reshape(
            self.n_steps_per_rollout, self.n_envs)
        for epoch in range(self.n_gradient_steps):
          # Check if the training_step has requested to stop training on the current batch.
          if not self.continue_training:
            break
          # If recurrent layers are used, the training batch samples should occur in the same order they were collected. Therefore, minibatches are shuffled by environment instead.
          np.random.shuffle(env_inds)
          for start in range(0, self.n_envs, envs_per_batch):
            end = start + envs_per_batch
            minibatch_env_inds = env_inds[start:end]
            indices = flat_inds[:, minibatch_env_inds].ravel()
            yield RecurrentRolloutBatch(
                observations=samples.observations[indices],
                actions=samples.actions[indices],
                dones=samples.dones[indices],
                values=samples.values[indices],
                log_probs=samples.log_probs[indices],
                advantages=samples.advantages[indices],
                returns=samples.returns[indices],
                hidden_states=(
                    self.initial_hidden_state[0][:, minibatch_env_inds],
                    self.initial_hidden_state[1][:, minibatch_env_inds]),
            )

  def train_dataloader(self):
    """
    Create the dataloader for our OnPolicyModel
    """
    self.dataset = OnPolicyDataset(
        rollout_generator=self.collect_rollouts,
    )
    return DataLoader(dataset=self.dataset, batch_size=None)

  def forward(self,
              x: torch.Tensor,
              hidden_state: torch.Tensor,
              done: torch.Tensor):
    """
    Compute the unprocessed output of the network. These outputs do not have to take a specific form. It is up to the act() function to process the output into the desired form.

    Parameters
    ----------
    x : torch.Tensor
        Observation Tensor
    hidden_state : torch.Tensor
        Hidden state of the recurrent layers
    done : torch.Tensor
        Tensor indicating whether the episode is done

    Returns
    -------
    Tuple[torch.Tensor, ...]
        The network output
    """
    raise NotImplementedError

  def act(self,
          x: torch.Tensor,
          hidden_state: torch.Tensor,
          done: torch.Tensor):
    """
    Compute the processed output of the network. 

    Parameters
    ----------
    x : torch.Tensor
        Input observation
    hidden_state : torch.Tensor
        Hidden state of the recurrent layers
    done : torch.Tensor
        Tensor indicating whether the episode is done

    Returns
    -------
    Tuple[torch.Tensor, ...]
        The elements of the output tuple are:
        - action: The action selected by the network
        - value: The value of the input observation
        - log_prob: The log-probability of the selected action
        - entropy: The entropy of the action distribution
        - hidden_state: the updated hidden state of the recurrent layers
    """
    raise NotImplementedError

  def evaluate_actions(self,
                       observations: torch.Tensor,
                       actions: torch.Tensor,
                       hidden_states: Tuple[torch.Tensor],
                       dones: torch.Tensor):
    """
    Compute the log-probability of the input actions, along with the entropy of the action distribution and the value of the input observation.

    Parameters
    ----------
    observations : torch.Tensor
        Current environment state/observation
    actions : torch.Tensor
        Actions to evaluate
    hidden_states : torch.Tensor
        Hidden state of the recurrent layers
    dones : torch.Tensor
        Tensor indicating whether the episode is done

    Returns
    -------
    Tuple[torch.Tensor, ...]
        The elements of the output tuple are:
        - log_prob: The log-probability of the selected action
        - entropy: The entropy of the action distribution
        - value: The value of the input observation
        - hidden_state: the updated hidden state of the recurrent layers
    """
    raise NotImplementedError
