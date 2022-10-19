from typing import Optional, Tuple, Union
import gym
import numpy as np
import pytorch_lightning as pl
import torch
from lightning_rl.common.rl_model import RLModel
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from lightning_rl.common.buffers import ReplayBuffer, ExperienceBatch


class OffPolicyModel(RLModel):

  def __init__(
      self,
      env: Union[gym.Env, VecEnv, str],
      batch_size: int = 256,
      replay_buffer_size: int = int(1e6),
      n_warmup_steps: int = 100,
      train_freq: int = 1,
      n_episodes_per_epoch: int = 1,
      n_gradient_steps: int = 1,
      gamma: float = 0.99,
      squashed_actions: bool = False,
      seed: Optional[int] = None,
  ):
    """
    Base class for Off-Policy Algorithms

    Parameters
    ----------
    env : Union[gym.Env, VecEnv, str]
        The environment to learn from
        (if registered in Gym, can be str. Can be None for loading trained models)
    batch_size : int, optional
        Mini-batch size used for the gradient update, by default 256
    replay_buffer_size : int, optional
        The maximum number of experiences in the replay buffer, by default int(1e6)
    n_warmup_steps : int, optional
        The number of steps to collect before training, by default 100
    train_freq : int, optional
        The number of steps performed between each model update, by default -1
    n_episodes_per_epoch : int, optional
        The number of rollouts to collect for each Pytorch Lightning epoch, by default 1
    n_gradient_steps : int, optional
        The number of gradient steps to perform for each rollout, by default 1
    gamma : float, optional
        The discount factor, by default 0.99
    squashed_actions : bool, optional
        Whether to use squashed actions, by default False
    seed : Optional[int], optional
        Seed for the pseudo random number generators, by default None
    """
    super().__init__(
        env=env,
        support_multi_env=True,
        seed=seed)
    assert self.env.num_envs == 1, "OffPolicyModel only supports non-vectorized environments"
    self.batch_size = batch_size
    self.replay_buffer_size = replay_buffer_size
    self.n_warmup_steps = n_warmup_steps
    self.train_freq = train_freq
    self.n_episodes_per_epoch = n_episodes_per_epoch
    self.n_gradient_steps = n_gradient_steps
    self.gamma = gamma
    self.squashed_actions = squashed_actions

    # TODO: Implement the replay buffer
    self.replay_buffer = ReplayBuffer()

    def reset(self):
      """
      Reset the environment and the step counter
      """
      self.state = self.env.reset()
      self.total_step_count = 0

    def on_step(self):
      """
      Callback for each step taken in the environment
      """
      raise NotImplementedError

    def train_dataloader(self):
      """
      Create the dataloader for the model
      """
      return OffPolicyDataLoader(self)

    def scale_actions(self,
                      actions: np.ndarray,
                      squashed=False) -> Tuple[np.ndarray, np.ndarray]:
      """
      Convert the actions to the range required by the action space. If the actions are squashed between [-1, 1], the actions are linearly scaled into the range [low, high]. If the actions are not squashed, they are clipped into the valid range.

      Parameters
      ----------
      actions : np.ndarray
          Input actions
      squashed : bool, optional
          If true, actions are assumed to be squashed into the range [-1, 1], by default False

      Returns
      -------
      Tuple[np.ndarray, np.ndarray]
          Valid action values
      """
      low, high = self.action_space.low, self.action_space.high
      center = (low + high) / 2
      if squashed:
        actions = center + actions * (high - low) / 2
      else:
        actions = np.clip(
            actions,
            self.action_space.low,
            self.action_space.high
        )
      return actions

  def act(
      self,
      state: np.ndarray,
      deterministic: bool = False
  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform an action based on the current state of the environment

    Parameters
    ----------
    state : np.ndarray
        Input state
    deterministic : bool, optional
        If true, samples the action deterministically, by default False

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The action to perform
    """
    with torch.no_grad():
        # Convert to pytorch tensor
      state_tensor = torch.as_tensor(state).to(
          device=self.device, dtype=torch.float32)
      actions = self.predict(state_tensor, deterministic=False)

    # Clip and scale actions appropriately
    if isinstance(self.action_space, gym.spaces.Box):
      actions = self.scale_actions(actions, self.squashed_actions)
    elif isinstance(self.action_space, (gym.spaces.Discrete,
                                        gym.spaces.MultiDiscrete,
                                        gym.spaces.MultiBinary)):
      actions = actions.astype(np.int32)
    return actions

  def collect_experience(self) -> None:
    """
    Collect experiences and store them in the replay buffer
    """

    self.eval()

    step_count = 0
    while step_count < self.train_freq:
      if self.total_step_count < self.n_warmup_steps:
        # Act randomly until we have collected enough warmup experiences
        action = np.array([self.action_space.sample()])
      else:
        # Act based on the current state
        action = self.act(self.state, deterministic=False)

      next_state, reward, done, info = self.env.step(action)

      if isinstance(self.action_space, gym.spaces.Discrete):
        # Reshape in case of discrete action
        action = action.reshape(-1, 1)

      # Store the experience in the replay buffer
      self.replay_buffer.append(self.state, action, reward, next_state, done)

      self.state = next_state
      self.total_step_count += 1
      step_count += 1

    self.train()

    self.log('n_steps', self.n_steps, on_step=True, prog_bar=True, logger=True)


class OffPolicyDataLoader:
  def __init__(self, model: OffPolicyModel):
    self.model = model

  def __iter__(self):
    for i in range(self.model.num_rollouts):
      self.model.collect_rollouts()
      for j in range(self.n_gradient_steps):
        yield self.model.replay_buffer.sample()
