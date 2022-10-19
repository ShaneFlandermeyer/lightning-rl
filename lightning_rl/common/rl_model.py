import inspect
import warnings
import pytorch_lightning as pl
import gym
from typing import List, Union, Optional, Tuple, Dict, Any
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
import torch
from gym.wrappers.monitor import Monitor
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from stable_baselines3.common.env_util import is_wrapped


class RLModel(pl.LightningModule):
  """
    Base class for all RL algorithms

    Parameters
    ----------
    env : Union[gym.Env, VecEnv, str]
        The environment to learn from
        (if registered in Gym, can be str. Can be None for loading trained models)
    support_multi_env : bool, optional
        Whether the algorithm supports training
      with multiple environments in parallel, by default False
    seed : Optional[int], optional
        Seed for the pseudo random number generators, by default None

    Raises
    ------
    ValueError
        _description_
  """

  def __init__(
      self,
      env: Union[gym.Env, VecEnv, str],
      support_multi_env: bool = False,
      seed: Optional[int] = None
  ) -> None:
    super().__init__()

    # Create the environment
    if isinstance(env, str):
      self.env = gym.make(env)
    else:
      self.env = env

    # Make the environment a vector environment
    if (isinstance(self.env, gym.Env)):
      self.env = DummyVecEnv([lambda: self.env])
      
    self.eval_env = self.env

    self.observation_space = self.env.observation_space
    self.action_space = self.env.action_space
    self.n_envs = self.env.num_envs

    if seed:
      self.seed = seed
      self.set_random_seed(self.seed)
    if not support_multi_env and self.n_envs > 1:
      raise ValueError(
          "Error: the model does not support multiple envs; it requires " "a single vectorized environment."
      )

    self.reset()

  def predict(self,
              obs: Union[Tuple, Dict[str, Any], np.ndarray, int], deterministic: bool = False) -> np.ndarray:
    """
    Override this function with the predict function of your own mode

    Parameters
    ----------
    obs : Union[Tuple, Dict[str, Any], np.ndarray, int]
        The input observations
    deterministic : bool, optional
        If true, samples the action deterministically, by default False

    Returns
    -------
    np.ndarray
        The chosen action

    Raises
    ------
    NotImplementedError
    """
    raise NotImplementedError

  def save_hyperparameters(self, frame=None, exclude=['env', 'eval_env']):
    """
    Utility function to save the hyperparameters of the model.
    This function behaves identically to LightningModule.save_hyperparameters, but will by default exclude the Gym environments
    #lightningmodule-hyperparameters for more details
    See https://pytorch-lightning.readthedocs.io/en/latest/hyperparameters.html
    """
    if not frame:
      frame = inspect.currentframe().f_back
    if not exclude:
      return super().save_hyperparameters(frame=frame)
    if isinstance(exclude, str):
      exclude = (exclude, )
    init_args = pl.utilities.parsing.get_init_args(frame)
    include = [k for k in init_args.keys() if k not in exclude]
    if len(include) > 0:
      super().save_hyperparameters(*include, frame=frame)

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
      state = torch.tensor(state).to(self.device)
      action = self.predict(state, deterministic=deterministic)

    if isinstance(self.action_space, gym.spaces.Box):
      action = np.clip(action, self.action_space.low, self.action_space.high)
    elif isinstance(self.action_space, (gym.spaces.Discrete,
                                        gym.spaces.MultiDiscrete,
                                        gym.spaces.MultiBinary)):
      action = action.astype(np.int32)

    return action

  def training_epoch_end(self, outputs) -> None:
    """
    Run this function at the end of each training epoch

    Parameters
    ----------
    outputs : _type_
        Training step outputs
    """
    return

  def reset(self) -> None:
    """
    Reset the environment
    """
    self._last_obs = self.env.reset()
    self._last_dones = np.zeros((self.env.num_envs,), dtype=np.bool)

  def set_random_seed(self, seed: int) -> None:
    """
    Set the seed for all RNGs

    Parameters
    ----------
    seed : int
        Random seed
    """
    pl.seed_everything(seed)
    self.action_space.seed(seed)
    if self.env:
      self.env.seed(seed)
      
  def evaluate(
    self,
    eval_env: Union[gym.Env, VecEnv, str],
    n_eval_episodes: int,
    deterministic: bool = True,
    render: bool = False,
    record: bool = False,
    record_filename: Optional[str] = None) -> Tuple[List[float], List[int]]:
    """
    Evaluate the model in an evaluation environment

    Parameters
    ----------
    eval_env : Union[gym.Env, VecEnv, str]
        Evaluation environment
    n_eval_episodes : int
        Number of episodes used for evaluation
    deterministic : bool, optional
        If true, actions are chosen deterministically, by default True
    render : bool, optional
        If true, renders the environment, by default False
    record : bool, optional
        If true, records a video of the episode, by default False
    record_filename : Optional[str], optional
        If specified, saves the recorded video to a file with this name, by default None

    Returns
    -------
    Tuple[List[float], List[int]]
        _description_
    """
    
    if isinstance(eval_env, VecEnv):
            assert eval_env.num_envs == 1, "Cannot run eval_env in parallel. eval_env.num_env must equal 1"
            
    # Create the environment
    if isinstance(eval_env, str):
      eval_env = gym.make(eval_env)

    # Make the environment a vector environment
    if (isinstance(eval_env, gym.Env)):
      eval_env = DummyVecEnv([lambda: eval_env])

    if not is_wrapped(eval_env, Monitor):
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )
        
    episode_rewards, episode_lengths = [], []
    if record:
        recorder = VideoRecorder(env=eval_env, path=record_filename)
        
    not_reseted = True
    for i in range(n_eval_episodes):
        done = False
        episode_rewards += [0.0]
        episode_lengths += [0]
        # Number of loops here might differ from true episodes
        # played, if underlying wrappers modify episode lengths.
        # Avoid double reset, as VecEnv are reset automatically.
        if not isinstance(eval_env, VecEnv) or not_reseted:
            obs = eval_env.reset()
            not_reseted = False
        while not done:
            action = self.act(obs, deterministic)
            obs, reward, done, info = eval_env.step(action)
            episode_rewards[-1] += reward
            episode_lengths[-1] += 1
            if render:
                eval_env.render()
            if record:
                recorder.capture_frame()
        if is_wrapped(eval_env, Monitor):
            # Do not trust "done" with episode endings.
            # Remove vecenv stacking (if any)
            if isinstance(eval_env, VecEnv):
                info = info[0]
            if "episode" in info.keys():
                # Monitor wrapper includes "episode" key in info if environment
                # has been wrapped with it. Use those rewards instead.
                episode_rewards[-1] = info["episode"]["r"]
                episode_lengths[-1] = info["episode"]["l"]
    if record:
        recorder.close()
    return episode_rewards, episode_lengths


if __name__ == '__main__':
  env = gym.make('CartPole-v1')
  model = RLModel(env)
  model.act(np.zeros((3,)))
