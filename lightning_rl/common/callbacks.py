from pytorch_lightning.callbacks import Callback
import gym
from typing import List, Tuple, Union, Optional
import numpy as np
import pytorch_lightning as pl


class EvalCallback(Callback):
  """
  Callback for evaluating an agent.

  Parameters
  ----------
  Callback : _type_
      _description_
  """

  def __init__(self,
               eval_env: Union[gym.Env, gym.vector.VectorEnv],
               n_eval_episodes: int = 5,
               eval_freq: int = 10000,
               deterministic: bool = True,
               render: bool = False) -> None:
    # TODO: Remove this constraint
    if isinstance(eval_env, gym.vector.VectorEnv):
      assert eval_env.num_envs == 1, "Cannot run eval_env in parallel. eval_env.num_env must equal 1"
      
    self.eval_env = eval_env
    # Make the environment a vector environment for consistency
    if (isinstance(self.eval_env, gym.Env)):
      self.eval_env = gym.vector.SyncVectorEnv([lambda: self.eval_env])
      
    self.n_eval_episodes = n_eval_episodes
    self.eval_freq = eval_freq
    self.deterministic = deterministic
    self.render = render

  def on_train_epoch_start(self, *args, **kwargs) -> None:
    # TODO: These may not always be the args
    trainer, model = args
    model.eval()
    rewards, lengths = self.evaluate(model=model)
    model.train()
    model.log_dict({
        'mean_reward': np.mean(rewards),
        'mean_length': np.mean(lengths),
    },
        prog_bar=True, logger=True, sync_dist=True)

  def evaluate(self,
               model: pl.LightningModule) -> Tuple[List[float], List[int]]:
    """
    Evaluate the model in an evaluation environment

    Parameters
    ----------
    eval_env : Union[gym.Env, gym.vector.VectorEnv, str]
        Evaluation environment
    n_eval_episodes : int
        Number of episodes used for evaluation
    deterministic : bool, optional
        If true, actions are chosen deterministically, by default True
    render : bool, optional
        If true, renders the environment, by default False

    Returns
    -------
    Tuple[List[float], List[int]]
        _description_
    """
    episode_rewards, episode_lengths = [], []

    reset = False
    for _ in range(self.n_eval_episodes):
      done = False
      episode_rewards += [0.0]
      episode_lengths += [0]
      # Number of loops here might differ from true episodes
      # played, if underlying wrappers modify episode lengths.
      # Avoid double reset, as VecEnv are reset automatically.
      if not isinstance(self.eval_env, gym.vector.VectorEnv) or not reset:
        obs = self.eval_env.reset()
        reset = True
      while not done:
        action = model.act(obs, self.deterministic)
        obs, reward, done, info = self.eval_env.step(action)
        episode_rewards[-1] += reward
        episode_lengths[-1] += 1
        if self.render:
          self.eval_env.render()
    return np.mean(episode_rewards), np.mean(episode_lengths)
