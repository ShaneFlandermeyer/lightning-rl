# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import argparse
import os
import random
import time
from distutils.util import strtobool
from typing import Optional, Tuple

# import gymnasium as gym
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from lightning_rl.common.advantage import estimate_advantage
from lightning_rl.models.on_policy.ppo import ppo_loss


def parse_args():
  # fmt: off
  parser = argparse.ArgumentParser()
  parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
      help="the name of this experiment")
  parser.add_argument("--seed", type=int, default=1,
      help="seed of the experiment")
  parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
      help="if toggled, `torch.backends.cudnn.deterministic=False`")
  parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
      help="if toggled, cuda will be enabled by default")
  parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
      help="whether to capture videos of the agent performances (check out `videos` folder)")

  # Algorithm specific arguments
  parser.add_argument("--env-id", type=str, default="PongNoFrameskip-v4",
      help="the id of the environment")
  parser.add_argument("--total-timesteps", type=int, default=10000000,
      help="total timesteps of the experiments")
  parser.add_argument("--learning-rate", type=float, default=5e-4,
      help="the learning rate of the optimizer")
  parser.add_argument("--num-envs", type=int, default=8,
      help="the number of parallel game environments")
  parser.add_argument("--num-steps", type=int, default=128,
      help="the number of steps to run in each environment per policy rollout")
  parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
      help="Toggle learning rate annealing for policy and value networks")
  parser.add_argument("--gamma", type=float, default=0.99,
      help="the discount factor gamma")
  parser.add_argument("--gae-lambda", type=float, default=0.95,
      help="the lambda for the general advantage estimation")
  parser.add_argument("--num-minibatches", type=int, default=4,
      help="the number of mini-batches")
  parser.add_argument("--update-epochs", type=int, default=4,
      help="the K epochs to update the policy")
  parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
      help="Toggles advantages normalization")
  parser.add_argument("--clip-coef", type=float, default=0.1,
      help="the surrogate clipping coefficient")
  parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
      help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
  parser.add_argument("--ent-coef", type=float, default=0.01,
      help="coefficient of the entropy")
  parser.add_argument("--vf-coef", type=float, default=0.25,
      help="coefficient of the value function")
  parser.add_argument("--max-grad-norm", type=float, default=0.5,
      help="the maximum norm for the gradient clipping")
  parser.add_argument("--target-kl", type=float, default=None,
      help="the target KL divergence threshold")
  args = parser.parse_args()
  args.batch_size = int(args.num_envs * args.num_steps)
  args.minibatch_size = int(args.batch_size // args.num_minibatches)
  # fmt: on
  return args


def make_env(env_id, seed, idx, capture_video, run_name):
  def thunk():
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if capture_video:
      if idx == 0:
        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    env = gym.wrappers.AtariPreprocessing(
        env, screen_size=84, grayscale_obs=True, grayscale_newaxis=False)
    env = gym.wrappers.FrameStack(env, 1)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

  return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
  torch.nn.init.orthogonal_(layer.weight, std)
  torch.nn.init.constant_(layer.bias, bias_const)
  return layer


class Agent(nn.Module):
  def __init__(self, envs):
    super().__init__()
    self.network = nn.Sequential(
        layer_init(nn.Conv2d(1, 32, 8, stride=4)),
        nn.ReLU(),
        layer_init(nn.Conv2d(32, 64, 4, stride=2)),
        nn.ReLU(),
        layer_init(nn.Conv2d(64, 64, 3, stride=1)),
        nn.ReLU(),
        nn.Flatten(),
        layer_init(nn.Linear(64 * 7 * 7, 512)),
        nn.ReLU(),
    )
    self.gru = nn.GRU(512, 128)
    for name, param in self.gru.named_parameters():
      if "bias" in name:
        nn.init.constant_(param, 0)
      elif "weight" in name:
        nn.init.orthogonal_(param, 1.0)
    self.actor = layer_init(
        nn.Linear(128, envs.single_action_space.n), std=0.01)
    self.critic = layer_init(nn.Linear(128, 1), std=1)

  def get_states(self,
                 x: torch.Tensor,
                 gru_state: torch.Tensor,
                 done: torch.Tensor
                 ) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
    hidden = self.network(x / 255.0)

    # GRU logic
    batch_size = gru_state.shape[1]
    hidden = hidden.reshape((-1, batch_size, self.gru.input_size))
    done = done.reshape((-1, batch_size))
    new_hidden = []
    for h, d in zip(hidden, done):
      h, gru_state = self.gru(
          h.unsqueeze(0),
          (1.0 - d).view(1, -1, 1) * gru_state
      )
      new_hidden += [h]
    new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
    return new_hidden, gru_state

  def get_value(self,
                x: torch.Tensor,
                gru_state: torch.Tensor,
                done: torch.Tensor) -> torch.Tensor:
    hidden, _ = self.get_states(x, gru_state, done)
    return self.critic(hidden)

  def get_action_and_value(self,
                           x: torch.Tensor,
                           gru_state: torch.Tensor,
                           done: torch.Tensor,
                           action: Optional[torch.Tensor] = None):
    hidden, gru_state = self.get_states(x, gru_state, done)
    logits = self.actor(hidden)
    probs = Categorical(logits=logits)
    if action is None:
      action = probs.sample()
    return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), gru_state


if __name__ == "__main__":
  args = parse_args()
  run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
  writer = SummaryWriter(f"runs/{run_name}")
  writer.add_text(
      "hyperparameters",
      "|param|value|\n|-|-|\n%s" % (
          "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
  )

  # TRY NOT TO MODIFY: seeding
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.backends.cudnn.deterministic = args.torch_deterministic

  device = torch.device("cuda" if torch.cuda.is_available()
                        and args.cuda else "cpu")

  # env setup
  envs = gym.vector.AsyncVectorEnv(
      [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name)
       for i in range(args.num_envs)]
  )
  assert isinstance(envs.single_action_space,
                    gym.spaces.Discrete), "only discrete action space is supported"

  agent = Agent(envs).to(device)
  optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

  # ALGO Logic: Storage setup
  obs = torch.zeros((args.num_steps, args.num_envs) +
                    envs.single_observation_space.shape).to(device)
  actions = torch.zeros((args.num_steps, args.num_envs) +
                        envs.single_action_space.shape).to(device)
  logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
  rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
  dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
  values = torch.zeros((args.num_steps, args.num_envs)).to(device)

  # TRY NOT TO MODIFY: start the game
  global_step = 0
  start_time = time.time()
  next_obs = torch.Tensor(envs.reset()).to(device)
  next_done = torch.zeros(args.num_envs).to(device)
  next_gru_state = torch.zeros(agent.gru.num_layers, args.num_envs,
                               agent.gru.hidden_size).to(device)
  num_updates = args.total_timesteps // args.batch_size

  for update in range(1, num_updates + 1):
    initial_gru_state = next_gru_state.clone()
    # Annealing the rate if instructed to do so.
    if args.anneal_lr:
      frac = 1.0 - (update - 1.0) / num_updates
      lrnow = frac * args.learning_rate
      optimizer.param_groups[0]["lr"] = lrnow

    for step in range(0, args.num_steps):
      global_step += 1 * args.num_envs
      obs[step] = next_obs
      dones[step] = next_done

      # ALGO LOGIC: action logic
      with torch.no_grad():
        action, logprob, _, value, next_gru_state = agent.get_action_and_value(
            next_obs, next_gru_state, next_done)
        values[step] = value.flatten()
      actions[step] = action
      logprobs[step] = logprob

      # TRY NOT TO MODIFY: execute the game and log data.
      next_obs, reward, done, info = envs.step(action.cpu().numpy())
      rewards[step] = torch.tensor(reward).to(device).view(-1)
      next_obs, next_done = torch.Tensor(next_obs).to(
          device), torch.Tensor(done).to(device)
      next_value = agent.get_value(next_obs,
                                   next_gru_state,
                                   next_done).reshape(1, -1)

      for item in info:
        if "episode" in item.keys():
          print(
              f"global_step={global_step}, episodic_return={item['episode']['r']}")
          writer.add_scalar("charts/episodic_return",
                            item["episode"]["r"], global_step)
          writer.add_scalar("charts/episodic_length",
                            item["episode"]["l"], global_step)
          break

    # bootstrap value if not done
    advantages, returns = estimate_advantage(
        rewards=rewards,
        values=values,
        dones=dones,
        last_value=next_value,
        last_done=next_done,
        n_steps=args.num_steps,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
    )

    # flatten the batch
    batch_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
    batch_log_probs = logprobs.reshape(-1)
    batch_actions = actions.reshape((-1,) + envs.single_action_space.shape)
    batch_dones = dones.reshape(-1)
    batch_advantages = advantages.reshape(-1)
    batch_returns = returns.reshape(-1)
    batch_values = values.reshape(-1)

    # Optimizing the policy and value network
    assert args.num_envs % args.num_minibatches == 0

    envs_per_batch = args.num_envs // args.num_minibatches
    env_inds = np.arange(args.num_envs)
    flat_inds = np.arange(args.batch_size).reshape(
        args.num_steps, args.num_envs)
    for epoch in range(args.update_epochs):
      np.random.shuffle(env_inds)
      for start in range(0, args.num_envs, envs_per_batch):
        end = start + envs_per_batch
        minibatch_env_inds = env_inds[start:end]
        minibatch_inds = flat_inds[:, minibatch_env_inds].ravel()

        _, new_log_probs, entropy, new_values, _ = agent.get_action_and_value(
            batch_obs[minibatch_inds],
            initial_gru_state[:, minibatch_env_inds],
            batch_dones[minibatch_inds],
            batch_actions.long()[minibatch_inds])
        loss, info = ppo_loss(
            batch_advantages=batch_advantages[minibatch_inds],
            batch_log_probs=batch_log_probs[minibatch_inds],
            batch_values=batch_values[minibatch_inds],
            batch_returns=batch_returns[minibatch_inds],
            new_log_probs=new_log_probs,
            new_values=new_values.flatten(),
            entropy=entropy,
            clip_range=args.clip_coef,
            value_coef=args.vf_coef,
            entropy_coef=args.ent_coef,
            normalize_advantage=args.norm_adv,
        )

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
        optimizer.step()

      if args.target_kl is not None:
        if info['approx_kl'] > args.target_kl:
          break

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    for key, value in info.items():
      writer.add_scalar(key, value, global_step)

  envs.close()
  writer.close()
