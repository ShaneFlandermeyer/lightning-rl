# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import pybullet_envs  # noqa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from lightning_rl.common.buffers import ReplayBuffer

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
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    
    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="HopperBulletEnv-v0",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--exploration-noise", type=float, default=0.1,
        help="the scale of the exploration noise")
    parser.add_argument("--learning_starts", type=int, default=5e3,
        help="timestep to start learning")
    parser.add_argument("--policy-lr", type=float, default=3e-4,
        help="the learning rate of the policy network optimizer")
    parser.add_argument("--q-lr", type=float, default=3e-4,
        help="the learning rate of the Q network optimizer")
    parser.add_argument("--policy-frequency", type=int, default=2,
        help="the frequency of training the policy (delayed)")
    parser.add_argument("--target-network-frequency", type=int, default=1,
        help="the frequency of updates for the target networks")
    parser.add_argument("--noise-clip", type=float, default=0.5,
        help="noise clip parameter of the Target Policy Smoothing Regularization")
    parser.add_argument("--alpha", type=float, default=0.2,
        help="Entropy regularization coefficient.")
    parser.add_argument("--autotune", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
        help="automatic tuning of the entropy coefficient")
    args = parser.parse_args()
    # fmt: on
    return args
  
def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk
  
class SoftQNetwork(nn.Module):
  def __init__(self, env: gym.vector.VectorEnv):
    super().__init__()
    self.fc1 = nn.Linear(np.prod(env.single_observation_space.shape) + np.prod(env.single_action_space.shape), 256)
    self.fc2 = nn.Linear(256, 256)
    self.fc3 = nn.Linear(256, 1)
  
  def forward(self, x: torch.Tensor, a: torch.Tensor):
    x = torch.cat([x, a], dim=1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x
  
LOG_STD_MIN = 2
LOG_STD_MAX = -5

class Actor(nn.Module):
  def __init__(self, env: gym.vector.VectorEnv):
    super().__init__()
    self.fc1 = nn.Linear(np.prod(env.single_observation_space.shape), 256)
    self.fc2 = nn.Linear(256, 256)
    self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
    self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
    # Action rescaling
    self.register_buffer(
      "action_scale", torch.tensor((env.single_action_space.high - env.single_action_space.low) / 2.0, dtype=torch.float32)
    )
    self.register_buffer(
      "action_bias", torch.tensor((env.single_action_space.high + env.single_action_space.low) / 2.0, dtype=torch.float32)
    )
  
  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    mean = self.fc_mean(x)
    logstd = self.fc_logstd(x)
    logstd = torch.tanh(logstd)
    logstd = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (logstd + 1)
    
    return mean, logstd
  
  def get_action(self, x):
    mean, logstd = self(x)
    std = logstd.exp()
    normal = torch.distributions.Normal(mean, std)
    xt = normal.rsample() # For reparameterization trick (mean + std * N(0,1))
    yt = torch.tanh(xt)
    action = yt * self.action_scale + self.action_bias
    logprob = normal.log_prob(xt)
    # Enforcing action bound
    logprob -= torch.log(self.action_scale * (1 - yt.pow(2)) + 1e-6)
    logprob = logprob.sum(1, keepdim=True)
    mean = torch.tanh(mean) * self.action_scale + self.action_bias
    return action, logprob, mean
    
  
if __name__ == '__main__':
  args = parse_args()
  run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
  writer = SummaryWriter(f"runs/{run_name}")
  writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
  
  envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
  
  # TRY NOT TO MODIFY: seeding
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.backends.cudnn.deterministic = args.torch_deterministic
  device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
  # env setup
  envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
  assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
  
  max_action = float(envs.single_action_space.high[0])
  
  actor = Actor(envs).to(device)
  qf1 = SoftQNetwork(envs).to(device)
  qf2 = SoftQNetwork(envs).to(device)
  qf1_target = SoftQNetwork(envs).to(device)
  qf2_target = SoftQNetwork(envs).to(device)
  q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
  actor_optimizer = optim.Adam(actor.parameters(), lr=args.policy_lr)
  
  # Automatic entropy tuning
  if args.autotune:
    target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha = log_alpha.exp().item()
    alpha_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
  else:
    alpha = args.alpha
    
  envs.single_observation_space.dtype = np.float32
  
  # Create the replay buffer
  # TODO: Getting some instabilities in the training, probably because timeouts are not handled properly in the replay buffer/bellman update. This shouldn't matter in newer versions of gym since the termination/truncation signals are separated
  rb = ReplayBuffer(args.buffer_size)
  rb.create_tensor('observations', envs.single_observation_space.shape, envs.single_observation_space.dtype)
  rb.create_tensor('next_observations', envs.single_observation_space.shape, envs.single_observation_space.dtype)
  rb.create_tensor('actions', envs.single_action_space.shape, envs.single_action_space.dtype)
  rb.create_tensor('rewards', (1,), np.float32)
  rb.create_tensor('dones', (1,), bool)
  rb.create_tensor('infos', (1,), dict)
  
  start_time = time.time()
  
  # TRY NOT TO MODIFY: start the game
  obs = envs.reset()
  for global_step in range(args.total_timesteps):
    # ALSO LOGIC: put action logic here
    if global_step < args.learning_starts:
      actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
    else:
      actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
      actions = actions.detach().cpu().numpy()
      
    # TRY NOT TO MODIFY: Execute the game and log data
    next_obs, rewards, dones, infos = envs.step(actions)
    
    # TRY NOT TO MODIFY: record rewards for plotting purposes
    for info in infos:
        if "episode" in info.keys():
            print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
            writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
            writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
            break
    # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
    real_next_obs = next_obs.copy()
    for idx, d in enumerate(dones):
        if d:
            real_next_obs[idx] = infos[idx]["terminal_observation"]
            
    rb.add(observations=obs, 
           next_observations=real_next_obs, 
           actions=actions, 
           rewards=rewards, 
           dones=dones, 
           infos=infos)
    
    # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
    obs = next_obs
    
    # ALGO LOGIC: training
    if global_step > args.learning_starts:
      # Sample the replay buffer and convert to tensors
      data = rb.sample(args.batch_size)
      observations = torch.Tensor(data.observations).to(device)
      next_observations = torch.Tensor(data.next_observations).to(device)
      actions = torch.Tensor(data.actions).to(device)
      rewards = torch.Tensor(data.rewards).to(device)
      dones = torch.Tensor(data.dones).to(device)
      
      # Train critic
      with torch.no_grad():
        next_state_actions, next_state_logprobs, _ = actor.get_action(next_observations)
        q1_next_target = qf1_target(next_observations, next_state_actions)
        q2_next_target = qf2_target(next_observations, next_state_actions)
        min_q_next_target = torch.min(q1_next_target, q2_next_target) - alpha * next_state_logprobs
        next_q_value = rewards.flatten() + (1 - dones.flatten()) * args.gamma * min_q_next_target.view(-1)
        
      q1_action_values = qf1(observations, actions).view(-1)
      q2_action_values = qf2(observations, actions).view(-1)
      q1_loss = F.mse_loss(q1_action_values, next_q_value)
      q2_loss = F.mse_loss(q2_action_values, next_q_value)
      q_loss = q1_loss + q2_loss
      
      q_optimizer.zero_grad()
      q_loss.backward()
      q_optimizer.step()
      
      # Train actor (every policy_frequency steps)
      if global_step % args.policy_frequency == 0:
        for _ in range(args.policy_frequency):
          a, logprob_a, _ = actor.get_action(observations)
          q1_a = qf1(observations, a)
          q2_a = qf2(observations, a)
          min_q_a = torch.min(q1_a, q2_a).view(-1)
          actor_loss = ((alpha * logprob_a) - min_q_a).mean()
          
          actor_optimizer.zero_grad()
          actor_loss.backward()
          actor_optimizer.step()
          
          if args.autotune:
            with torch.no_grad():
              _, logprob, _ = actor.get_action(observations)
            alpha_loss = (-log_alpha * (logprob + target_entropy)).mean()
            
            alpha_optimizer.zero_grad()
            alpha_loss.backward()
            alpha_optimizer.step()
            alpha = log_alpha.exp().item()
    
      # update the target networks
      if global_step % args.target_network_frequency == 0:
        for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
          target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
        for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
          target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
      
      # Write statistics to tensorboard
      if global_step % 100 == 0:
        writer.add_scalar("losses/qf1_values", q1_action_values.mean().item(), global_step)
        writer.add_scalar("losses/qf2_values", q2_action_values.mean().item(), global_step)
        writer.add_scalar("losses/qf1_loss", q1_loss.item(), global_step)
        writer.add_scalar("losses/qf2_loss", q2_loss.item(), global_step)
        writer.add_scalar("losses/qf_loss", q_loss.item() / 2.0, global_step)
        writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
        writer.add_scalar("losses/alpha", alpha, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        if args.autotune:
            writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)