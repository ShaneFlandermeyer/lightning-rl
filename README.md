# Lightning RL

Lightning RL implements several reinforcement learning algorithms using [Pytorch Lightning](https://www.pytorchlightning.ai/) as a backend. Many RL libraries have deep class structures that makes it difficult to modify or implement algorithms. The goal of this repository is to simplify the development process by keeping implementation details as transparent as possible. This adds a bit more boilerplate code before an algorithm can be used, but makes the implementations easier to understand.

In most cases, new classes of algorithms can be added by defining a loss functions in the ```training_step()``` method and defining the network architecture in a subclass. For example, PPO uses the "standard" on-policy data collection loop and a custom loss function. However, the data collection loop can also be modified by overriding the ```collect_rollouts()``` function (e.g., PPG requires separate policy/auxiliary phases during data collection).

Pytorch Lightning can also distribute training to multiple CPUs, GPUs, and/or compute nodes so that research experiments can be scaled up for tougher problems.

## Dependencies

To install the dependencies for Lightning RL, run the following from the root directory of this project

```bash
pip install -e .
```

## Examples

- Example usages for each algorithm can be found in the ```examples/``` directory.

## Algorithms

- [x] Deep Q-Networks (DQN)
- [x] Advantage Actor-Critic (A2C)
- [x] Proximal Policy Optimization (PPO)
- [x] Phasic Policy Gradient (PPG)
