# Lightning RL

Lightning RL implements several reinforcement learning algorithms using [Pytorch Lightning](https://www.pytorchlightning.ai/) as a backend. LightningRL abstracts away the data collection loop, making algorithm implementations concise and easy to maintain (for most algorithms, only the ```training_step()``` method for computing loss must be implemented). Pytorch Lightning makes it easy to distribute training to multiple CPUs, GPUs, and/or compute nodes so that research experiments can be scaled up.

I got the idea for this repository from [Lightning Baselines3](https://github.com/HenryJia/lightning-baselines3), and much of the code for "standard" algorithms like DQN and A2C is taken directly from there. However, Lightning RL will diverge into its own thing as I build it up for my personal research.

## Dependencies

To install the dependencies for Lightning RL, run the following from the root directory of this project

```bash
pip install -e .
```

## Examples

- Example usages for each algorithm can be found in the ```examples/``` directory.

## Algorithms

- [x] DQN
- [X] A2C
- [ ] PPO
- [ ] ICM