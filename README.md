# Lightning RL

Lightning RL implements several reinforcement learning algorithms using [Pytorch Lightning](https://www.pytorchlightning.ai/) as a backend. LightningRL abstracts away the data collection loop, making algorithm implementations concise and easy to maintain (for most algorithms, only the ```training_step()``` method for computing loss must be implemented). Pytorch Lightning makes it easy to distribute training to multiple CPUs, GPUs, and/or compute nodes so that research experiments can be easily scaled up.

The overall structure of this repository is taken from [Lightning Baselines3](https://github.com/HenryJia/lightning-baselines3), and most of the code for "standard" algorithms like DQN and A2C is taken directly from there. However, Lightning RL will likely diverge as it is modified to fit my personal research needs.

## Dependencies

To install the dependencies for Lightning RL, run the following from the root directory of this project

```bash
pip install -e .
```

## Examples

- Example usages for each algorithm can be found in the ```examples/``` directory. 