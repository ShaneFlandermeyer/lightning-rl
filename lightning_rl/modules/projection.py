from typing import Callable
import torch.nn as nn
from lightning_rl.common.layer_init import delta_ortho_init


class LinearProjection(nn.Module):
  def __init__(self,
               in_dim: int,
               out_dim: int,
               weight_init: Callable = delta_ortho_init):
    super().__init__()
    self.projection = nn.Linear(in_dim, out_dim)
    self.out_dim = out_dim
    self.apply(weight_init)

  def forward(self, x):
    return self.projection(x)


class NormedProjection(nn.Module):
  def __init__(self,
               in_dim: int,
               out_dim: int,
               weight_init: Callable = delta_ortho_init):
    super().__init__()
    self.out_dim = out_dim
    self.projection = nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.LayerNorm(out_dim),
        nn.Tanh()
    )
    self.out_dim = out_dim
    self.apply(weight_init)

  def forward(self, x):
    return self.projection(x)
