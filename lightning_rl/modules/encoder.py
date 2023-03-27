from typing import Callable, List, Optional
import torch
import torch.nn as nn
import numpy as np
from lightning_rl.common.layer_init import delta_ortho_init

from lightning_rl.common.utils import get_out_shape


class CNNEncoder(nn.Module):
  """Convolutional encoder of pixels observations."""

  def __init__(self, cnn: nn.Module, projection: nn.Module):
    super().__init__()
    self.cnn = cnn
    self.projection = projection
    self.out_dim = projection.out_dim

  def forward(self, x, detach: bool = False):
    x = self.cnn(x)
    if detach:
      x = x.detach()
    return self.projection(x)
