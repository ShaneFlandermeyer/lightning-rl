from typing import Callable, Optional
import torch
import torch.nn as nn
import numpy as np

from lightning_rl.common.utils import get_out_shape


class NatureEncoder(nn.Module):
  """CNN architecture from the DQN Nature paper."""

  def __init__(self,
               c: int, h: int, w: int,
               out_features: int = 512,
               layer_init: Optional[Callable] = None):
    super().__init__()
    
    if layer_init is None:
      layer_init = lambda layer : layer
    self.convs = nn.ModuleList([
      layer_init(nn.Conv2d(c, 32, kernel_size=8, stride=4)),
      layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
      layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
    ])
    
    hidden_shape = get_out_shape(self.convs, (c, h, w))
    self.fc = nn.Linear(np.prod(hidden_shape), out_features)

  def forward(self, x, detach: bool = False):
    for conv in self.convs:
      x = torch.relu(conv(x))
    
    if detach:
      x = x.detach() 
      
    x = x.view(x.shape[0], -1)
    x = self.fc(x)
    return x
    
  def copy_conv_weights_from(self, source: nn.Module):
    for i in range(len(self.convs)):
      self.convs[i].weight.data = source.convs[i].weight.data
      self.convs[i].bias.data = source.convs[i].bias.data
  
  
