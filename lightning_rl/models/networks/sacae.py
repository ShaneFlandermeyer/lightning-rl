from typing import Tuple
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from lightning_rl.common.utils import get_out_shape
import gymnasium as gym


class SACAEEncoder(nn.Module):
  """Convolutional encoder for image-based observations."""
  def __init__(self, c: int, h: int, w: int,
               out_features: int,
               ) -> None:
    super().__init__()
    
    self.input_shape = (c, h, w)
    self.feature_dim = out_features
    self.n_layers = 4
    self.n_filters = 32
    
    self.convs = [nn.Conv2d(c, self.n_filters, 3, stride=2)]
    for _ in range(1, self.n_layers):
      self.convs.append(nn.Conv2d(self.n_filters, self.n_filters, 3, stride=1))
    self.convs = nn.ModuleList(self.convs)
    
    conv_out_shape = get_out_shape(self.convs, self.input_shape)
    self.fc = nn.Sequential(
      nn.Linear(np.prod(conv_out_shape), out_features),
      nn.LayerNorm(out_features),
    )
    
  def forward_conv(self, x: torch.Tensor) -> torch.Tensor:
    # Apply convolutional sequence
    for i in range(len(self.convs)):
      x = self.convs[i](x)
      x = torch.relu(x)
      
    # Flatten
    h = x.view(x.size(0), -1)
    return h
  
  def forward(self, x: torch.Tensor, detach: bool = False) -> torch.Tensor:
    # Apply convolutional sequence
    h = self.forward_conv(x)
    
    if detach:
      h = h.detach()
    
    out = self.fc(h)
    out = torch.tanh(out)
    return out
  
  def copy_conv_weights_from(self, source: nn.Module):
    for i in range(len(self.convs)):
      self.convs[i].weight = source.convs[i].weight
      self.convs[i].bias = source.convs[i].bias
      