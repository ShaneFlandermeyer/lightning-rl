from typing import Tuple
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from lightning_rl.common.utils import get_out_shape
import gymnasium as gym


class EncoderCNN(nn.Module):
  """Convolutional encoder for image-based observations."""
  def __init__(self, 
               obs_shape: Tuple[int, ...], 
               feature_dim: int,
               n_layers: int = 4,
               n_filters: int = 32,
               ) -> None:
    super().__init__()
    
    assert len(obs_shape) == 3
    self.obs_shape = obs_shape
    self.feature_dim = feature_dim
    self.n_layers = n_layers
    
    self.convs = [nn.Conv2d(obs_shape[0], n_filters, 3, stride=2)]
    for _ in range(1, n_layers):
      self.convs.append(nn.Conv2d(n_filters, n_filters, 3, stride=1))
    self.convs = nn.ModuleList(self.convs)
    
    conv_out_shape = get_out_shape(self.convs, obs_shape)
    self.fc = nn.Sequential(
      nn.Linear(conv_out_shape, feature_dim),
      nn.LayerNorm(feature_dim),
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
      self.convs[i].weight = source.conv[i].weight
      self.convs[i].bias = source.conv[i].bias
      