from typing import Callable, Optional
import torch
import torch.nn as nn
import numpy as np


class NatureEncoder(nn.Module):
  """CNN architecture from the DQN Nature paper."""

  def __init__(self,
               c: int, h: int, w: int,
               out_features: int = 512,
               layer_init: Optional[Callable] = None):
    super().__init__()
    
    if layer_init is None:
      layer_init = lambda layer : layer
    self.conv1 = layer_init(nn.Conv2d(c, 32, kernel_size=8, stride=4))
    self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
    self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
    
    x = torch.zeros(c, h, w)
    hidden_shape = self.conv3(self.conv2(self.conv1(x))).shape
    self.fc = nn.Linear(np.prod(hidden_shape), out_features)

  def forward(self, x, detach: bool = False):
    h = self.conv1(x)
    h = torch.relu(h)
    h = self.conv2(h)
    h = torch.relu(h)
    h = self.conv3(h)
    h = torch.relu(h)
    
    if detach:
      h = h.detach()
      
    h = h.view(h.shape[0], -1)
    h = self.fc(h)
    return h
  
  def copy_conv_weights_from(self, source: nn.Module):
    self.conv1.weight.data = source.conv1.weight.data
    self.conv1.bias.data = source.conv1.bias.data
    self.conv2.weight.data = source.conv2.weight.data
    self.conv2.bias.data = source.conv2.bias.data
    self.conv3.weight.data = source.conv3.weight.data
    self.conv3.bias.data = source.conv3.bias.data
  
  
