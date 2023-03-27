from typing import Callable, List, Optional
import torch
import torch.nn as nn
import numpy as np
from lightning_rl.common.layer_init import delta_ortho_init

from lightning_rl.common.utils import get_out_shape


class NatureCNN(nn.Module):
  """CNN architecture from the DQN Nature paper."""

  def __init__(self,
               c: int, h: int, w: int,
               layer_init: Optional[Callable] = None):
    super().__init__()
    
    if layer_init is None:
      layer_init = lambda layer : layer
      
    self.convs = nn.ModuleList([
      layer_init(nn.Conv2d(c, 32, kernel_size=8, stride=4)),
      layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
      layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
    ])
    self.out_dim = get_out_shape((c, h, w), self.convs)

  def forward(self, x):
    for conv in self.convs:
      x = torch.relu(conv(x))
      
    x = x.view(x.shape[0], -1)
    return x
    
  def copy_conv_weights_from(self, source: nn.Module):
    for i in range(len(self.convs)):
      self.convs[i].weight.data = source.convs[i].weight.data
      self.convs[i].bias.data = source.convs[i].bias.data
      
class SACAECNN(nn.Module):
  """Convolutional encoder for image-based observations."""
  def __init__(self, c: int, h: int, w: int) -> None:
    super().__init__()
    
    self.input_shape = (c, h, w)
    self.n_layers = 4
    self.n_filters = 32
    
    self.convs = [nn.Conv2d(c, self.n_filters, 3, stride=2)]
    for _ in range(1, self.n_layers):
      self.convs.append(nn.Conv2d(self.n_filters, self.n_filters, 3, stride=1))
    self.convs = nn.ModuleList(self.convs)
    
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Apply convolutional sequence
    for i in range(len(self.convs)):
      x = self.convs[i](x)
      x = torch.relu(x)
      
    # Flatten
    h = x.view(x.size(0), -1)
    return h
  
  def copy_conv_weights_from(self, source: nn.Module):
    for i in range(len(self.convs)):
      self.convs[i].weight = source.convs[i].weight
      self.convs[i].bias = source.convs[i].bias
      
################################################################################
# Impala CNN
################################################################################
class ResidualBlock(nn.Module):
  """A basic two-layer residual block."""

  def __init__(self, in_channels: int) -> None:
    super(ResidualBlock, self).__init__()
    self.conv1 = nn.Conv2d(
        in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(
        in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    out = nn.functional.relu(x)
    out = self.conv1(out)
    out = nn.functional.relu(out)
    out = self.conv2(out)
    return out + x
  
  def copy_conv_weights_from(self, source: nn.Module):
    self.conv1.weight.data = source.conv1.weight.data
    self.conv1.bias.data = source.conv1.bias.data
    self.conv2.weight.data = source.conv2.weight.data
    self.conv2.bias.data = source.conv2.bias.data


class ImpalaBlock(nn.Module):
  """
  An "impala block" as described in Espeholt et al. 2018.

  Contains two residual blocks and a pooled convolutional layer
  """

  def __init__(self, in_channels: int, out_channels: int) -> None:
    super(ImpalaBlock, self).__init__()
    self.conv = nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=3, stride=1, padding=1)
    self.res1 = ResidualBlock(out_channels)
    self.res2 = ResidualBlock(out_channels)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.conv(x)
    x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)
    x = self.res1(x)
    x = self.res2(x)
    return x
  
  def copy_conv_weights_from(self, source: nn.Module):
    self.conv.weight.data = source.conv.weight.data
    self.conv.bias.data = source.conv.bias.data
    self.res1.copy_conv_weights_from(source.res1)
    self.res2.copy_conv_weights_from(source.res2)


class ImpalaCNN(nn.Module):
  """The full impala network from Espeholt et al. 2018."""

  def __init__(self, c: int, h: int, w: int,
               hidden_channels: List[int] = [16, 32, 32],
               ) -> None:
    super(ImpalaCNN, self).__init__()
    # Add the impala blocks to the network
    impala_blocks = [ImpalaBlock(
        in_channels=c, out_channels=hidden_channels[0])]
    for i in range(1, len(hidden_channels)):
      impala_blocks.append(
          ImpalaBlock(in_channels=hidden_channels[i-1], out_channels=hidden_channels[i]))
    self.impala_blocks = nn.ModuleList(impala_blocks)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    for i in range(len(self.impala_blocks)):
      x = self.impala_blocks[i](x)
    h = nn.functional.relu(x)
    
    h = h.view(h.size(0), -1)
    
    return h
  
  def copy_conv_weights_from(self, source: nn.Module):
    for i in range(len(self.impala_blocks)):
      self.impala_blocks[i].copy_conv_weights_from(source.impala_blocks[i])

