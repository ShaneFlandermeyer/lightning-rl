from typing import List
import torch.nn as nn
import torch
import numpy as np
from lightning_rl.common.utils import get_out_shape

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


class ImpalaEncoder(nn.Module):
  """The full impala network from Espeholt et al. 2018."""

  def __init__(self, c: int, h: int, w: int,
               hidden_channels: List[int] = [16, 32, 32],
               out_features: int = 256) -> None:
    super(ImpalaEncoder, self).__init__()
    # Add the impala blocks to the network
    impala_blocks = [ImpalaBlock(
        in_channels=c, out_channels=hidden_channels[0])]
    for i in range(1, len(hidden_channels)):
      impala_blocks.append(
          ImpalaBlock(in_channels=hidden_channels[i-1], out_channels=hidden_channels[i]))
    self.impala_blocks = nn.ModuleList(impala_blocks)

    out_shape = get_out_shape(self.impala_blocks, (c, h, w))
    self.fc = nn.Linear(in_features=np.prod(
        out_shape), out_features=out_features)

  def forward(self, x: torch.Tensor, detach: bool = False) -> torch.Tensor:
    for i in range(len(self.impala_blocks)):
      x = self.impala_blocks[i](x)
    h = nn.functional.relu(x)
    
    if detach:
      h = h.detach()
      
    h = h.view(h.shape[0], -1)
    h = self.fc(h)
    h = nn.functional.relu(h)
    return h
  
  def copy_conv_weights_from(self, source: nn.Module):
    for i in range(len(self.impala_blocks)):
      self.impala_blocks[i].copy_conv_weights_from(source.impala_blocks[i])

if __name__ == '__main__':
  im = torch.zeros(1, 3, 84, 84)
  net = ImpalaEncoder(*im.shape[1:])
  print(net)
  net(im)
