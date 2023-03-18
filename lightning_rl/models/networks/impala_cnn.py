from typing import List
import torch.nn as nn
import torch
import numpy as np


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


class ImpalaNetwork(nn.Module):
  """The full impala network from Espeholt et al. 2018."""

  def __init__(self, c: int, h: int, w: int,
               hidden_channels: List[int] = [16, 32, 32],
               out_features: int = 256) -> None:
    super(ImpalaNetwork, self).__init__()
    # Add the impala blocks to the network
    impala_blocks = [ImpalaBlock(
        in_channels=c, out_channels=hidden_channels[0])]
    for i in range(1, len(hidden_channels)):
      impala_blocks.append(
          ImpalaBlock(in_channels=hidden_channels[i-1], out_channels=hidden_channels[i]))
    self.impala_blocks = nn.Sequential(*impala_blocks)

    out_shape = self.impala_blocks(torch.zeros(c, h, w)).shape
    self.fc = nn.Linear(in_features=np.prod(
        out_shape), out_features=out_features)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.impala_blocks(x)
    x = nn.functional.relu(x)
    x = x.view(x.shape[0], -1)
    x = self.fc(x)
    x = nn.functional.relu(x)
    return x
  
if __name__ == '__main__':
  im = torch.zeros(1, 3, 84, 84)
  net = ImpalaNetwork(*im.shape[1:])
  print(net)
  net(im)
