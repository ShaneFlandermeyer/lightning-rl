import numpy as np
import torch
import torch.nn as nn


def ortho_init(layer, std=np.sqrt(2), bias_const=0.0):
  torch.nn.init.orthogonal_(layer.weight, std)
  torch.nn.init.constant_(layer.bias, bias_const)
  return layer


def norm_init(layer, norm_dim, scale=1.0):
  with torch.no_grad():
    layer.weight.data *= scale / \
        layer.weight.norm(dim=norm_dim, p=2, keepdim=True)
    layer.bias *= 0
  return layer


def delta_ortho_init(m):
  """Custom weight init for Conv2D and Linear layers."""
  if isinstance(m, nn.Linear):
    nn.init.orthogonal_(m.weight.data)
    m.bias.data.fill_(0.0)
  elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
    # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
    assert m.weight.size(2) == m.weight.size(3)
    m.weight.data.fill_(0.0)
    m.bias.data.fill_(0.0)
    mid = m.weight.size(2) // 2
    gain = nn.init.calculate_gain('relu')
    nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)
