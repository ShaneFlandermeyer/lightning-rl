import numpy as np
import torch

def ortho_init(layer, std=np.sqrt(2), bias_const=0.0):
  torch.nn.init.orthogonal_(layer.weight, std)
  torch.nn.init.constant_(layer.bias, bias_const)
  return layer

def norm_init(layer, norm_dim, scale=1.0):
  with torch.no_grad():
    layer.weight.data *= scale / layer.weight.norm(dim=norm_dim, p=2, keepdim=True)
    layer.bias *= 0
  return layer