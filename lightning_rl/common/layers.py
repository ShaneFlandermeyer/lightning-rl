import torch.nn as nn
import numpy as np
import torch
from torch import nn, Tensor
from typing import Tuple
import math
import torch.nn.functional as F


class NoisyLinear(nn.Linear):
  """
  A noisy linear layer.

  The weights and biases of this layer are normally distributed with a mean
  equal to their original values and a user-specified standard
  deviation/variance. The standard deviation of each weight and bias is a
  trainable Pytorch parameter.
  """

  def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
    super().__init__(in_features, out_features, bias=bias)
    # Create a matrix for the standard deviation of each weight in the
    # fully-connected layer. To make this a trainable parameter, we have to add
    # it to the object as a nn.Parameter
    weights = torch.full((out_features, in_features), sigma_init)
    self.sigma_weight = nn.Parameter(weights)
    epsilon_weight = torch.zeros(out_features, in_features)
    self.register_buffer("epsilon_weight", epsilon_weight)
    if bias:
      # Create a learnable parameter controlling the standard deviation of the
      # bias of each connection to the output
      weights = torch.full((out_features,), sigma_init)
      self.sigma_bias = nn.Parameter(weights)
      epsilon_bias = torch.zeros(out_features)
      self.register_buffer("epsilon_bias", epsilon_bias)
    # Initialize the weights and biases of the layer
    self.reset_parameters()

  def reset_parameters(self):
    """
    Initialize the weights and biases in the layer
    """
    std = math.sqrt(3 / self.in_features)
    self.weight.data.uniform_(-std, std)
    self.bias.data.uniform_(-std, std)

  def forward(self, input) -> Tensor:
    """
    Forward pass through the layer.

    The weights and biases of the layer are scaled such that the mean is the
    value of the original layer parameter and the standard deviation is added on
    top of this, sampled from a normal distribution

    Args:
        input (Tensor): Input Tensor
    Returns:
        Tensor
    """
    self.epsilon_weight.normal_()
    bias = self.bias
    if bias is not None:
      self.epsilon_bias.normal_()
      bias = bias + self.sigma_bias * self.epsilon_bias.data
    v = self.weight + self.sigma_weight * self.epsilon_weight.data
    return F.linear(input, v, bias)
