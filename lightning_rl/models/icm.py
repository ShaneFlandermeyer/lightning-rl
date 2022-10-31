from typing import Optional, Tuple
import gym
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F


class ICM(pl.LightningModule):
  """
  An Intrinsic Curiosity Module (ICM) for self-supervised exploration.

  In Burda2018, the IDF and forward dynamics networks were heads on top of the embedding network with extra fully-connected layers of dimensionality 512.

  See: https://arxiv.org/pdf/1705.05363.pdf
  """

  def __init__(self,
               observation_space: gym.Space,
               action_space: gym.Space,
               eta: float = 0.01,
               seed: Optional[int] = None,
               ) -> None:
    super().__init__()
    self.observation_space = observation_space
    self.action_space = action_space
    self.eta = eta
    self.seed = seed

    if seed is not None:
      pl.seed_everything(seed)

  def forward(self, inputs: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
    """
    Run the ICM to generate the outputs for the forward/inverse dynamics models.

    Parameters
    ----------
    inputs : Tuple[torch.Tensor]
      Tuple containing the state, next state, and action in the original features space.

    Returns
    -------
    Tuple[torch.Tensor]
      Tuple containing the actual and predicted next state (in the embedded feature space) and the action predicted by the inverse model.


    Raises
    ------
    NotImplementedError
        _description_
    """
    raise NotImplementedError

  def compute_intrinsic_reward(self, 
                               state: torch.Tensor, 
                               next_state: torch.Tensor, 
                               action: torch.Tensor):
    """
    Compute the intrinsic reward according to equation (6) in the paper. The reward is proportional to the scaled MSE between the predicted and actual next state in the embedded feature space.

    Parameters
    ----------
    state : torch.Tensor
        Current state in the original feature space.
    next_state : torch.Tensor
        Next state in the original feature space.
    action : torch.Tensor
        Action taken in the current state.

    Returns
    -------
    torch.Tensor
        Intrinsic reward.
    """
    
    # Convert action into one-hot encoding
    # TODO: This only works for discrete action spaces
    # TODO: This assumes a non-vectorized environment
    action_onehot = torch.FloatTensor(
        int(action.item()), self.action_space.n).to(self.device)
    action_onehot.zero_()
    action_onehot.scatter_(1, action, 1)

    # Convert the actual next state and predicted next state in the embedded feature space, along with the action predicted by the inverse model.
    actual_next_state_feature, predicted_next_state_feature, predicted_action = self.forward(
        [state, next_state, action_onehot])

    # Compute intrinsic reward
    intrinsic_reward = self.eta * \
        F.MSELoss(actual_next_state_feature,
                  predicted_next_state_feature, reduction='none').mean(dim=-1)
    return intrinsic_reward.cpu().numpy()
