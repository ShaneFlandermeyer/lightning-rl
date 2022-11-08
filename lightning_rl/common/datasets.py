from torch.utils.data import IterableDataset
import typing

class OnPolicyDataset(IterableDataset):
  
  def __init__(self, 
               rollout_generator: typing.Callable,
               ) -> None:
    self.rollout_generator = rollout_generator
  
  def __iter__(self) -> typing.Iterator:
    iterator = self.rollout_generator()
    return iterator
      
  