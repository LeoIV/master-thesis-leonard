from abc import ABC, abstractmethod
from typing import Union, Tuple


class ModelWrapper(ABC):

    @abstractmethod
    def _build(self, input_dim: Union[Tuple[int, int], Tuple[int, int, int]]):
        raise NotImplementedError

    @abstractmethod
    def compile(self, learning_rate: float, r_loss_factor: float):
        raise NotImplementedError

    @abstractmethod
    def save(self, folder: str):
        raise NotImplementedError

    @abstractmethod
    def load_weights(self, filepath: str):
        raise NotImplementedError

    @abstractmethod
    def train(self, training_data, batch_size, epochs, run_folder, print_every_n_batches=100, initial_epoch=0,
              lr_decay=1, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def plot_model(self, run_folder: str):
        raise NotImplementedError
