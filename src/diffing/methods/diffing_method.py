from abc import ABC, abstractmethod
from omegaconf import DictConfig


class DiffingMethod(ABC):
    @abstractmethod
    def __init__(self, cfg: DictConfig):
        pass

    @abstractmethod
    def run(self):
        pass


    @abstractmethod
    def visualize(self):
        pass

    @property
    def verbose(self) -> bool:
        """Check if verbose logging is enabled."""
        return getattr(self.cfg, 'verbose', False)
