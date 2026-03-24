from abc import ABC, abstractmethod

import numpy as np

class DM(ABC):
    channel: int

    @abstractmethod
    def transform(self, cmd) -> np.ndarray:
        pass

    @abstractmethod
    def send(self, cmd) -> np.ndarray:
        pass

    @abstractmethod
    def open(self):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def get_actuator_positions(self):
        pass

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
