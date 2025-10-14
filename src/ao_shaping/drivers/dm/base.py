from abc import ABC, abstractmethod


class DM(ABC):
    channel: int

    @abstractmethod
    def transform(self, cmd):
        pass

    @abstractmethod
    def send(self, cmd):
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
