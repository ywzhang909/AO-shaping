from abc import ABC, abstractmethod

import numpy as np


class DM(ABC):
    DM_NUM: int
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
    def get_actuator_positions(self) -> np.ndarray:
        ...

    def is_connected(self) -> bool:
        """Check if the DM is connected.

        Returns:
            True if connected, False otherwise.
        """
        return False

    def set_channel_voltage(self, channel: int, voltage: float) -> None:
        """Set voltage for a single channel.

        Args:
            channel: Channel index.
            voltage: Voltage value.

        Raises:
            NotImplementedError: If the DM does not support single-channel setting.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support set_channel_voltage"
        )

    def set_all_voltage_by_arr(self, voltages: np.ndarray) -> None:
        """Set all channel voltages by array.

        Args:
            voltages: Array of voltages for all channels.

        Raises:
            NotImplementedError: If the DM does not support array-based setting.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support set_all_voltage_by_arr"
        )

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
