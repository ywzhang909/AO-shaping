"""NI DAQ-based ADC voltage acquisition driver.

Reads analog voltage signals from NI DAQ devices using the nidaqmx library.
Supports configurable channels, sample rate, and sample count.

The reference implementation in ``adc_dm_adam.py`` uses:

.. code-block:: python

    with nidaqmx.Task() as task:
        task.ai_channels.add_ai_voltage_chan("Dev1/ai0")
        task.timing.cfg_samp_clk_timing(
            rate=sample_rate,
            sample_mode=AcquisitionType.HW_TIMED_SINGLE_POINT,
            samps_per_chan=samples_per_channel,
        )
        task.start()
        data = task.read(number_of_samples_per_channel=samples_per_channel)
        return np.mean(data)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from ao_shaping.drivers.device_base import Device, DeviceError, DeviceState, DeviceType

# ---------------------------------------------------------------------------
# Optional nidaqmx import — gracefully degrade when the SDK is not installed
# ---------------------------------------------------------------------------
try:
    import nidaqmx
    from nidaqmx.constants import AcquisitionType

    NIDAQMX_AVAILABLE = True
except ImportError:
    NIDAQMX_AVAILABLE = False
    nidaqmx = None  # type: ignore[assignment]
    AcquisitionType = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------
class NidaqADCError(DeviceError):
    """Base exception for NidaqADC errors."""


class NidaqADCNotFoundError(NidaqADCError):
    """Raised when the NI DAQ device is not found."""


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
class NidaqADC(Device):
    """NI DAQ-based analog voltage acquisition driver.

    Parameters
    ----------
    device_name : str
        NI DAQ device name (e.g. ``"Dev1"``).
    channel : str
        Analog input channel (e.g. ``"ai0"``).
    sample_rate : int
        Acquisition sample rate in Hz.
    samples_per_channel : int
        Number of samples to read per ``read()`` call.
    device_id : str
        Optional unique identifier (auto-generated if empty).
    """

    device_type = DeviceType.OTHER  # ADC is not in the standard enum
    manufacturer = "National Instruments"
    model = "NI DAQ (nidaqmx)"
    version = "1.0.0"

    def __init__(
        self,
        device_name: str = "Dev1",
        channel: str = "ai0",
        sample_rate: int = 5000,
        samples_per_channel: int = 10,
        device_id: str = "",
    ) -> None:
        super().__init__(device_id)

        self._device_name = device_name
        self._channel = channel
        self._sample_rate = sample_rate
        self._samples_per_channel = samples_per_channel

        self._task: Any = None
        self._virt_chan: str = f"{device_name}/{channel}"

        self._register_parameters()

    # ------------------------------------------------------------------
    # Parameter registration
    # ------------------------------------------------------------------
    def _register_parameters(self) -> None:
        self.register_parameter(
            "device_name",
            self._device_name,
            description="NI DAQ device name (e.g. Dev1)",
        )
        self.register_parameter(
            "channel",
            self._channel,
            description="Analog input channel (e.g. ai0)",
        )
        self.register_parameter(
            "sample_rate",
            self._sample_rate,
            min_value=1,
            max_value=1_000_000,
            unit="Hz",
            description="Acquisition sample rate",
        )
        self.register_parameter(
            "samples_per_channel",
            self._samples_per_channel,
            min_value=1,
            max_value=1_000_000,
            description="Number of samples per read call",
        )

    # ------------------------------------------------------------------
    # Device lifecycle
    # ------------------------------------------------------------------
    def open(self) -> None:
        """Open the NI DAQ task.

        Raises
        ------
        NidaqADCError
            If nidaqmx is not available or the task cannot be created.
        """
        if not NIDAQMX_AVAILABLE:
            logger.error("nidaqmx package not installed")
            raise NidaqADCError(
                "nidaqmx is not installed. "
                "Install it with: uv add nidaqmx  (or pip install nidaqmx)"
            )

        self._set_state(DeviceState.CONNECTING)
        logger.debug(
            "Opening ADC: {} @ {} Hz, {} samples/ch",
            self._virt_chan,
            self._sample_rate,
            self._samples_per_channel,
        )
        try:
            self._task = nidaqmx.Task()
            self._task.ai_channels.add_ai_voltage_chan(self._virt_chan)
            self._task.timing.cfg_samp_clk_timing(
                rate=self._sample_rate,
                sample_mode=AcquisitionType.HW_TIMED_SINGLE_POINT,
                samps_per_chan=self._samples_per_channel,
            )
            self._task.start()
            self._set_state(DeviceState.READY)
            logger.info(
                "ADC opened: {} @ {} Hz, {} samples/ch",
                self._virt_chan,
                self._sample_rate,
                self._samples_per_channel,
            )
        except Exception as e:
            logger.exception("Failed to open ADC task")
            self._set_state(DeviceState.ERROR, str(e))
            raise NidaqADCError(f"Failed to open ADC task: {e}") from e

    def close(self) -> None:
        """Stop and close the NI DAQ task."""
        if self._task is not None:
            try:
                self._task.stop()
                self._task.close()
                logger.debug("ADC task stopped and closed")
            except Exception as e:
                logger.warning("Error closing ADC task: {}", e)
            finally:
                self._task = None
        self._set_state(DeviceState.DISCONNECTED)
        logger.info("ADC closed")

    def is_connected(self) -> bool:
        """Check if the ADC task is active."""
        return self._task is not None and self._state == DeviceState.READY

    def get_hardware_info(self) -> dict[str, Any]:
        """Return hardware-specific information."""
        return {
            "device_name": self._device_name,
            "channel": self._channel,
            "virtual_channel": self._virt_chan,
            "sample_rate": self._sample_rate,
            "samples_per_channel": self._samples_per_channel,
            "nidaqmx_available": NIDAQMX_AVAILABLE,
        }

    # ------------------------------------------------------------------
    # Acquisition API
    # ------------------------------------------------------------------
    def read(self, samples: int | None = None) -> np.ndarray:
        """Read voltage samples from the ADC.

        Parameters
        ----------
        samples : int, optional
            Number of samples to read. Defaults to ``samples_per_channel``.

        Returns
        -------
        np.ndarray
            1-D array of voltage readings in volts.

        Raises
        ------
        NidaqADCError
            If the task is not open or the read fails.
        """
        if not self.is_connected() or self._task is None:
            raise NidaqADCError("ADC is not open. Call open() first.")

        n = samples or self._samples_per_channel
        logger.debug("Reading {} samples from {}", n, self._virt_chan)
        try:
            data = self._task.read(number_of_samples_per_channel=n)
            arr = np.asarray(data, dtype=np.float64)
            logger.debug("ADC read: {} samples, mean={:.6f} V", len(arr), np.mean(arr))
            return arr
        except Exception as e:
            logger.exception("ADC read failed")
            self._set_state(DeviceState.ERROR, str(e))
            raise NidaqADCError(f"ADC read failed: {e}") from e

    def read_mean(self, samples: int | None = None) -> float:
        """Read voltage samples and return the mean.

        Parameters
        ----------
        samples : int, optional
            Number of samples to read. Defaults to ``samples_per_channel``.

        Returns
        -------
        float
            Mean voltage in volts.
        """
        n = samples or self._samples_per_channel
        mean_val = float(np.mean(self.read(samples=n)))
        logger.debug("ADC read_mean: {:.6f} V ({} samples)", mean_val, n)
        return mean_val
