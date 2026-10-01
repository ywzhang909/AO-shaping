"""Shared abstract contract for wavefront-sensor drivers.

The WFS family was the only device family without a shared base class: the real
driver extended :class:`~ao_shaping.drivers.device_base.Device` directly, so
there was no contract a simulated WFS could implement to be a drop-in
replacement. The camera family solves this with ``BaseCamera`` and the DM family
with ``DM``; this module closes the same gap for WFS.

The abstract surface is exactly the set the optimizers and runners call
(verified by grepping ``optimizer/`` and ``runners/``), so implementing it is
sufficient to run any of them against a simulated sensor.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ao_shaping.drivers.device_base import Device, DeviceType
from ao_shaping.utils.wavefront.zernike_calc import (
    calc_n_zernike_terms as _canonical_calc_n_zernike_terms,
)


class BaseWFS(Device, ABC):
    """Abstract base for wavefront sensors (real or simulated)."""

    device_type: DeviceType = DeviceType.WFS
    manufacturer: str = "Unknown"
    model: str = "BaseWFS"

    def __init__(self, device_id: str = "") -> None:
        super().__init__(device_id)
        self._register_wfs_parameters()

    def _register_wfs_parameters(self) -> None:
        """Register the parameters every WFS driver must expose."""
        self.register_parameter(
            "exposure_time_ms",
            default_value=0.0,
            min_value=0.0,
            max_value=70.0,
            unit="ms",
            description="WFS camera exposure time; 0 selects auto exposure.",
        )
        self.register_parameter(
            "remove_tilt",
            default_value=False,
            unit="",
            description="Remove tip/tilt from the measured wavefront.",
        )
        self.register_parameter(
            "pupil_diameter",
            default_value=2.7,
            min_value=0.0,
            unit="mm",
            description="Assumed pupil diameter used for phase reconstruction.",
        )
        self.register_parameter(
            "pupil_center",
            default_value=(0.0, 0.0),
            unit="mm",
            description="Pupil centroid used for phase reconstruction.",
        )

    @staticmethod
    def calc_n_zernike_terms(n: int) -> int:
        """Number of Zernike terms up to radial order ``n``, **including** piston.

        Differs from :func:`~ao_shaping.utils.wavefront.zernike_calc.calc_n_zernike_terms`
        by exactly the piston term, which the WFS slope fit carries but the
        canonical helper omits. Kept as a single implementation here so the two
        cannot drift apart unnoticed.
        """
        return _canonical_calc_n_zernike_terms(n) + 1

    @abstractmethod
    def take_image(self, n_sample: int = 10, dynamicNoiseCut: bool = True) -> None:
        """Acquire a spot-field image and derive spot centroids and diameters."""

    @abstractmethod
    def get_spots_statics(
        self,
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """Return ``(intensities, (centroid_x, centroid_y))`` for the last capture."""

    @abstractmethod
    def build_subaperture_mask(
        self,
        n_avg: int = 30,
        threshold_ratio: float = 0.3,
        edge_clip: int = 1,
        plot: bool = False,
    ) -> np.ndarray:
        """Boolean mask of subapertures that passed the flux threshold."""

    @abstractmethod
    def get_spot_deviation(
        self, cancel_tile: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Spot displacement from the reference, in metres."""

    @abstractmethod
    def get_wavefront(self, cancel_tile: bool = False) -> tuple[np.ndarray, dict]:
        """Reconstructed wavefront map in metres plus its metadata."""

    @abstractmethod
    def get_zernike(self, zernike_order: int = 10) -> np.ndarray:
        """Zernike coefficients fitted to the last wavefront, in **micrometres**.

        The micrometre unit is the historical contract of the WFS family and is
        asserted by :mod:`tests.ao_shaping.drivers.wfs.test_base_wfs`. Callers
        must convert with
        :func:`~ao_shaping.utils.wavefront.zernike_utils.um_to_waves` before
        doing anything in waves.
        """


__all__ = ["BaseWFS"]
