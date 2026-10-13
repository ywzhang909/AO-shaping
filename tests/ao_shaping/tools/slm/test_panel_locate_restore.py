"""SLM state is restored after a panel-location probe."""

import pytest
import numpy as np

from ao_shaping.tools.slm.panel.panel_locate import _paired_delta, _restore_slm_display


class _SlmStub:
    def __init__(self, slot):
        self.slot = slot
        self.restored = None

    def get_displayed_memory_number(self):
        return self.slot

    def get_current_grayscale(self):
        return 17

    def set_grayscale(self, gray):
        self.restored = ("gray", gray)

    def _display_memory(self, slot):
        self.restored = ("slot", slot)


@pytest.mark.parametrize(
    ("slot", "expected"), [(None, ("gray", 17)), (5, ("slot", 5))]
)
def test_restores_original_display_even_on_error(slot, expected):
    slm = _SlmStub(slot)
    with pytest.raises(RuntimeError, match="probe failed"):
        with _restore_slm_display(slm):
            raise RuntimeError("probe failed")
    assert slm.restored == expected


def test_paired_flat_removes_linear_power_drift():
    box = np.ones((2, 2), dtype=bool)
    delta, drift = _paired_delta(np.full((2, 2), 100.0),
                                 np.full((2, 2), 90.0),
                                 np.full((2, 2), 80.0), box, 0.25)
    assert delta == 0
    assert drift == pytest.approx(2 / 9)


def test_paired_flat_rejects_power_step():
    box = np.ones((2, 2), dtype=bool)
    with pytest.raises(ValueError, match="drift"):
        _paired_delta(np.ones((2, 2)), np.ones((2, 2)),
                      np.full((2, 2), 0.1), box, 0.2)
