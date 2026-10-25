"""Fixed-budget acquisition policy shared by the SLM panel probes.

``beam_extent`` and ``panel_locate`` each measured the spot displacement /
extent as an energy ratio between a flat reference and a tilted panel
state, and each carried its own copy of the same fixed-frame budget
(``_PROBE_DISCARD`` / ``_PROBE_FRAMES`` / ``_probe_policy``). The budget
is one bench measurement, so it lives here once.

``PROBE_FRAMES`` was measured on the 2026-10-24 run: 7 kept frames give
the per-pixel median enough samples to reject one corrupted frame (peak
sigma 1.4%). ``--frames`` may raise the budget, never lower it.
"""

from __future__ import annotations

from ao_shaping.tools.kernels.bench_kernels import ProbePolicy

__all__ = ["PROBE_DISCARD", "PROBE_FRAMES", "probe_policy"]

#: Mounted frames after the write (LCOS flip) before the kept ones.
PROBE_DISCARD = 2
#: Measured kept-frame budget -- see the module docstring. ``--frames`` may
#: raise it, never lower it.
PROBE_FRAMES = 7


def probe_policy(frames: int) -> ProbePolicy:
    """Fixed-budget acquisition for an energy-ratio measurement."""
    return ProbePolicy(
        mode="frames",
        n_frames=max(int(frames), PROBE_FRAMES),
        n_discard=PROBE_DISCARD,
        combine="median",
    )
