"""Ratchet guard: SPGD sign decisions must go through the shared helper.

``ao_shaping/optimizer/spgd.py`` is the single source of truth for the SPGD
gradient sign. A hand-rolled sign silently optimises the OPPOSITE objective --
that happened twice in this repo (``optimizer/wfless/pib.py``, ``optimizer/wf/rms.py``)
and both times was only caught on hardware.

This test fails if:
* a migrated SPGD site stops using ``spgd_gradient``; or
* a new file appears that looks like an SPGD loop (has a perturbation variable
  and a ``gradient = ...`` assignment) while neither using the helper nor being
  on the explicit ascending-frame exemption list.

The exemption list is the ``param += update`` family, which is correct as-is.
Shrink it by migrating those sites to the helper; never grow it silently.
"""

import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[3] / "src" / "ao_shaping"

# Sites already routed through spgd_gradient (param -= update frame).
MIGRATED = (
    "optimizer/wfless/pib.py",
    "optimizer/wfless/slm_zernike_pib.py",
    "optimizer/wf/rms.py",
    "optimizer/wf/rms_by_zernike.py",
    "optimizer/combined_optimizer.py",
    "tools/train_data_collect.py",
)

# Correct, deliberately NOT migrated yet: they use the additive `param += update`
# frame (and sim_spgd / slm_square_shaping carry a `+ disturb/2` return-to-centre
# term), so migrating them means flipping operators, not just the sign.
EXEMPT_ASCENDING_FRAME = {
    "optimizer/wfless/sim_spgd.py",
    "optimizer/wfless/slm_square_shaping.py",
    "optimizer/wfless/adc_dm_adam.py",
    "optimizer/wfless/gready_cam.py",
}

_GRADIENT_ASSIGNMENT = re.compile(r"^[ \t]*gradient\s*=", re.MULTILINE)


@pytest.mark.parametrize("rel", MIGRATED)
def test_migrated_site_uses_shared_helper(rel: str):
    src = (SRC / rel).read_text(encoding="utf8")
    assert "spgd_gradient" in src, (
        f"{rel} no longer uses optimizer/spgd.py::spgd_gradient -- the SPGD sign "
        "must have a single source of truth"
    )


def test_exempt_list_only_contains_existing_files():
    for rel in EXEMPT_ASCENDING_FRAME:
        assert (SRC / rel).is_file(), f"exemption points at a missing file: {rel}"


def test_no_unexpected_hand_rolled_spgd_gradient():
    """Discover new SPGD-looking files that bypass the shared helper."""
    offenders: list[str] = []
    for package in ("optimizer", "tools"):
        for path in (SRC / package).rglob("*.py"):
            rel = path.relative_to(SRC).as_posix()
            if rel in EXEMPT_ASCENDING_FRAME:
                continue
            src = path.read_text(encoding="utf8")
            if "disturb" not in src:
                continue
            if _GRADIENT_ASSIGNMENT.search(src) and "spgd_gradient" not in src:
                offenders.append(rel)
    assert not offenders, (
        "Hand-rolled SPGD gradient detected; route it through "
        f"optimizer/spgd.py::spgd_gradient: {offenders}"
    )
