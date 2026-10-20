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

The two ``optimize_pib`` paths used to decide the sign themselves. They now
delegate the whole epoch to ``optimizer/spid_pib_loop.py``, which is where the
sign is decided -- so they are checked for *that* delegation instead, and the
scaffold itself is a migrated site.
"""

import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[3] / "src" / "ao_shaping"

#: The shared SPGD-PIB scaffold: bench + epoch mechanics for both PIB paths.
SCAFFOLD = "optimizer/spid_pib_loop.py"

#: Sites already routed through spgd_gradient (param -= update frame).
MIGRATED = (
    SCAFFOLD,
    "optimizer/wfless/slm_zernike_pib.py",
    "optimizer/wf/rms.py",
    "optimizer/wf/rms_by_zernike.py",
    "tools/train_data_collect.py",
)

#: Sites that no longer decide the sign themselves but hand the whole epoch to
#: the scaffold, which does decide it.
DELEGATED = (
    "optimizer/wfless/pib.py",
    "optimizer/combined_optimizer.py",
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

_GRADIENT_ASSIGNMENT = re.compile(r"^[ \t]*gradient\s*=(.*)$", re.MULTILINE)

#: A gradient handed over by the scaffold, e.g. ``gradient = _epoch.gradient``.
#: Anything else on the right-hand side means the file computed it itself.
_SCAFFOLD_GRADIENT = re.compile(r"^[ \t]*[\w.]+\.gradient\s*$")


@pytest.mark.parametrize("rel", MIGRATED)
def test_migrated_site_uses_shared_helper(rel: str):
    src = (SRC / rel).read_text(encoding="utf8")
    assert "spgd_gradient" in src, (
        f"{rel} no longer uses optimizer/spgd.py::spgd_gradient -- the SPGD sign "
        "must have a single source of truth"
    )


@pytest.mark.parametrize("rel", DELEGATED)
def test_delegated_site_routes_through_the_scaffold(rel: str):
    """The delegation is only legitimate while the scaffold decides the sign."""
    src = (SRC / rel).read_text(encoding="utf8")
    assert "spid_pib_loop" in src, (
        f"{rel} must take the SPGD epoch from {SCAFFOLD}; if it computes the "
        "gradient itself again, the sign decision has forked"
    )
    assert SCAFFOLD in MIGRATED, (
        f"the scaffold {SCAFFOLD} must itself be a migrated site, otherwise "
        "delegating to it would hide the sign decision"
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
            if not _GRADIENT_ASSIGNMENT.search(src):
                continue
            if "spgd_gradient" in src:
                continue
            if rel in DELEGATED and all(
                _SCAFFOLD_GRADIENT.match(rhs)
                for rhs in _GRADIENT_ASSIGNMENT.findall(src)
            ):
                continue
            offenders.append(rel)
    assert not offenders, (
        "Hand-rolled SPGD gradient detected; route it through "
        f"optimizer/spgd.py::spgd_gradient: {offenders}"
    )


def test_the_scan_actually_bites(tmp_path: Path):
    """A guard that finds nothing is worse than none -- prove it can fail."""
    planted = tmp_path / "pkg" / "rogue.py"
    planted.parent.mkdir(parents=True)
    planted.write_text(
        "disturb_v = rng.binomial(1, 0.5, (n,))\n"
        "gradient = -diff * disturb_v\n",
        encoding="utf8",
    )
    src = planted.read_text(encoding="utf8")
    assert _GRADIENT_ASSIGNMENT.search(src)
    assert not all(
        _SCAFFOLD_GRADIENT.match(rhs) for rhs in _GRADIENT_ASSIGNMENT.findall(src)
    )

    handed_over = tmp_path / "pkg" / "delegate.py"
    handed_over.write_text(
        "disturb_v = rng.binomial(1, 0.5, (n,))\n"
        "gradient = _epoch.gradient\n",
        encoding="utf8",
    )
    src = handed_over.read_text(encoding="utf8")
    assert all(
        _SCAFFOLD_GRADIENT.match(rhs) for rhs in _GRADIENT_ASSIGNMENT.findall(src)
    )
