"""Bottom-up repro + timing of the Zernike-selectbox abort.

Real Streamlit AppTest, no hardware. Measures per-layer wall time to isolate the
slow spot, and reproduces the abort by switching the selectbox to Zernike and by
changing n_max (which changes the fixed-row st.data_editor length).

Run:  PYTHONPATH=src .venv/bin/python tests/ao_shaping/gui/slm/test_zernike_abort_repro.py
"""

from __future__ import annotations

import sys
import time
from typing import Any

import streamlit as st
from streamlit.testing.v1 import AppTest


# ---------------------------------------------------------------------------
# App scripts (each runs in its own fresh Streamlit script context).
# ---------------------------------------------------------------------------
def app_zernike_alone() -> None:
    import streamlit as st

    from ao_shaping.gui.slm.pattern_controls import ZernikeControl

    ctrl = ZernikeControl(slm_id=1)
    params: dict[str, Any] = ctrl.render("slm1")
    st.write({"n_max": params["n_max"], "n_coeffs": len(params["coefficients"])})


def app_pattern_select() -> None:
    import streamlit as st

    from ao_shaping.gui.slm.multi_slm_controller import render_pattern_controls

    pattern_type, _params = render_pattern_controls(1)
    st.write({"pattern": pattern_type})


# ---------------------------------------------------------------------------
# Layer 1 (bottom): Zernike control in isolation.
# ---------------------------------------------------------------------------
def test_layer1_first_render() -> None:
    t0 = time.perf_counter()
    at = AppTest.from_function(app_zernike_alone, default_timeout=30)
    at.run()
    dt = time.perf_counter() - t0
    assert not at.exception, f"ABORT first render: {at.exception!r}"
    n_max = at.number_input(key="slm1_zernike_n_max").value
    print(f"[layer1] first render {dt:.3f}s, no abort, n_max={n_max}")


def test_layer1_increase_n_max() -> None:
    at = AppTest.from_function(app_zernike_alone, default_timeout=30)
    at.run()
    assert not at.exception
    for target in (6, 7, 8, 10):
        t0 = time.perf_counter()
        at.number_input(key="slm1_zernike_n_max").set_value(target)
        at.run()
        dt = time.perf_counter() - t0
        if at.exception:
            print(f"[layer1] ABORT at n_max={target}: {at.exception!r}")
            return
        print(f"[layer1] n_max ->{target}: {dt:.3f}s, no abort")
    print("[layer1] all n_max changes ok, no abort")


# ---------------------------------------------------------------------------
# Layer 2 (top): real selectbox in render_pattern_controls, switch to Zernike.
# ---------------------------------------------------------------------------
def test_layer2_selectbox_switch_to_zernike() -> None:
    at = AppTest.from_function(app_pattern_select, default_timeout=30)
    t0 = time.perf_counter()
    at.run()
    dt = time.perf_counter() - t0
    assert not at.exception, f"ABORT first: {at.exception!r}"
    first = at.selectbox(key="slm1_pattern_type").value
    print(f"[layer2] first render {dt:.3f}s, default pattern={first!r}")

    t0 = time.perf_counter()
    at.selectbox(key="slm1_pattern_type").set_value("Zernike")
    at.run()
    dt = time.perf_counter() - t0
    if at.exception:
        print(f"[layer2] ABORT on switch->Zernike: {at.exception!r}")
        return
    print(f"[layer2] switch ->Zernike: {dt:.3f}s, no abort")

    t0 = time.perf_counter()
    at.number_input(key="slm1_zernike_n_max").set_value(10)
    at.run()
    dt = time.perf_counter() - t0
    if at.exception:
        print(f"[layer2] ABORT on n_max->10: {at.exception!r}")
        return
    print(f"[layer2] n_max ->10: {dt:.3f}s, no abort")


if __name__ == "__main__":
    tests = [
        test_layer1_first_render,
        test_layer1_increase_n_max,
        test_layer2_selectbox_switch_to_zernike,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as exc:  # noqa: BLE001 - report any failure
            failed += 1
            print(f"FAIL: {t.__name__}: {exc!r}")
    print(f"\n{'ALL PASS' if failed == 0 else str(failed) + ' FAILED'}")
    sys.exit(1 if failed else 0)
