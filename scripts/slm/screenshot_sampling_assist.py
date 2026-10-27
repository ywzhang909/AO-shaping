from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

# --- mock a connected SLM so the connected branch renders ---------------------
if "slm1" not in st.session_state:
    st.session_state["slm1"] = type("SLM", (), {"is_open": True})()
    st.session_state["slm1_connected"] = True
if "camera_pixel_um" not in st.session_state:
    st.session_state["camera_pixel_um"] = 2.2

# --- pre-fill sampling-assist fields from env --------------------------------
cfg = json.loads(os.environ.get("SAMPLING_CFG", "{}"))
for name, val in cfg.items():
    st.session_state[f"slm1_shp_gs-refine_link_{name}"] = val

# --- render the popover body inline (no click needed for a screenshot) -------
_orig_popover = st.popover


class _Inline:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _inline_popover(*a, **k):
    st.markdown("### 远场尺寸换算 (inline capture)")
    return _Inline()


st.popover = _inline_popover

from ao_shaping.gui.slm.shaping_controls import render_closed_loop_shaping

render_closed_loop_shaping(1)
