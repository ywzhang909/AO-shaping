"""Golden: the exact Streamlit call sequence of ``_render_slm_settings``.

Issue #167 asks for the long GUI render functions to be split, and for a
regression to prove the split changed nothing. "Nothing changed" is not
observable by reading the diff, so it is recorded: this test drives
``_render_slm_settings`` against a recording stub and pins the **ordered
sequence of Streamlit calls** (labels, widget keys, captions, column counts).

Any refactor that reorders a widget, renames a key, drops a caption or moves a
write into another column changes the sequence and fails here. That is the
property the split must preserve -- not the code shape.

Three scenarios, because the branches differ in depth:

* ``plain`` -- nominal state;
* ``mismatch`` -- the widget wavelength differs from the device value and a
  config file matched, reaching the warning / sync / refresh-config branches
  that the plain scenario never touches;
* ``actions`` -- every button is pressed, reaching the handlers (apply
  correction, save config, save phase CSV) that hold the deepest nesting.

The golden is regenerated with::

    AO_SLM_SETTINGS_GOLDEN_UPDATE=1 pytest tests/ao_shaping/gui/slm/test_slm_settings_render_golden.py

which rewrites the JSON next to this file. Re-basing the golden is a deliberate
act, not a side effect.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from ao_shaping.gui.slm import multi_slm_controller as gui

GOLDEN = Path(__file__).with_name("slm_settings_render_golden.json")
UPDATE = os.environ.get("AO_SLM_SETTINGS_GOLDEN_UPDATE") == "1"


class _Upload:
    """Stand-in for a file-uploader result."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._buffer = bytearray(b"a,b\n1,2\n")

    def getbuffer(self) -> memoryview:
        return memoryview(self._buffer)


class _RecordingSt:
    """Stand-in for ``streamlit`` that records every call in order.

    Widgets return inert values unless the scenario marks that button as
    pressed, so no device command fires and the recorded sequence is stable.
    """

    def __init__(
        self,
        session_state: dict[str, Any],
        pressed: frozenset[str] = frozenset(),
        upload: Any = None,
    ) -> None:
        self.session_state = session_state
        self.pressed = pressed
        self.upload = upload
        self.calls: list[dict[str, Any]] = []

    def _record(self, kind: str, **detail: Any) -> None:
        self.calls.append({"kind": kind, **detail})

    @contextmanager
    def _block(self, kind: str, **detail: Any):
        self._record(kind, **detail)
        yield self

    def expander(self, label: str, **kwargs: Any):
        return self._block("expander", label=label, **kwargs)

    def columns(self, spec: Any, **kwargs: Any):
        self._record("columns", spec=spec)
        count = spec if isinstance(spec, int) else len(spec)
        return tuple(self._block("column") for _ in range(count))

    def markdown(self, body: str, **kwargs: Any) -> None:
        self._record("markdown", body=body)

    def write(self, body: Any, **kwargs: Any) -> None:
        self._record("write", body=str(body))

    def caption(self, body: Any, **kwargs: Any) -> None:
        self._record("caption", body=str(body))

    def json(self, body: Any, **kwargs: Any) -> None:
        self._record("json", body=body)

    def number_input(self, label: str, **kwargs: Any) -> float:
        self._record("number_input", label=label, key=kwargs.get("key"))
        return float(kwargs.get("value", 0))

    def selectbox(self, label: str, **kwargs: Any) -> Any:
        self._record("selectbox", label=label, key=kwargs.get("key"))
        options = kwargs.get("options") or []
        return options[0] if options else None

    def checkbox(self, label: str, **kwargs: Any) -> bool:
        self._record("checkbox", label=label, key=kwargs.get("key"))
        return bool(kwargs.get("value", False))

    def file_uploader(self, label: str, **kwargs: Any) -> Any:
        self._record("file_uploader", label=label, key=kwargs.get("key"))
        return self.upload

    def button(self, label: str, **kwargs: Any) -> bool:
        self._record("button", label=label, key=kwargs.get("key"))
        return kwargs.get("key") in self.pressed

    def success(self, body: Any, **kwargs: Any) -> None:
        self._record("success", body=str(body))

    def info(self, body: Any, **kwargs: Any) -> None:
        self._record("info", body=str(body))

    def warning(self, body: Any, **kwargs: Any) -> None:
        self._record("warning", body=str(body))

    def error(self, body: Any, **kwargs: Any) -> None:
        self._record("error", body=str(body))

    def rerun(self, **kwargs: Any) -> None:
        self._record("rerun")


class _FakeCorrection:
    def __init__(self) -> None:
        self.is_valid = True
        self.csv_path = Path("correction.csv")


class _FakeSLM:
    """Minimal stand-in for an open Santec handle."""

    def __init__(self) -> None:
        self.slm_number = 1
        self._serial_number = "SN-0001"
        self.wavelength = 1064
        self.Pixel_Size_um = 8.0
        self._correction = _FakeCorrection()
        self.correction_enabled = True
        self.correction_csv_path = Path("correction.csv")
        self.shift_x = 0
        self.shift_y = 0
        self.saved = False
        self.correction_csv: str | None = None

    def load_config(self) -> dict[str, Any]:
        return {"wavelength": self.wavelength}

    def load_correction_from_csv(self, path: str | None) -> bool:
        self.correction_csv = path
        return path is not None

    def get_displayed_phase(self) -> tuple[Any, Any]:
        return None, None

    def set_shift(self, shift_x: int = 0, shift_y: int = 0) -> None:
        self.shift_x = shift_x
        self.shift_y = shift_y

    def save_config(self) -> None:
        self.saved = True


def _session_state(*, mismatch: bool = False) -> dict[str, Any]:
    state: dict[str, Any] = {
        "slm0": _FakeSLM(),
        "slm0_width": 1920,
        "slm0_height": 1200,
        "slm0_pixel_pitch_um": 8.0,
        "slm0_bits": 10,
        "slm0_wavelength": 1064,
        "slm0_video_mode": "内存模式",
        "slm0_shift_x": 0,
        "slm0_shift_y": 0,
        "slm0_use_correction": True,
    }
    if mismatch:
        # The widget value differs from the device value, and a config file was
        # matched: this reaches the warning / sync / refresh-config branches that
        # the plain scenario never touches.
        state["slm0_wavelength"] = 532
        state["slm0_wavelength_mismatch"] = {"serial": "SN-0001"}
        state["slm0_loaded_config"] = {"wavelength": 1064}
    return state


#: Every button key in the settings sidebar. Pressing all of them drives the
#: handlers, which is where the deepest nesting lives.
_ALL_BUTTONS = frozenset(
    {
        "slm0_sync_wl_btn",
        "slm0_set_wl_btn",
        "slm0_set_mode_btn",
        "slm0_apply_shift_btn",
        "slm0_refresh_config_btn",
        "slm0_apply_corr_btn",
        "slm0_clear_corr_btn",
        "slm0_save_config_btn",
        "slm0_save_csv_btn",
    }
)


def _render(
    monkeypatch: pytest.MonkeyPatch,
    *,
    mismatch: bool = False,
    pressed: frozenset[str] = frozenset(),
    upload: Any = None,
) -> list[dict[str, Any]]:
    recorder = _RecordingSt(
        _session_state(mismatch=mismatch), pressed=pressed, upload=upload
    )
    monkeypatch.setattr(gui, "st", recorder)
    monkeypatch.setattr(gui, "widget_default", lambda key, default: default)
    for helper in (
        "_render_grayscale_config",
        "_render_base_phase_block",
        "_render_defocus_overlay_block",
    ):
        monkeypatch.setattr(gui, helper, lambda *a, **k: None)
    for handler in (
        "set_wavelength",
        "set_video_mode",
        "set_shift",
        "toggle_correction",
        "refresh_phase_preview",
    ):
        monkeypatch.setattr(gui, handler, lambda *a, **k: None)
    gui._render_slm_settings(0)
    return recorder.calls


def test_render_sequence_matches_the_golden(monkeypatch: pytest.MonkeyPatch) -> None:
    scenarios = {
        "plain": _render(monkeypatch),
        "mismatch": _render(monkeypatch, mismatch=True),
        "actions": _render(
            monkeypatch,
            mismatch=True,
            pressed=_ALL_BUTTONS,
            upload=_Upload("correction.csv"),
        ),
    }

    if UPDATE:
        GOLDEN.write_text(
            json.dumps(scenarios, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        expected = scenarios
    else:
        assert GOLDEN.exists(), (
            f"missing golden {GOLDEN.name}; regenerate with "
            "AO_SLM_SETTINGS_GOLDEN_UPDATE=1"
        )
        expected = json.loads(GOLDEN.read_text(encoding="utf-8"))

    assert scenarios == expected
