"""Regression: a keyed widget must not also be given a ``value=`` from its key.

Reported on 2026-10-17 after clicking 「探测」 on the session exposure::

    The widget with key "bench_exposure_ms" was created with a default value
    but also had its value set via the Session State API.

Streamlit initialises a keyed widget *from* ``st.session_state[key]`` and warns
when ``value=`` is supplied as well, resolving the conflict in favour of session
state. So ``value=st.session_state.get(key, default)`` is doubly wrong: it logs
a warning on **every rerun**, and the number it passes is ignored anyway. The
probe button made this fire constantly, because it writes the key and then
reruns.

Why this is checked at the source level
---------------------------------------
The obvious test — instantiate the widget in bare mode and assert no warning —
is **vacuous**: ``policies.check_session_state_rules`` returns early unless
``runtime.exists()``, and bare mode has no ScriptRunContext, so *no* widget ever
warns. A version of this test that passed vacuously would be worse than none.

So the guard is an AST scan: no ``st.<widget>(...)`` call in this module may pass
a ``value=`` that reads ``st.session_state`` *and* a ``key=``. That catches the
pattern wherever it is reintroduced, without depending on Streamlit internals.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest
import streamlit as st

from ao_shaping.gui.slm import bench_settings as bs

MODULE = Path(bs.__file__)
#: The SLM controller page: sidebar + closed-loop shaping, all in one place.
PAGE = sorted((MODULE.parent).glob("*.py"))

#: Streamlit widgets that take both ``key`` and ``value``.
_WIDGETS = frozenset(
    {
        "button", "checkbox", "color_picker", "date_input", "file_uploader",
        "multiselect", "number_input", "radio", "select_slider", "selectbox",
        "slider", "text_area", "text_input", "time_input", "toggle",
    }
)


def _kwarg(call: ast.Call, name: str) -> ast.expr | None:
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _mentions_session_state(node: ast.AST) -> bool:
    return any(
        isinstance(n, ast.Attribute) and n.attr == "session_state"
        for n in ast.walk(node)
    )


def _wrapped_in_helper(node: ast.expr) -> bool:
    """``widget_default(key, default)`` is the sanctioned form — allow it.

    Reading a *mirror* key inside the default argument is legitimate; what the
    helper adds is returning ``None`` once the widget's own key holds a value,
    which is what suppresses the warning.
    """
    return isinstance(node, ast.Call) and _called_name(node) == "widget_default"


def _called_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


class TestWidgetDefault:
    def test_absent_key_seeds_the_default(self, state) -> None:
        assert bs.widget_default("fresh_key", 7.0) == 7.0

    def test_present_key_defers_to_session_state(self, state) -> None:
        state["k"] = 0.142
        assert bs.widget_default("k", 0.0) is None

    def test_a_falsey_session_value_is_still_defers(self, state) -> None:
        """``0`` / ``False`` are real values here (exposure 0 = auto)."""
        state["exp"] = 0.0
        assert bs.widget_default("exp", 99.0) is None
        state["crop"] = False
        assert bs.widget_default("crop", True) is None

    def test_the_helper_is_exported(self) -> None:
        assert "widget_default" in bs.__all__


class TestNoKeyedWidgetAlsoTakesASessionStateValue:
    def test_no_widget_reads_session_state_into_a_keyed_value(self) -> None:
        offenders: list[str] = []
        for module in PAGE:
            tree = ast.parse(
                module.read_text(encoding="utf-8-sig"), filename=str(module)
            )
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else None
                if name not in _WIDGETS:
                    continue
                value, key = _kwarg(node, "value"), _kwarg(node, "key")
                if (
                    key is not None
                    and value is not None
                    and not _wrapped_in_helper(value)
                    and _mentions_session_state(value)
                ):
                    offenders.append(f"{module.name}:{node.lineno} st.{name}")
        assert not offenders, (
            "keyed widget(s) given a value read from session_state — Streamlit "
            "warns on every rerun and ignores the value anyway; use "
            "widget_default(key, default):\n  " + "\n  ".join(offenders)
        )

    def test_the_scan_actually_catches_the_pattern(self) -> None:
        """Guards the guard, at the level where it is meaningful."""
        sample = ast.parse(
            "import streamlit as st\n"
            "st.number_input('w', value=float(st.session_state.get('k', 0.0)), key='k')\n"
            "st.number_input('w', value=1.0, key='k')\n"
            "st.number_input('w', value=1.0)\n"
        )
        caught = [
            n.lineno
            for n in ast.walk(sample)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "number_input"
            and _kwarg(n, "key") is not None
            and _kwarg(n, "value") is not None
            and _mentions_session_state(_kwarg(n, "value"))
        ]
        assert caught == [2], caught


@pytest.fixture
def state(monkeypatch):
    state: dict[str, Any] = {}
    monkeypatch.setattr(st, "session_state", state)
    return state
