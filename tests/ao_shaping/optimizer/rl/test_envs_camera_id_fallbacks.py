"""Pin the RL envs' camera-id fallback to the canonical DeviceConfig.

``optimizer/rl/envs.py`` reads ``Far_Cam_ID`` at module scope (an import-time
knob, TODO 8.9 item 78). Its inverted fallback used to disagree with
``config.DeviceConfig`` and ``.env`` (TODO 8.9 item 80), so an env-unset
import drove the far camera as device 1 while config-driven consumers used
device 0. These tests read both sources as text/AST instead of importing
``envs``, which would drag in gymnasium and the driver packages.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ENVS = ROOT / "src/ao_shaping/optimizer/rl/envs.py"
CONFIG = ROOT / "src/ao_shaping/config.py"


def _envs_far_fallback() -> int:
    """The literal fallback of the module-level Far_Cam_ID read in envs.py."""
    tree = ast.parse(ENVS.read_text(encoding="utf-8"), filename=str(ENVS))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "Far_Cam_ID" for t in node.targets):
            continue
        call = node.value
        assert isinstance(call, ast.Call) and getattr(call.func, "id", "") == "int", (
            "Far_Cam_ID must be int(os.environ.get(...))"
        )
        inner = call.args[0]
        assert isinstance(inner, ast.Call) and getattr(inner.func, "attr", "") == "get"
        name_arg, fallback = inner.args[0], inner.args[1]
        assert isinstance(name_arg, ast.Constant) and name_arg.value == "Far_Cam_ID"
        assert isinstance(fallback, ast.Constant), "fallback must be a literal"
        return int(fallback.value)
    raise AssertionError("no module-level Far_Cam_ID read found in envs.py")


def _device_config_fallback(attr: str) -> int:
    """The literal fallback DeviceConfig reads for ``attr`` in config.py."""
    tree = ast.parse(CONFIG.read_text(encoding="utf-8"), filename=str(CONFIG))
    for cls in tree.body:
        if not (isinstance(cls, ast.ClassDef) and cls.name == "DeviceConfig"):
            continue
        for fn in cls.body:
            if not (isinstance(fn, ast.FunctionDef) and fn.name == "__init__"):
                continue
            for stmt in fn.body:
                target = stmt.targets[0] if isinstance(stmt, ast.Assign) else None
                if not isinstance(target, ast.Attribute) or target.attr != attr:
                    continue
                stmt_call = stmt.value
                assert isinstance(stmt_call, ast.Call) and getattr(
                    stmt_call.func, "id", ""
                ) == "int", f"DeviceConfig.{attr} must be int(os.environ.get(...))"
                fallback = stmt_call.args[0].args[1]
                assert isinstance(fallback, ast.Constant), "fallback must be a literal"
                return int(fallback.value)
    raise AssertionError(f"DeviceConfig.{attr} not found in config.py")


def test_source_layout_is_where_this_file_assumes() -> None:
    """A silent path/depth slip would make every other check vacuously pass."""
    assert ENVS.is_file() and CONFIG.is_file(), (
        f"source layout moved: {ENVS} / {CONFIG} -- fix parents[4] first"
    )


def test_far_cam_fallback_matches_device_config() -> None:
    assert _envs_far_fallback() == _device_config_fallback("far_cam_id"), (
        "envs.py's Far_Cam_ID fallback disagrees with config.DeviceConfig "
        "(or .env): an env-unset process would open the wrong camera"
    )


def test_near_cam_is_not_a_dead_module_read() -> None:
    """Nothing in envs.py consumes Near_Cam_ID; the dead read must not return."""
    tree = ast.parse(ENVS.read_text(encoding="utf-8"), filename=str(ENVS))
    dead = [
        node.lineno
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "Near_Cam_ID" for t in node.targets)
    ]
    assert not dead, (
        f"envs.py re-introduced a module-level Near_Cam_ID read at lines {dead}; "
        "nothing in the module consumes it -- delete it or wire a consumer"
    )
