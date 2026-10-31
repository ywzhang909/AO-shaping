"""CCD 图像分析 UI (Streamlit)."""

from __future__ import annotations

from typing import Any

# 光束质量分析 (PySide6, 与 Streamlit 页面无关) 走惰性导出: 该子包会引入 Qt,
# 不能在 `import ao_shaping.gui.ccd` 时就加载。
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "BeamQualityApp": ("ao_shaping.gui.ccd.beam_quality", "BeamQualityApp"),
    "beam_quality_main": ("ao_shaping.gui.ccd.beam_quality", "main"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(target[0]), target[1])
    globals()[name] = value
    return value
