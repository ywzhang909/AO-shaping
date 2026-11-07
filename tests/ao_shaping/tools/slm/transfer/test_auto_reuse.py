"""自动重用历史模型：探测光路指纹、扫描历史检查点、按兼容性自动选择 (免设备)。"""

from __future__ import annotations

import json
from pathlib import Path

from ao_shaping.tools.slm.transfer.checkpoint_registry import (
    CHECKPOINT_NAME,
    register_session,
)
from ao_shaping.tools.slm.transfer.optical_session import save_fingerprint


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _session_with_fingerprint(root: Path, fingerprint: dict) -> Path:
    """一个带检查点 + 给定指纹的会话目录。"""
    _write(root / "journal.json", [{"stage": "plan", "status": "complete",
                                    "plan": {"shapes": ["square"]}}])
    _write(root / "shape" / "square" / "capture" / "manifest.json", [{"peak": 100.0}])
    body = {"ramp_matrix": [[1, 2], [3, 4]], "grid_size": [320, 320]}
    _write(root / "nominal" / CHECKPOINT_NAME, body)
    _write(root / "fit" / "result.json",
           {"metrics": [{"cv_measured": 0.2, "correlation": 0.9}]})
    save_fingerprint(fingerprint, root)
    return root


def _compatible_fingerprint() -> dict:
    """一份与默认检查点指纹兼容的当前光路指纹。"""
    return {
        "window_span": 820.0,
        "window_step": 120.0,
        "zero_radius": 50.0,
        "period": 128.0,
        "peak_counts": 100.0,
        "full_scale": 100.0,
        "exposure_ms": 5.0,
        "camera_format": "MONO8",
        "pixel_size_um": 2.2,
        "control_grid": 128,
        "fft_size": 1024,
        "optics_validity": "ok",
        "zero_xy": [471.0, 355.0],
    }


def _incompatible_fingerprint() -> dict:
    """一份曝光/照明完全不同的指纹 (分数应低于阈值)。"""
    fp = _compatible_fingerprint()
    fp["exposure_ms"] = 200.0
    fp["peak_counts"] = 500.0
    fp["full_scale"] = 100.0
    return fp


def test_auto_select_returns_best_checkpoint(tmp_path):
    """当前光路与历史检查点兼容时，返回该检查点的目录。"""
    from ao_shaping.tools.slm.transfer.auto_reuse import auto_select_checkpoint

    root = _session_with_fingerprint(tmp_path / "session", _compatible_fingerprint())
    register_session(root)

    result = auto_select_checkpoint(_compatible_fingerprint(), root=tmp_path)

    assert result is not None
    assert "path" in result
    assert "score" in result
    assert "verdict" in result
    assert Path(result["path"]) == root / "nominal"


def test_auto_select_returns_none_when_incompatible(tmp_path):
    """当前光路与历史检查点不兼容时，返回 None。"""
    from ao_shaping.tools.slm.transfer.auto_reuse import auto_select_checkpoint

    root = _session_with_fingerprint(tmp_path / "session", _compatible_fingerprint())
    register_session(root)

    result = auto_select_checkpoint(_incompatible_fingerprint(), root=tmp_path)

    assert result is None


def test_auto_select_returns_none_when_no_checkpoints(tmp_path):
    """没有历史检查点时，返回 None。"""
    from ao_shaping.tools.slm.transfer.auto_reuse import auto_select_checkpoint

    (tmp_path / "empty_session").mkdir()
    result = auto_select_checkpoint(_compatible_fingerprint(), root=tmp_path)
    assert result is None


def test_auto_select_returns_none_when_root_missing(tmp_path):
    """扫描根目录不存在时，返回 None。"""
    from ao_shaping.tools.slm.transfer.auto_reuse import auto_select_checkpoint

    result = auto_select_checkpoint(_compatible_fingerprint(),
                                     root=tmp_path / "nonexistent")
    assert result is None


def test_auto_select_skips_stale_checkpoints(tmp_path):
    """已标 stale 的检查点被跳过。"""
    from ao_shaping.tools.slm.transfer.auto_reuse import auto_select_checkpoint

    root = _session_with_fingerprint(tmp_path / "session", _compatible_fingerprint())
    register_session(root)

    # 篡改检查点内容使哈希失配 → scan 时标 stale
    checkpoint = root / "nominal" / CHECKPOINT_NAME
    checkpoint.write_text('{"tampered": true}', encoding="utf-8")

    result = auto_select_checkpoint(_compatible_fingerprint(), root=tmp_path)
    assert result is None


def test_auto_select_picks_highest_score(tmp_path):
    """多个历史检查点时，选兼容性分数最高的。"""
    from ao_shaping.tools.slm.transfer.auto_reuse import auto_select_checkpoint

    # 会话 A：完全兼容
    fp_a = _compatible_fingerprint()
    session_a = _session_with_fingerprint(tmp_path / "session_a", fp_a)
    register_session(session_a)

    # 会话 B：曝光不同 (部分兼容)
    fp_b = _compatible_fingerprint()
    fp_b["exposure_ms"] = 10.0
    session_b = _session_with_fingerprint(tmp_path / "session_b", fp_b)
    register_session(session_b)

    result = auto_select_checkpoint(_compatible_fingerprint(), root=tmp_path)

    assert result is not None
    # 两个检查点都兼容，但 A 的分数更高
    assert Path(result["path"]) == session_a / "nominal"


def test_probe_for_fingerprint_structure():
    """probe_for_fingerprint 的返回结构 (不实际开相机，只检查签名)。"""
    import inspect

    from ao_shaping.tools.slm.transfer.auto_reuse import probe_for_fingerprint

    sig = inspect.signature(probe_for_fingerprint)
    params = list(sig.parameters.keys())
    assert "camera" in params
    assert "cam_id" in params
    assert "exposure_ms" in params
    assert "frames" in params
    assert "label" in params
    assert "on_frame" in params
