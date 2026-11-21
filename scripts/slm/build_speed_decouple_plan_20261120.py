"""Build the frozen speed-decouple batch plan from the T1 phase matrix (2026-11-20).

纯离线：只读 T1 manifest 与相位 .npy，产出 ``speed_decouple_plan/1`` 批次 JSON
和 6 个 model 诊断相机的面板原生相位 .npy。不 import 任何设备驱动，不打开 SLM/CCD/激光，
不创建采集输出目录。

为什么 30 条而不是 48 条
------------------------
冻结方案 §2.1 名义 24 目标 x 2 方法 = 48；但冻结 checkpoint 的
``grid_size=128 / fft_size=512`` 只覆盖零级 ±253.35 CCD px，六种形状的 500/1000/1500
目标不可表示，T1 manifest 明确标 ``skipped_unrepresentable`` (phase_path=null)，
按方案 §7「不可表示的 model 相位须标诊断专用，不静默产出」不生成相位。因此实际带相位的
transition = 24 GS + 6 model(200 px) = 30 条。本脚本据此构建 30 条，不伪造缺失相位。

model 相机的 128x128 -> 面板 (1200,1920)
----------------------------------------
复用 P3 规范映射 ``ao_shaping.tools.slm.transfer.capture.to_panel``：
``pitch = 1920//128 = 15``，``np.repeat`` 块复制后裁 ``[:1200]``。这里直接内联同一
三段纯计算（不 import capture，其模块级 import 了策略/硬件模块），保持 builder 完全离线。
radians -> gray 仍由采集器端点 ``slm.create_phase_from_array`` 负责。

运行::

    python scripts/slm/build_speed_decouple_plan_20261120.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MATRIX_DIR = ROOT / "data/debug/slm_transfer/dyn_phase_matrix_20261120"
PANEL_PHASE_DIR = MATRIX_DIR / "phases_panel"
DEFAULT_MANIFEST = MATRIX_DIR / "manifest.json"
DEFAULT_PLAN = ROOT / "data/debug/slm_transfer/speed_decouple_plan_20261120.json"

#: 冻结参数（dynamic_decoupling_experiment_plan_20261120.md §2.3 + T3 探测结果）
PANEL_WH = [1920, 1200]
DURATION_S = 5.0
FPS = 10
EASING = "smooth"
MEASURE = {"strategy": "every", "every_n_frames": 1}
#: T3 12-bit 探测结论：可用 0.03-0.12 ms，冻结 0.08 ms（peak 2212.6 / 3890.25，未饱和）
EXPOSURE_MS = 0.08
BIT_DEPTH = 12
START_FLAT_GRAY = 0


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def to_panel(grid: np.ndarray) -> np.ndarray:
    """与 ``capture.to_panel`` 逐位一致：块复制 pitch=1920//size，裁 [:1200]。"""
    size = grid.shape[0]
    if grid.shape != (size, size) or 1920 % size or 1200 % (1920 // size):
        raise ValueError("grid must exactly tile the physical panel")
    pitch = 1920 // size
    return np.repeat(np.repeat(grid, pitch, axis=0), pitch, axis=1)[:1200]


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def build(manifest_path: Path, plan_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    used = [
        e for e in manifest["entries"]
        if e["status"] in ("written", "skipped") and e.get("phase_path")
    ]
    if len(used) != 30:
        raise RuntimeError(
            f"期望 30 条带相位条目 (24 gs + 6 model)，实际 {len(used)}；"
            "T1 产物不完整，拒绝构建计划"
        )

    PANEL_PHASE_DIR.mkdir(parents=True, exist_ok=True)
    transitions: list[dict[str, Any]] = []
    converted: list[str] = []
    for e in used:
        src = MATRIX_DIR / e["phase_path"]
        method = e["method"]
        arr = np.load(src, allow_pickle=False)
        if method == "gs":
            if arr.shape != (1200, 1920):
                raise RuntimeError(f"{src}: GS 相位 shape={arr.shape} != (1200,1920)")
            phase_path = src
        elif method == "model":
            if arr.shape != (128, 128):
                raise RuntimeError(f"{src}: model 相位 shape={arr.shape} != (128,128)")
            panel_arr = to_panel(arr)
            if panel_arr.shape != (1200, 1920):
                raise RuntimeError(f"to_panel 结果 shape={panel_arr.shape} != (1200,1920)")
            out = PANEL_PHASE_DIR / f"{e['shape']}_{e['size_px']}_model_panel.npy"
            np.save(out, panel_arr.astype(np.float64))
            converted.append(rel(out))
            phase_path = out
        else:
            raise RuntimeError(f"未知 method={method!r}")

        label = f"{e['shape']}_{e['size_px']}_{method}"
        transitions.append({
            "label": label,
            "group": {
                "shape": e["shape"],
                "size_px": int(e["size_px"]),
                "method": method,
                "model_phase_diagnostic_only": method == "model",
                "matrix_fingerprint": manifest["fingerprint"],
            },
            "start": {"kind": "flat", "gray": START_FLAT_GRAY},
            "target": {"kind": "phase", "path": rel(phase_path)},
            "target_mask": None,
        })

    plan = {
        "schema": "speed_decouple_plan/1",
        "source": rel(plan_path),
        "panel": PANEL_WH,
        "duration_s": DURATION_S,
        "fps": FPS,
        "easing": EASING,
        "measure": dict(MEASURE),
        "exposure_ms": EXPOSURE_MS,
        "bit_depth": BIT_DEPTH,
        "transitions": transitions,
    }
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = plan_path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    tmp.replace(plan_path)

    n_gs = sum(t["group"]["method"] == "gs" for t in transitions)
    n_model = sum(t["group"]["method"] == "model" for t in transitions)
    print(f"plan: {rel(plan_path)}")
    print(f"transitions: {len(transitions)} (gs={n_gs}, model={n_model})")
    print(f"model panel phases converted: {len(converted)}")
    for c in converted:
        print(f"  {c}")
    print(f"panel={PANEL_WH} duration_s={DURATION_S} fps={FPS} "
          f"exposure_ms={EXPOSURE_MS} bit_depth={BIT_DEPTH}")
    return plan


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    build(args.manifest, args.plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
