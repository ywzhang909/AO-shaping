"""把某一晚的现场几何标定固化成 ``data/slm_geometry_calib.npz`` 快照。

为什么要这个脚本
----------------
``hw_forward_model_fidelity_ladder.py`` 的 ``--geometry-npz`` 读的是 npz 存档;
2026-11-16 那次阶梯显式传了 2026-10-23 的存档几何 (``center y=962.63``), 而同晚
现场重标定给出 ``y=1104.84`` —— y 差 142 px, 于是阶梯的模型-实测对照被陈旧原点
主导 (flat 图样 Pearson 0.065; 只换中心即升到 0.894)。

现场标定的结果当时只落在 ``data/debug/differentiable_opt/<run>/run_config.json``
的 ``geometry`` 块里, 没有任何脚本把它写回 npz, 所以后续任何按默认路径取几何的
离线复算都会再次读到那份陈旧存档。本脚本补这一步: 读 ``run_config.json`` 的
``geometry`` 块 → 按 ``calib_geometry_from_orders.py`` 的同一 schema 写 npz, 并把
被替换的旧文件按它自己的 ``provenance.measured_at`` 存成
``data/slm_geometry_calib_<YYYYMMDD>.npz``, 保证可回退、可溯源。

判据
----
* 只接受 ``geometry.measured_this_run == true`` 的现场标定; 存档回退 (``source``
  里出现「存档」) 一律拒收 —— 否则等于把旧值再固一遍。
* ``Kx``/``Ky``/``center``/``rotation_deg``/``fwhm0``/``mirror`` 原样落盘, 不做平均
  或截断; ``k_mean`` 是派生量, 留给读端自己算。
* ``--check`` 只比对不写盘, 不一致返回非零。

用法
----
``python scripts/slm/persist_live_geometry.py --run-config <run_config.json>``
``python scripts/slm/persist_live_geometry.py --check``
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "data" / "slm_geometry_calib.npz"
DEFAULT_RUN_CONFIG = (
    ROOT / "data" / "debug" / "differentiable_opt" / "20261117_022658" / "run_config.json"
)

#: 落盘字段: 与 ``load_geometry_npz`` / ``calib_geometry_from_orders.py`` 一致。
SCALAR_KEYS = ("Kx", "Ky", "rotation_deg", "fwhm0", "resid_x", "resid_y", "crop_side")
#: 比对容差 (npz 为 float64, run_config 由 json 读回, 只差序列化位数)。
TOL = 1e-6


def _rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _field(geom: dict[str, Any], name: str) -> Any:
    """取 ``geometry`` 字段: run_config 落盘用小写 ``kx``/``ky``, npz 用大写。"""
    for key in (name, name.lower(), name.upper()):
        if key in geom:
            return geom[key]
    raise KeyError(name)


def _lower_keys(geom: dict[str, Any]) -> set[str]:
    return {str(k).lower() for k in geom}


def load_run_config_geometry(path: Path) -> dict[str, Any]:
    """读 ``run_config.json`` 的 ``geometry`` 块, 拒收非现场标定。"""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"run_config.json 不存在: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    geom = data.get("geometry")
    if not isinstance(geom, dict):
        raise ValueError(f"{p} 里没有 geometry 块")
    if not bool(geom.get("measured_this_run")):
        raise ValueError(
            f"{p} 的 geometry.measured_this_run 为假 (source={geom.get('source')!r}); "
            "本脚本只固化现场实测标定, 不接受存档回退"
        )
    source = str(geom.get("source", ""))
    if "存档" in source:
        raise ValueError(f"{p} 的 geometry 来自存档 ({source!r}), 不能当作现场标定固化")
    missing = [
        k for k in ("center", "Kx", "Ky", "rotation_deg", "fwhm0", "mirror") if k.lower() not in _lower_keys(geom)
    ]
    if missing:
        raise ValueError(f"{p} 的 geometry 块缺字段: {missing}")
    center = [float(v) for v in geom["center"]]
    if len(center) != 2:
        raise ValueError(f"geometry.center 应为 (cy, cx), 实得 {center}")
    return {
        "run_config": p,
        "run_config_rel": _rel(p),
        "timestamp_utc": str(data.get("timestamp_utc", "")),
        "devices": data.get("devices"),
        "center": center,
        "Kx": float(_field(geom, "Kx")),
        "Ky": float(_field(geom, "Ky")),
        "rotation_deg": float(geom["rotation_deg"]),
        "fwhm0": float(geom["fwhm0"]),
        "mirror": bool(geom["mirror"]),
        "resid_x": float(geom.get("resid_x", 0.0)),
        "resid_y": float(geom.get("resid_y", 0.0)),
        "k_choice": str(geom.get("k_choice", "K = (Kx + Ky)/2 (各向同性)")),
    }


def snapshot_name_for(npz_path: Path) -> Path:
    """旧 npz 的存档名: 取它 ``provenance.measured_at`` 的日期, 退回 mtime。"""
    stamp = ""
    try:
        with np.load(npz_path, allow_pickle=False) as d:
            raw = d["provenance"] if "provenance" in d.files else None
        if raw is not None:
            prov = json.loads(str(raw))
            stamp = str(prov.get("measured_at", ""))
    except (OSError, ValueError, json.JSONDecodeError):
        stamp = ""
    day = "".join(ch for ch in stamp[:10] if ch.isdigit())
    if len(day) != 8:
        day = datetime.fromtimestamp(npz_path.stat().st_mtime).strftime("%Y%m%d")
    return npz_path.with_name(f"{npz_path.stem}_{day}{npz_path.suffix}")


def build_arrays(cfg: dict[str, Any], cfg_rel: str) -> dict[str, Any]:
    """按 npz schema 组装数组 (``provenance`` 为 JSON 字符串)。"""
    devices = cfg.get("devices") or {}
    provenance = {
        "source": "scripts/slm/persist_live_geometry.py",
        "measured_at": cfg["timestamp_utc"],
        "from_run_config": cfg_rel,
        "slm": devices.get("slm") if isinstance(devices, dict) else None,
        "camera": devices.get("camera") if isinstance(devices, dict) else None,
        "k_choice": cfg["k_choice"],
        "note": "现场 SLMCCDCalibrator 标定固化; 字段原样落盘, 未做平均或截断",
    }
    out: dict[str, Any] = {
        "center": np.asarray(cfg["center"], dtype=np.float64),
        "Kx": np.float64(cfg["Kx"]),
        "Ky": np.float64(cfg["Ky"]),
        "rotation_deg": np.float64(cfg["rotation_deg"]),
        "fwhm0": np.float64(cfg["fwhm0"]),
        "mirror": np.bool_(cfg["mirror"]),
        "resid_x": np.float64(cfg["resid_x"]),
        "resid_y": np.float64(cfg["resid_y"]),
        "crop_side": np.int64(0),
        "beam_center": np.zeros(2, dtype=np.float64),
        "beam_sigma": np.zeros(2, dtype=np.float64),
        "provenance": np.str_(json.dumps(provenance, ensure_ascii=False)),
    }
    assert set(out) >= set(SCALAR_KEYS) | {"center", "mirror"}
    return out


def compare(npz_path: Path, cfg: dict[str, Any]) -> list[str]:
    """比对现盘 npz 与 run_config, 返回差异描述 (空 = 一致)。"""
    diffs: list[str] = []
    with np.load(npz_path, allow_pickle=False) as d:
        files = set(d.files)
        missing = (set(SCALAR_KEYS) | {"center", "mirror"}) - files
        if missing:
            return [f"npz 缺字段: {sorted(missing)}"]
        center = [float(v) for v in d["center"]]
        if max(abs(center[0] - cfg["center"][0]), abs(center[1] - cfg["center"][1])) > TOL:
            diffs.append(f"center {center} != {cfg['center']}")
        for key in ("Kx", "Ky", "rotation_deg", "fwhm0", "resid_x", "resid_y"):
            got = float(d[key])
            want = float(cfg[key])
            if abs(got - want) > max(TOL, abs(want) * TOL):
                diffs.append(f"{key} {got} != {want}")
        if bool(d["mirror"]) != bool(cfg["mirror"]):
            diffs.append(f"mirror {bool(d['mirror'])} != {cfg['mirror']}")
    return diffs


def _previous_center(npz_path: Path) -> list[float] | None:
    if not npz_path.is_file():
        return None
    try:
        with np.load(npz_path, allow_pickle=False) as d:
            return [float(v) for v in d["center"]]
    except (OSError, KeyError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", default=str(DEFAULT_RUN_CONFIG))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--check", action="store_true", help="只比对现盘 npz 与 run_config, 不写盘"
    )
    parser.add_argument(
        "--no-archive", action="store_true", help="不把被替换的旧 npz 存成日期快照"
    )
    args = parser.parse_args()

    cfg = load_run_config_geometry(Path(args.run_config))
    out = Path(args.out)
    logger.info(
        "[geometry] 现场标定 {}: center(cy,cx)=({:.3f},{:.3f}) Kx={:.1f} Ky={:.1f} "
        "rot={:.4f}° fwhm0={:.2f}",
        cfg["run_config_rel"],
        cfg["center"][0],
        cfg["center"][1],
        cfg["Kx"],
        cfg["Ky"],
        cfg["rotation_deg"],
        cfg["fwhm0"],
    )

    if args.check:
        if not out.is_file():
            logger.error("[geometry] 目标 npz 不存在: {}", out)
            return 2
        diffs = compare(out, cfg)
        if diffs:
            logger.error("[geometry] {} 与现场标定不一致:\n  {}", _rel(out), "\n  ".join(diffs))
            return 1
        logger.info("[geometry] check clean: {} 与现场标定逐字段一致", _rel(out))
        return 0

    previous = _previous_center(out)
    arrays = build_arrays(cfg, cfg["run_config_rel"])
    out.parent.mkdir(parents=True, exist_ok=True)
    archived: Path | None = None
    if out.is_file() and not args.no_archive:
        archived = snapshot_name_for(out)
        if archived != out:
            shutil.copy2(out, archived)
            logger.info("[geometry] 旧 npz 已存档: {}", _rel(archived))
    np.savez(out, **arrays)
    logger.info("[geometry] 已写入 {} ({} 个字段)", _rel(out), len(arrays))

    if previous is not None:
        dcy = cfg["center"][0] - previous[0]
        dcx = cfg["center"][1] - previous[1]
        logger.info(
            "[geometry] 中心变化: Δcy={:+.3f} px Δcx={:+.3f} px (旧 {} → 新 {})",
            dcy,
            dcx,
            previous,
            cfg["center"],
        )
    diffs = compare(out, cfg)
    if diffs:
        logger.error("[geometry] 回读校验失败:\n  {}", "\n  ".join(diffs))
        return 1
    logger.info("[geometry] 回读校验通过 (center/Kx/Ky/rotation_deg/fwhm0/mirror)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
