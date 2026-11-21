"""湍流整形仿真的公共挂具 —— 干扰注入与伴随产物的唯一实现。

``scripts/slm/slm_pib_sim_run.py`` 率先需要这两件事: ①把 :class:`SimDisturbance`
挂到进程级仿真系统并**抵抗 reset 丢失**; ②把干扰的实测幅度与屏序列写成报告的
伴随产物 (``disturbance.json`` / ``disturbance.npz``)。#253 的湍流整形 runner
需要同一套 —— 复制会漂移 (本仓库已多次因此返工), 所以收口到这里, 脚本与 runner
都只是消费方。

reset 丢失的机理 (勿回退): runner 的数字孪生补丁在函数体内 ``import`` 并调用
``slm_pib_sim.reset_system(seed)``, 只带 seed; 若挂具先建立过带扰动的系统, 这次
reset 会静默换成无扰动系统, 而伴随清单仍声称有扰动 —— 运行与自己的产物矛盾。
补丁的 import 在调用时解析模块属性, 所以包装属性即可根治。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from ao_shaping.drivers.sim.disturbance import SimDisturbance

#: 伴随产物文件名 (报告按此读取)。
COMPANION_JSON = "disturbance.json"
COMPANION_NPZ = "disturbance.npz"

_injected: SimDisturbance | None = None


def install_disturbance_system(
    disturbance: SimDisturbance, *, seed: int | None = None
) -> None:
    """把 ``disturbance`` 挂到进程级仿真系统, 并让之后的每次 reset 都重新挂上。

    Args:
        disturbance: 已构造的干扰模型。
        seed: 传 int 时立即以该种子重建系统并挂扰动; ``None`` 只安装包装,
            惰性首建时经包装自动挂上。
    """
    global _injected
    _injected = disturbance

    import ao_shaping.drivers.sim.slm_pib_sim as sim_module

    if getattr(sim_module.reset_system, "_disturbance_wrapper", False):
        if seed is not None:
            sim_module.reset_system(seed)
        return
    original = sim_module.reset_system

    def _reset_with_disturbance(seed=None, *, disturbance=None):
        if disturbance is None:
            disturbance = _injected
        return original(seed, disturbance=disturbance)

    _reset_with_disturbance._disturbance_wrapper = True  # type: ignore[attr-defined]
    sim_module.reset_system = _reset_with_disturbance
    if seed is not None:
        sim_module.reset_system(seed)


def write_disturbance_companion(
    disturbance: SimDisturbance,
    run_dir: Path,
    *,
    run_meta: dict[str, Any],
    archive_factor: int = 8,
    archive_max: int = 12,
) -> tuple[Path, Path]:
    """把干扰的实测幅度与屏序列写进 ``run_dir`` 的伴随产物。

    幅度一律**按实测** (``disturbance.stats()``), 绝不给解析值 —— 理由见
    :mod:`ao_shaping.drivers.sim.disturbance` 模块 docstring。

    Returns:
        ``(json_path, npz_path)``
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    archive = disturbance.archive(factor=archive_factor, max_count=archive_max)
    trace = disturbance.trace()
    npz_path = run_dir / COMPANION_NPZ
    np.savez_compressed(
        npz_path,
        screens=archive["screens"],
        streak_indices=archive["streak_indices"],
        call_rms=np.asarray(trace["call_rms"], dtype=np.float64),
        call_streak_index=np.asarray(trace["call_streak_index"], dtype=np.int32),
        archive_factor=np.int32(archive_factor),
    )

    measured = disturbance.stats()
    payload = {
        "enabled": bool(disturbance.enabled),
        "mode": disturbance.config.mode,
        "config": disturbance.to_dict()["config"],
        "measured": measured,
        "run": {
            **run_meta,
            "screens_archived": int(archive["screens"].shape[0]),
            "streaks_total": int(measured["streaks_used"]),
        },
    }
    json_path = run_dir / COMPANION_JSON
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    free = int(archive["screens"].shape[0])
    if free < int(measured["streaks_used"]):
        logger.info(
            "archived {} of {} used screens (sub-sampled for size); call_rms kept in full",
            free,
            int(measured["streaks_used"]),
        )
    return json_path, npz_path


#: 产物根目录家族 (与 ``slm_pib_sim_run`` 原始语义一致: 只认 ``slm_pib_*`` 且 resolve)。
DEFAULT_ARTIFACT_GLOB = "slm_pib_*"


def artifact_roots(data_root: Path | str, *, pattern: str = DEFAULT_ARTIFACT_GLOB) -> set[Path]:
    """当前 ``data_root/debug`` 下匹配的产物根目录 (resolve 后)。"""
    debug_root = Path(data_root) / "debug"
    if not debug_root.is_dir():
        return set()
    return {p.resolve() for p in debug_root.glob(pattern) if p.is_dir()}


def find_new_artifact_dir(
    data_root: Path | str, before: set[Path], *, pattern: str = DEFAULT_ARTIFACT_GLOB
) -> Path | None:
    """返回本次运行刚创建的产物根目录。

    用前后集合差而不是「按 mtime 取最新」, 否则并发会话会让这里挑错目录。
    差集为空时回落最新已有目录 (不匹配 ``pattern`` 的目录永不参与, 也不返回)。
    """
    after = artifact_roots(data_root, pattern=pattern)
    new = sorted(after - before, key=lambda p: p.stat().st_mtime, reverse=True)
    if new:
        return new[0]
    existing = sorted(after, key=lambda p: p.stat().st_mtime, reverse=True)
    return existing[0] if existing else None


def run_dir_of(artifact_root: Path) -> Path:
    """产物根目录里真正装 ``.pkl`` 的那层子目录 (报告按 ``slm_pib_*/*`` 收集)。"""
    subdirs = (
        [p for p in artifact_root.iterdir() if p.is_dir()]
        if artifact_root.is_dir()
        else []
    )
    with_pkl = [p for p in subdirs if any(p.glob("*.pkl"))]
    if with_pkl:
        return max(with_pkl, key=lambda p: p.stat().st_mtime)
    return artifact_root
