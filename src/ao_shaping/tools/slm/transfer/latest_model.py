"""把最新拟合模型按 SLM 序列号保存到 ``data/slm_models/<serial>/latest``。

``latest`` 是目录, 含 ``frozen_parameters.json`` 及其侧车。替换流程:
写入同盘临时目录 → 旧 ``latest`` 改名为备份 → 临时目录 rename 成 ``latest`` →
删除备份。任何读者只会看到完整的 ``latest``。

序列号来源: 采集段在 SLM open 后写入 ``capture/.capture_done`` 的 ``slm_serial``。
读不到时跳过保存, 不报错。本模块只做文件写, 不打开任何设备。
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

from loguru import logger

from ao_shaping.utils.io.file import ROOT_DIR as _PROJECT_ROOT


def models_root() -> Path:
    """``SLM_MODELS_DIR`` 或 ``<项目根>/data/slm_models``。"""
    return Path(os.environ.get("SLM_MODELS_DIR", _PROJECT_ROOT / "data" / "slm_models"))


def serial_from_capture(session_dir: Path) -> str | None:
    """从会话目录下 ``capture/.capture_done`` 读 SLM 序列号 (无则 None)。"""
    candidate = Path(session_dir) / "capture" / ".capture_done"
    if not candidate.is_file():
        return None
    try:
        meta = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning(f"无法读 {candidate} 取序列号: {exc!r}")
        return None
    serial = meta.get("slm_serial")
    return str(serial) if serial else None


def save_latest_model(checkpoint: Path, serial: str | None) -> Path | None:
    """把 ``checkpoint`` 目录 (含 ``frozen_parameters.json``) 保存为该序列号的 latest。

    Returns:
        ``latest`` 目录; 缺文件或缺序列号时跳过, 返回 None。
    """
    checkpoint = Path(checkpoint)
    source = checkpoint / "frozen_parameters.json"
    if not source.is_file():
        logger.warning(f"latest 保存跳过: {source} 不存在")
        return None
    if not serial:
        logger.info("latest 保存跳过: 没有 SLM 序列号 (仿真/不可读)")
        return None

    latest = models_root() / str(serial) / "latest"
    latest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(dir=str(latest.parent), prefix=".latest.tmp-"))
    try:
        for item in checkpoint.iterdir():
            if item.is_file():
                shutil.copyfile(item, tmp_dir / item.name)
        backup = None
        if latest.exists():
            backup = latest.parent / f".latest.old-{tmp_dir.name}"
            latest.rename(backup)
        try:
            tmp_dir.rename(latest)
        except BaseException:
            if backup is not None and backup.exists():
                backup.rename(latest)
            raise
        if backup is not None:
            shutil.rmtree(backup, ignore_errors=True)
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
    logger.info("已按序列号保存最新模型: {} -> {}", source, latest / "frozen_parameters.json")
    return latest
