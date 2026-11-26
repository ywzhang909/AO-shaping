"""离线测试: transfer 整形相位 → 面板弧度的共享读入口。

契约来自 ``scripts/slm/_transfer_phase.py``：重放路径只展开模型相位，不需要
ramp 矩阵/几何校正，也不 import 任何设备驱动。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.slm._transfer_phase import (
    DEFAULT_TRANSFER_PHASE,
    load_transfer_phase,
)

ROOT = Path(__file__).resolve().parents[3]


def test_default_transfer_phase_is_the_latest_accepted_dense_final() -> None:
    path = ROOT / DEFAULT_TRANSFER_PHASE
    if not path.is_file():
        pytest.skip(f"冻结 transfer 产物不在本机 (data/debug 硬件数据未入库): {path}")
    resolved, model, panel = load_transfer_phase()
    assert resolved == DEFAULT_TRANSFER_PHASE
    assert model.shape == (640, 640)
    assert panel.shape == (1200, 1920)
    assert np.array_equal(model, np.load(ROOT / DEFAULT_TRANSFER_PHASE, allow_pickle=False))
    expected = np.repeat(np.repeat(model, 3, axis=0), 3, axis=1)[:1200]
    assert np.array_equal(panel, expected)


def test_npz_uses_model_phase_and_never_falls_back_to_panel_gray(tmp_path: Path) -> None:
    model = np.linspace(-np.pi, np.pi, 64 * 64, dtype=np.float32).reshape(64, 64)
    gray = np.zeros((1200, 1920), dtype=np.uint16)
    path = tmp_path / "capture.npz"
    np.savez_compressed(path, model_phase=model, panel_gray=gray)
    _, loaded, panel = load_transfer_phase(path)
    assert np.array_equal(loaded, model)
    assert panel.shape == (1200, 1920)
    assert not np.array_equal(panel, gray)

    bad = tmp_path / "gray_only.npz"
    np.savez_compressed(bad, panel_gray=gray)
    with pytest.raises(ValueError, match="model_phase"):
        load_transfer_phase(bad)


def test_loader_rejects_non_tileable_shape_and_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "bad.npy"
    np.save(path, np.zeros((630, 640), dtype=np.float32))
    with pytest.raises(ValueError, match="1200x1920"):
        load_transfer_phase(path)
    with pytest.raises(FileNotFoundError):
        load_transfer_phase(tmp_path / "missing.npy")


def test_loader_does_not_import_device_drivers(tmp_path: Path) -> None:
    """单独进程验证：导入/调用读入口后 ``ao_shaping.drivers.*`` 仍为空。

    同进程断言会被其它测试先 import 的驱动污染，所以这里起干净子进程。
    路径用合成 npy：默认常量指向 data/debug 冻结产物（未入库，CI 机上不存在）。
    """
    synthetic = tmp_path / "synthetic_transfer_phase.npy"
    np.save(synthetic, np.zeros((640, 640), dtype=np.float64))
    code = (
        "import sys\n"
        "from scripts.slm._transfer_phase import load_transfer_phase\n"
        f"load_transfer_phase({str(synthetic)!r})\n"
        "bad = [n for n in sys.modules if n.startswith('ao_shaping.drivers')]\n"
        "print(','.join(bad))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "", proc.stdout
