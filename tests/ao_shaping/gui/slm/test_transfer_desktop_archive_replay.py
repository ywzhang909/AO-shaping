"""归档回放弹窗的离线契约 (离屏, 不碰任何设备)。

钉住三件事:

* import 本模块不会拉起 ``drivers.ccd`` / ``drivers.slm`` —— 回放替身必须是纯读盘;
* 小尺寸合成 npz 能上屏: CCD 画面 / 相位预览 / 剖面都来自真实解析路径;
* 缺目录、坏文件都只是一行可见提示, 弹窗不抛异常、不关窗。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# 必须在任何 QApplication 之前设置: CI / 无显示环境走离屏后端。
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from ao_shaping.gui.slm.transfer_desktop import archive_replay as replay

_REPO = Path(__file__).resolve().parents[4]
_REAL_SESSION = _REPO / "data" / "debug" / "slm_transfer" / "physical_20261102c"
_REAL_CAPTURE = _REAL_SESSION / "anchored" / "capture_0000.npz"


@pytest.fixture(scope="module")
def app() -> QApplication:
    """离屏 QApplication (一个进程只能有一个)。"""
    instance = QApplication.instance()
    if instance is None:
        instance = QApplication([])
    return instance


def _write_capture(path: Path, *, seed: int = 0) -> None:
    """写一个形状很小、但键与真实归档一致的 npz。"""
    rng = np.random.default_rng(seed)
    height, width = 40, 48
    yy, xx = np.mgrid[0:height, 0:width]
    beam = 1500.0 * np.exp(
        -(((xx - 24) ** 2) + ((yy - 20) ** 2)) / (2.0 * 4.0**2)
    )
    frames = np.stack(
        [
            (rng.random((height, width)) * 40.0 + beam).astype(np.uint16)
            for _ in range(3)
        ]
    )
    grid = 8
    xs = np.linspace(10.0, 38.0, grid)
    ys = np.linspace(8.0, 32.0, grid)
    camera_xy = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1)
    core = np.zeros((grid, grid), dtype=bool)
    core[2:6, 2:6] = True

    panel_gray = rng.integers(0, 1024, size=(16, 24)).astype(np.uint16)
    panel_phase = (panel_gray.astype(np.float64) / 1023.0) * 2.0 * np.pi
    np.savez_compressed(
        path,
        raw_frames=frames,
        camera_xy=camera_xy,
        target=rng.random((grid, grid)).astype(np.float32),
        core=core,
        model_phase=panel_phase[:grid, :grid],
        panel_gray=panel_gray,
        panel_phase=panel_phase,
    )


@pytest.fixture()
def session_dir(tmp_path: Path) -> Path:
    anchored = tmp_path / "session" / "anchored"
    anchored.mkdir(parents=True)
    _write_capture(anchored / "capture_0000.npz", seed=0)
    _write_capture(anchored / "capture_0001.npz", seed=1)
    return tmp_path / "session"


def test_import_pulls_no_hardware_drivers() -> None:
    """新进程里 import 本模块, sys.modules 不得出现相机 / SLM 驱动。"""
    code = (
        "import sys\n"
        "import ao_shaping.gui.slm.transfer_desktop.archive_replay\n"
        "print([m for m in sys.modules if m.startswith("
        "'ao_shaping.drivers.ccd') or m.startswith('ao_shaping.drivers.slm')])\n"
    )
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONIOENCODING="utf-8")
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        cwd=str(_REPO),
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "[]", proc.stdout


def test_synthetic_capture_renders(app: QApplication, session_dir: Path) -> None:
    dialog = replay.ArchiveReplayDialog(session_dir)
    try:
        assert dialog.capture_count() == 2
        sample = dialog.current_sample
        assert sample is not None
        assert sample.frame_count == 3
        assert sample.display_frame.dtype == np.uint8
        assert sample.display_frame.ndim == 2
        # 裁剪到光斑附近, 一定小于原始 40x48。
        assert sample.display_frame.shape[0] < 40
        assert sample.display_frame.shape[1] < 48
        assert sample.phase_rgb.shape == (16, 24, 3)
        assert dialog.message_text() == ""
        assert dialog._phase.pixmap() is not None
        assert not dialog._phase.pixmap().isNull()

        dialog.set_index(1)
        assert dialog.current_index() == 1
        assert dialog.current_sample.path.name == "capture_0001.npz"

        # 越界钳位, 不抛异常。
        dialog.set_index(99)
        assert dialog.current_index() == 1
    finally:
        dialog.close()
        dialog.deleteLater()


def test_empty_directory_shows_message(app: QApplication, tmp_path: Path) -> None:
    dialog = replay.ArchiveReplayDialog(tmp_path)
    try:
        assert dialog.capture_count() == 0
        assert dialog.current_sample is None
        assert "没有" in dialog.message_text()
    finally:
        dialog.close()
        dialog.deleteLater()


def test_corrupt_capture_is_reported_not_raised(
    app: QApplication, tmp_path: Path
) -> None:
    anchored = tmp_path / "bad" / "anchored"
    anchored.mkdir(parents=True)
    (anchored / "capture_0000.npz").write_bytes(b"not-a-zip")
    dialog = replay.ArchiveReplayDialog(tmp_path / "bad")
    try:
        assert dialog.capture_count() == 1
        assert dialog.current_sample is None
        assert "读取失败" in dialog.message_text()
    finally:
        dialog.close()
        dialog.deleteLater()


@pytest.mark.skipif(
    not _REAL_CAPTURE.exists(), reason="真实归档 capture_0000.npz 不在本机"
)
def test_real_archive_capture_loads() -> None:
    sample = replay.load_capture(_REAL_CAPTURE)
    assert sample.frame_count == 3
    assert sample.raw_frame.ndim == 2
    assert sample.display_frame.shape == sample.raw_frame.shape
    assert sample.phase_rgb.shape == (1200, 1920, 3)
    assert sample.raw_peak > sample.raw_mean


@pytest.mark.skipif(
    not _REAL_CAPTURE.exists(), reason="真实归档会话不在本机"
)
def test_real_archive_dialog_opens(app: QApplication) -> None:
    dialog = replay.ArchiveReplayDialog(_REAL_SESSION)
    try:
        assert dialog.capture_count() >= 1
        assert dialog.current_sample is not None
        assert dialog.message_text() == ""
    finally:
        dialog.close()
        dialog.deleteLater()
