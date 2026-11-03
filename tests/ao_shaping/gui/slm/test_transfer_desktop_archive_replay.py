"""归档回放弹窗的离线契约 (离屏, 不碰任何设备、不依赖真实 23.7 GB 历史树)。

钉住几件事:

* import 本模块不会拉起 ``drivers.ccd`` / ``drivers.slm`` —— 回放替身必须是纯读盘;
* :func:`discover_archives` 递归扫整个根、只认 ``*.pkl``、从新到旧、扫描阶段不解码;
* 三种 pkl 口味 (capture_records / 迭代相位 / 拟合指标) 都能回放并按需降级;
* 缺根、坏文件、兄弟 npz 缺失都只是一行可见提示, 弹窗不抛异常、不关窗。

所有 pickle 都用真实 ``pickle.dump`` 字节写到 ``tmp_path``, 键形状对齐真实归档。
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
from pathlib import Path

# 必须在任何 QApplication 之前设置: CI / 无显示环境走离屏后端。
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QPushButton, QWidget

from ao_shaping.gui.slm.transfer_desktop import archive_replay as replay

_REPO = Path(__file__).resolve().parents[4]


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
    beam = 1500.0 * np.exp(-(((xx - 24) ** 2) + ((yy - 20) ** 2)) / (2.0 * 4.0**2))
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
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _dump(path: Path, records: list[dict]) -> Path:
    """用真实 pickle 字节把 ``list[dict]`` 写到 ``path`` (父目录自动建)。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(records, handle)
    return path


def _write_capture_records(folder: Path, n: int = 2) -> Path:
    """``capture_records.pkl`` + n 个兄弟 ``capture_NNNN.npz``, 键形状对齐真实归档。"""
    folder.mkdir(parents=True, exist_ok=True)
    names = []
    for i in range(n):
        name = f"capture_{i:04d}.npz"
        _write_capture(folder / name, seed=i)
        names.append(name)
    records = [
        {
            "label": f"blk{i}",
            "file": nm,
            "frames": [],
            "metrics": {
                "cv": 0.1 * i,
                "mean": 30.0,
                "ee_roi": 0.5,
                "target_peak": 1600.0,
                "target_pearson": 0.4,
            },
            "per_frame": [],
            "peak": 1000.0 + i,
            "core_peak": 900.0,
            "saturation_pixels": 0,
            "exposure_ms": 5.0,
            "timestamp_utc": "2026-11-02T00:00:00Z",
            "monotonic_ns": i,
        }
        for i, nm in enumerate(names)
    ]
    return _dump(folder / "capture_records.pkl", records)


def _write_iterations_phase(path: Path, n: int = 3) -> Path:
    """相位口味 ``iterations.pkl``: 每条带 ``phase`` (弧度 ndarray) + ``terms``。"""
    rng = np.random.default_rng(123)
    records = [
        {
            "shape": "square",
            "block": 0,
            "iteration": i,
            "phase": (rng.random((16, 16)).astype(np.float32) * 6.28 - 3.14),
            "terms": {
                "step": "gd",
                "mode": "x",
                "loss": 1.0 - 0.1 * i,
                "ee_roi": 0.4,
                "l_cv": 0.2,
                "shape_loss": 0.3,
                "phase_rms": 0.5,
                "gradient_norm": 0.01,
            },
            "timestamp_utc": "2026-11-02T00:00:00Z",
            "monotonic_ns": i,
            "end_monotonic_ns": i + 1,
        }
        for i in range(n)
    ]
    return _dump(path, records)


def _write_iterations_fit(path: Path, n: int = 4) -> Path:
    """拟合口味 ``iterations.pkl``: ``loss`` + ``parameters``, 无 phase / 无 CCD。"""
    records = [
        {
            "iteration": i,
            "loss": 0.7 - 0.01 * i,
            "parameters": {
                "defocus_diopter": 0.01,
                "origin": [0.0, 0.0],
                "amplitude_center": 1.0,
                "amplitude_width": 2.0,
                "residual_noll_radians": [0.0],
                "gain": 1.0,
                "background": 10.0,
                "sigma_ccd_px": 1.5,
                "bias_counts": 0.0,
            },
            "metrics": [{"correlation": 0.9, "relative_l2": 0.1}],
            "timestamp_utc": "2026-11-02T00:00:00Z",
            "monotonic_ns": i,
        }
        for i in range(n)
    ]
    return _dump(path, records)


@pytest.fixture()
def tree(tmp_path: Path) -> Path:
    """一个含三种口味、多层会话目录的小型 debug 根。"""
    root = tmp_path / "debug" / "slm_transfer"
    _write_capture_records(root / "s1" / "anchored", n=2)
    _write_iterations_phase(root / "s1" / "anchored" / "iterations.pkl", n=3)
    _write_iterations_fit(root / "s2" / "fit" / "full" / "iterations.pkl", n=4)
    return root


def _select(dialog: replay.ArchiveReplayDialog, predicate) -> replay.ArchiveEntry:
    """按 ``archives()`` 顺序选中第一个满足谓词的归档 (无过滤时 == 列表顺序)。"""
    for row, entry in enumerate(dialog.archives()):
        if predicate(entry):
            dialog.select_archive(row)
            return entry
    raise AssertionError("没有匹配的归档")


def _visible_rows(dialog: replay.ArchiveReplayDialog) -> int:
    """过滤后仍可见的行数 (过滤是 ``setRowHidden``, 不是重建)。"""
    return sum(
        1
        for row in range(dialog._table.rowCount())
        if not dialog._table.isRowHidden(row)
    )


def _dump_button(dialog: replay.ArchiveReplayDialog, row: int) -> QPushButton:
    """「操作」列里那一行的归档按钮。"""
    widget = dialog._table.cellWidget(row, 3)
    assert isinstance(widget, QPushButton)
    return widget


# ---------------------------------------------------------------- import 契约
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


# ------------------------------------------------------------ discover_archives
def test_discover_archives_recursive_pkl_only_newest_first(tmp_path: Path) -> None:
    root = tmp_path / "root"
    p_old = _write_capture_records(root / "s1" / "anchored", n=2)
    p_mid = _write_iterations_phase(root / "s1" / "anchored" / "iterations.pkl")
    p_new = _write_iterations_fit(root / "s2" / "fit" / "full" / "iterations.pkl")
    # 混入非 pkl 文件: npz / txt 都必须被忽略。
    _write_capture(root / "s1" / "anchored" / "loose.npz", seed=7)
    (root / "notes.txt").write_text("ignore me", encoding="utf-8")

    # 受控 mtime: old < mid < new, 期望从新到旧。
    os.utime(p_old, (1000.0, 1000.0))
    os.utime(p_mid, (2000.0, 2000.0))
    os.utime(p_new, (3000.0, 3000.0))

    entries = replay.discover_archives(root)
    assert [e.path for e in entries] == [p_new, p_mid, p_old]
    assert len(entries) == 3
    assert all(e.path.suffix == ".pkl" for e in entries)
    assert all(e.size_bytes > 0 for e in entries)
    # 递归: 相对路径含多层会话目录。
    assert any(e.relative.startswith("s2/fit/full/") for e in entries)
    # kind 由 stem 派生。
    assert entries[0].kind == "iterations"
    assert entries[2].kind == "capture_records"


def test_discover_archives_same_mtime_ties_by_relative(tmp_path: Path) -> None:
    root = tmp_path / "root"
    a = _dump(root / "b" / "iterations.pkl", [{"iteration": 0, "loss": 1.0}])
    b = _dump(root / "a" / "iterations.pkl", [{"iteration": 0, "loss": 1.0}])
    os.utime(a, (5000.0, 5000.0))
    os.utime(b, (5000.0, 5000.0))
    entries = replay.discover_archives(root)
    assert [e.path for e in entries] == [b, a]  # 'a/...' < 'b/...'


def test_discover_archives_missing_empty_and_file_root(tmp_path: Path) -> None:
    assert replay.discover_archives(tmp_path / "does_not_exist") == []
    empty = tmp_path / "empty"
    empty.mkdir()
    assert replay.discover_archives(empty) == []
    a_file = tmp_path / "afile"
    a_file.write_text("x", encoding="utf-8")
    assert replay.discover_archives(a_file) == []


def test_discover_archives_never_opens_pkl(tmp_path: Path) -> None:
    """扫描阶段只枚举文件名: 一个解不开的坏 pkl 也必须被列出而不抛异常。"""
    root = tmp_path / "root"
    root.mkdir()
    bad = root / "garbage.pkl"
    bad.write_bytes(b"not a pickle at all")
    entries = replay.discover_archives(root)
    assert len(entries) == 1
    assert entries[0].path == bad
    # 若扫描阶段 pickle.load 过, 这里早就抛了。


# --------------------------------------------------------------- 既有纯逻辑
def test_discover_captures_and_load_capture_still_work(tmp_path: Path) -> None:
    session = tmp_path / "session"
    _write_capture_records(session / "anchored", n=2)
    paths = replay.discover_captures(session)
    assert len(paths) == 2
    sample = replay.load_capture(paths[0])
    assert isinstance(sample, replay.CaptureData)
    assert sample.frame_count == 3
    assert sample.display_frame.dtype == np.uint8
    assert sample.display_frame.ndim == 2
    assert sample.phase_rgb.shape == (16, 24, 3)


def test_phase_to_preview_grayscale_and_twilight() -> None:
    gray = np.linspace(0, 1023, 16, dtype=np.float32).reshape(4, 4)
    assert replay.phase_to_preview(gray, colormap="灰度").shape == (4, 4, 3)
    assert replay.phase_to_preview(gray).shape == (4, 4, 3)


# ---------------------------------------------------------------- load_archive
def test_load_archive_capture_records_resolves_sibling_npz(tmp_path: Path) -> None:
    folder = tmp_path / "root" / "s1" / "anchored"
    pkl = _write_capture_records(folder, n=2)
    entry = next(
        e for e in replay.discover_archives(tmp_path / "root") if e.path == pkl
    )
    payload = replay.load_archive(entry)
    assert payload.flavor == "capture_records"
    assert payload.notice == ""
    assert len(payload.records) == 2
    for i, rec in enumerate(payload.records):
        assert rec.index == i
        assert rec.label == f"blk{i}"
        assert rec.ccd_path == folder / f"capture_{i:04d}.npz"
        assert rec.ccd_path.is_file()
        assert rec.phase is None
        assert "cv" in rec.metrics


def test_load_archive_iterations_phase_yields_phase(tmp_path: Path) -> None:
    pkl = _write_iterations_phase(
        tmp_path / "root" / "anchored" / "iterations.pkl", n=3
    )
    entry = next(
        e for e in replay.discover_archives(tmp_path / "root") if e.path == pkl
    )
    payload = replay.load_archive(entry)
    assert payload.flavor == "iterations_phase"
    assert "无 CCD 帧" in payload.notice
    assert len(payload.records) == 3
    assert payload.records[0].phase is not None
    assert payload.records[0].phase.shape == (16, 16)
    assert payload.records[0].ccd_path is None
    assert "loss" in payload.records[0].metrics


def test_load_archive_iterations_fit_degrades(tmp_path: Path) -> None:
    pkl = _write_iterations_fit(
        tmp_path / "root" / "fit" / "full" / "iterations.pkl", n=4
    )
    entry = next(
        e for e in replay.discover_archives(tmp_path / "root") if e.path == pkl
    )
    payload = replay.load_archive(entry)
    assert payload.flavor == "iterations_fit"
    assert "拟合指标" in payload.notice
    assert all(r.phase is None and r.ccd_path is None for r in payload.records)
    assert "loss" in payload.records[0].metrics
    assert "gain" in payload.records[0].metrics


def test_load_archive_corrupt_raises_within_load_errors(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "broken.pkl").write_bytes(b"not a pickle")
    entry = replay.discover_archives(root)[0]
    with pytest.raises(replay._LOAD_ERRORS):
        replay.load_archive(entry)


# --------------------------------------------------------------------- 弹窗
def test_dialog_lists_all_archives(app: QApplication, tree: Path) -> None:
    dialog = replay.ArchiveReplayDialog(tree)
    try:
        assert dialog.archive_count() == 3
        assert dialog._table.rowCount() == 3
        assert len(dialog.archives()) == 3
        assert "3 个 pkl" in dialog._header.text()
    finally:
        dialog.close()
        dialog.deleteLater()


def test_dialog_capture_records_full_three_views(app: QApplication, tree: Path) -> None:
    dialog = replay.ArchiveReplayDialog(tree)
    try:
        entry = _select(dialog, lambda e: e.kind == "capture_records")
        assert dialog.current_archive == entry
        assert dialog.capture_count() == 2
        sample = dialog.current_sample
        assert sample is not None
        assert sample.frame_count == 3
        pixmap = dialog._phase.pixmap()
        assert pixmap is not None and not pixmap.isNull()
        assert dialog.message_text() == ""
        # 加载后名称列 tooltip 补上记录数。
        assert "2 条记录" in dialog._table.item(dialog._active_row, 0).toolTip()

        dialog.set_index(1)
        assert dialog.current_index() == 1
        assert dialog.current_sample.path.name == "capture_0001.npz"
        dialog.set_index(99)  # 越界钳位, 不抛。
        assert dialog.current_index() == 1
    finally:
        dialog.close()
        dialog.deleteLater()


def test_dialog_iterations_phase_shows_phase_no_ccd(
    app: QApplication, tree: Path
) -> None:
    dialog = replay.ArchiveReplayDialog(tree)
    try:
        _select(dialog, lambda e: e.relative.endswith("s1/anchored/iterations.pkl"))
        assert dialog.current_sample is None  # 没有 CCD
        pixmap = dialog._phase.pixmap()
        assert pixmap is not None and not pixmap.isNull()  # 但有相位预览
        assert "无 CCD 帧" in dialog.message_text()
        assert "loss" in dialog._meta.text()
    finally:
        dialog.close()
        dialog.deleteLater()


def test_dialog_iterations_fit_degrades_to_message(
    app: QApplication, tree: Path
) -> None:
    dialog = replay.ArchiveReplayDialog(tree)
    try:
        _select(dialog, lambda e: e.relative.endswith("s2/fit/full/iterations.pkl"))
        assert dialog.current_sample is None
        assert "拟合指标" in dialog.message_text()
        assert "拟合指标" in dialog._phase.text()
        assert "loss" in dialog._meta.text()
    finally:
        dialog.close()
        dialog.deleteLater()


def test_dialog_corrupt_pkl_reported_not_raised(
    app: QApplication, tmp_path: Path
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "broken.pkl").write_bytes(b"not a pickle")
    dialog = replay.ArchiveReplayDialog(root)
    try:
        assert dialog.archive_count() == 1
        assert "读取失败" in dialog.message_text()
        assert dialog.current_sample is None
        assert dialog.capture_count() == 0
    finally:
        dialog.close()
        dialog.deleteLater()


def test_dialog_missing_sibling_npz_marks_unavailable(
    app: QApplication, tmp_path: Path
) -> None:
    root = tmp_path / "root"
    _dump(
        root / "s1" / "anchored" / "capture_records.pkl",
        [{"label": "ghost", "file": "capture_9999.npz", "metrics": {"cv": 0.2}}],
    )
    dialog = replay.ArchiveReplayDialog(root)
    try:
        assert dialog.capture_count() == 1  # 记录保留, 不被丢弃
        assert dialog.current_sample is None
        assert "缺失" in dialog.message_text()
    finally:
        dialog.close()
        dialog.deleteLater()


def test_dialog_empty_root_shows_message(app: QApplication, tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    dialog = replay.ArchiveReplayDialog(empty)
    try:
        assert dialog.capture_count() == 0
        assert dialog.current_sample is None
        assert "没有" in dialog.message_text()
    finally:
        dialog.close()
        dialog.deleteLater()


def test_filter_narrows_archive_list(app: QApplication, tree: Path) -> None:
    dialog = replay.ArchiveReplayDialog(tree)
    try:
        assert _visible_rows(dialog) == 3
        dialog.apply_filter("fit")
        assert _visible_rows(dialog) == 1
        dialog.apply_filter("")
        assert _visible_rows(dialog) == 3
    finally:
        dialog.close()
        dialog.deleteLater()


# ------------------------------------------------------- 分列展示 / 归档 / 自动播放
def test_dialog_table_shows_four_columns(app: QApplication, tree: Path) -> None:
    """名称 / 时间 / 大小 / 操作: 名称是完整相对路径 (395 个归档只有两个文件名)。"""
    dialog = replay.ArchiveReplayDialog(tree)
    try:
        headers = [
            dialog._table.horizontalHeaderItem(column).text()
            for column in range(dialog._table.columnCount())
        ]
        assert headers == ["名称", "时间", "大小", "操作"]
        assert dialog._table.rowCount() == 3
        for row in range(dialog._table.rowCount()):
            assert isinstance(dialog._table.cellWidget(row, 3), QPushButton)
        entry = dialog.archives()[0]
        assert dialog._table.item(0, 0).text() == entry.relative
        assert dialog._table.item(0, 0).toolTip().startswith(str(entry.path))
        assert dialog._table.item(0, 1).text() == replay.format_mtime(entry.mtime)
        assert dialog._table.item(0, 2).text() == replay._human_size(entry.size_bytes)
        # 从新到旧: 时间列带 mtime, 且单调不增。
        times = [
            dialog._table.item(row, 1).data(Qt.ItemDataRole.UserRole)
            for row in range(dialog._table.rowCount())
        ]
        assert times == sorted(times, reverse=True)
    finally:
        dialog.close()
        dialog.deleteLater()


def test_discover_archives_skips_dump_subtree(tmp_path: Path) -> None:
    """ "已归档的 pkl 落在 ``dump/`` 下, 不能再出现在待归档列表里。"""
    root = tmp_path / "root"
    _write_capture_records(root / "s1" / "anchored", n=1)
    dumped = _write_iterations_phase(
        root / replay.DUMP_DIRNAME / "s1" / "iterations.pkl", n=1
    )
    entries = replay.discover_archives(root)
    assert [entry.relative for entry in entries] == ["s1/anchored/capture_records.pkl"]
    assert dumped.is_file()  # 只是不列出, 文件不动


def test_archive_to_dump_keeps_relative_path(tmp_path: Path) -> None:
    """目标 = ``<根>/dump/<相对路径>``; 只搬 pkl, 兄弟 npz 留在原处。"""
    root = tmp_path / "root"
    pkl = _write_capture_records(root / "s1" / "anchored", n=1)
    sibling = pkl.parent / "capture_0000.npz"
    entry = next(e for e in replay.discover_archives(root) if e.path == pkl)
    dest = replay.archive_to_dump(entry)
    assert (
        dest == root / replay.DUMP_DIRNAME / "s1" / "anchored" / "capture_records.pkl"
    )
    assert dest.is_file()
    assert not pkl.exists()
    assert sibling.is_file()
    assert replay.discover_archives(root) == []


def test_archive_to_dump_refuses_overwrite(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _write_capture_records(root / "s1" / "anchored", n=1)
    first = replay.discover_archives(root)[0]
    replay.archive_to_dump(first)
    # 原位再冒出一个同名 pkl: 目标已存在 → 拒绝, 绝不覆盖。
    again = _write_capture_records(root / "s1" / "anchored", n=1)
    entry = next(e for e in replay.discover_archives(root) if e.path == again)
    with pytest.raises(FileExistsError):
        replay.archive_to_dump(entry)
    assert again.is_file()


def test_archive_to_dump_raises_on_move_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    pkl = _write_capture_records(root / "s1" / "anchored", n=1)
    entry = next(e for e in replay.discover_archives(root) if e.path == pkl)

    def _boom(*_args, **_kwargs):
        raise OSError("磁盘不见了")

    monkeypatch.setattr(replay.shutil, "move", _boom)
    with pytest.raises(OSError):
        replay.archive_to_dump(entry)
    assert pkl.is_file()


def test_dump_button_needs_confirmation(
    app: QApplication, tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dialog = replay.ArchiveReplayDialog(tree)
    try:
        entry = _select(dialog, lambda e: e.kind == "capture_records")
        monkeypatch.setattr(dialog, "_confirm_dump", lambda _entry: False)
        _dump_button(dialog, dialog._row_of(entry)).click()
        assert entry.path.is_file()
        assert dialog.archive_count() == 3
        assert dialog._table.rowCount() == 3
        assert "已取消" in dialog.message_text()
    finally:
        dialog.close()
        dialog.deleteLater()


def test_dump_button_moves_current_archive_out_of_list(
    app: QApplication, tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dialog = replay.ArchiveReplayDialog(tree)
    try:
        entry = _select(dialog, lambda e: e.kind == "capture_records")
        sibling = entry.path.parent / "capture_0000.npz"
        monkeypatch.setattr(dialog, "_confirm_dump", lambda _entry: True)
        dialog.start_playback()
        assert dialog.is_playing()
        _dump_button(dialog, dialog._row_of(entry)).click()
        dest = tree / replay.DUMP_DIRNAME / entry.relative
        assert not entry.path.is_file()
        assert dest.is_file()
        assert sibling.is_file()  # 兄弟 npz 不动
        assert dialog.archive_count() == 2
        assert dialog._table.rowCount() == 2
        assert dialog.current_archive is None
        assert dialog.current_sample is None
        assert not dialog.is_playing()
        assert "已归档" in dialog.message_text()
        assert "2 个 pkl" in dialog._header.text()
    finally:
        dialog.close()
        dialog.deleteLater()


def test_dump_button_on_other_row_keeps_current_archive(
    app: QApplication, tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """按归档按钮不该顺手把那一行加载进来。"""
    dialog = replay.ArchiveReplayDialog(tree)
    try:
        current = _select(dialog, lambda e: e.kind == "capture_records")
        other = next(
            e
            for e in dialog.archives()
            if e.relative.endswith("s2/fit/full/iterations.pkl")
        )
        monkeypatch.setattr(dialog, "_confirm_dump", lambda _entry: True)
        _dump_button(dialog, dialog._row_of(other)).click()
        assert not other.path.is_file()
        assert dialog.current_archive == current
        assert dialog.current_sample is not None
        assert dialog.archive_count() == 2
    finally:
        dialog.close()
        dialog.deleteLater()


def test_autoplay_interval_follows_fps(app: QApplication, tree: Path) -> None:
    dialog = replay.ArchiveReplayDialog(tree)
    try:
        assert replay.fps_to_interval_ms(2.0) == 500
        assert replay.fps_to_interval_ms(10.0) == 100
        assert replay.fps_to_interval_ms(30.0) == 33
        _select(dialog, lambda e: e.kind == "capture_records")
        dialog.set_fps(10.0)
        dialog.start_playback()
        assert dialog.is_playing()
        assert dialog._play_timer.interval() == 100
        assert dialog._play.text() == "■ 停止"
        dialog.set_fps(25.0)  # 播放中改 fps, 下一拍生效
        assert dialog._play_timer.interval() == 40
        dialog.stop_playback()
        assert not dialog.is_playing()
        assert dialog._play.text() == "▶ 自动播放"
    finally:
        dialog.close()
        dialog.deleteLater()


def test_autoplay_wraps_at_last_record(app: QApplication, tree: Path) -> None:
    """走到最后一条回到第 0 条 (指标口味: 没有画面也要推进)。"""
    dialog = replay.ArchiveReplayDialog(tree)
    try:
        _select(dialog, lambda e: e.relative.endswith("s2/fit/full/iterations.pkl"))
        total = dialog.capture_count()
        assert total == 4
        dialog.set_index(total - 1)
        dialog.start_playback()
        dialog._play_timer.timeout.emit()
        assert dialog.current_index() == 0
        dialog._play_timer.timeout.emit()
        assert dialog.current_index() == 1
        dialog.stop_playback()
    finally:
        dialog.close()
        dialog.deleteLater()


def test_autoplay_stops_on_switch_and_manual_step(
    app: QApplication, tree: Path
) -> None:
    dialog = replay.ArchiveReplayDialog(tree)
    try:
        _select(dialog, lambda e: e.kind == "capture_records")
        dialog.start_playback()
        assert dialog.is_playing()
        _select(dialog, lambda e: e.relative.endswith("s1/anchored/iterations.pkl"))
        assert not dialog.is_playing()  # 换归档 → 停

        dialog.start_playback()
        dialog._on_next()  # 手动翻 → 停
        assert not dialog.is_playing()

        dialog.start_playback()
        dialog.close()
        assert not dialog.is_playing()  # 关窗 → 停
    finally:
        dialog.deleteLater()


# --------------------------------------------------------- open_archive_replay
def test_open_archive_replay_defaults_to_debug_root(
    app: QApplication, tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(replay, "DEBUG_ROOT", tree)
    parent = QWidget()
    dialog = replay.open_archive_replay(parent)
    try:
        assert dialog.archive_count() == 3
        assert parent._archive_replay_dialog is dialog
    finally:
        dialog.close()
        dialog.deleteLater()


def test_open_archive_replay_preselects_session(app: QApplication, tree: Path) -> None:
    parent = QWidget()
    dialog = replay.open_archive_replay(parent, tree, preselect=tree / "s1")
    try:
        assert dialog.current_archive is not None
        assert dialog.current_archive.relative.startswith("s1/")
        # 优先选中该会话下的 capture_records.pkl。
        assert dialog.current_archive.kind == "capture_records"
    finally:
        dialog.close()
        dialog.deleteLater()
