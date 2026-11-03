"""每块一次的 live ``on_frame`` 回调契约 (后端 → GUI 的唯一数据通道)。

钉住四件事:

* ``block_frame_payload`` 产生 plain-data 负载 (ndarray + int), 不含 Qt 对象;
* ``emit_block_frame`` 每个块只发一次, 取的是该块采集的**最后一帧**;
* 没有观察者 / 没有采集行时完全不发 (不构造负载、不开文件);
* 观察者抛异常不会让会话失败 (与 ``workflow._notify`` 的 on_stage 同一策略)。
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ao_shaping.tools.slm.transfer import validate


@pytest.fixture()
def capture(tmp_path):
    """一块采集的 npz: 3 帧, 每帧一个不同常量, 用来辨认「最后一帧」。"""
    raw = np.stack(
        [np.full((8, 10), v, dtype=np.uint16) for v in (10, 20, 30)]
    )
    name = "capture_0000.npz"
    np.savez_compressed(tmp_path / name, raw_frames=raw)
    # 内存缓存是模块级的, 同名文件可能命中上一次测试的缓存 —— 先清干净。
    validate._CAPTURE_FRAMES.pop(name, None)
    yield tmp_path, name, raw
    validate._CAPTURE_FRAMES.pop(name, None)


def _emit(on_frame, capture, **overrides):
    directory, name, _ = capture
    kwargs = dict(
        directory=directory,
        capture_row={"file": name, "label": "验收采集"},
        shape="triangle",
        block=2,
        block_index=3,
        blocks_total=4,
        iteration=12,
        iterations_total=12,
        remaining=12,
    )
    kwargs.update(overrides)
    validate.emit_block_frame(on_frame, **kwargs)


def test_payload_uses_the_last_frame_and_integer_progress(capture):
    seen: list[dict] = []
    _emit(seen.append, capture)
    assert len(seen) == 1
    payload = seen[0]
    raw = capture[2]
    np.testing.assert_array_equal(payload["frame"], raw[-1])
    assert payload["shape"] == "triangle"
    assert (payload["block"], payload["block_index"], payload["blocks_total"]) == (2, 3, 4)
    assert (payload["iteration"], payload["iterations_total"]) == (12, 12)
    assert payload["remaining"] == 12
    assert payload["label"] == "验收采集"
    for key in ("block", "block_index", "blocks_total", "iteration", "iterations_total", "remaining"):
        assert isinstance(payload[key], int)
    # 剖面是纯 numpy, GUI 可以直接画, 不必接触任何 Qt 对象。
    np.testing.assert_allclose(payload["profile_x"], raw[-1].mean(axis=0))
    np.testing.assert_allclose(payload["profile_y"], raw[-1].mean(axis=1))


def test_payload_copies_the_frame_so_reused_buffers_cannot_mutate_it():
    frame = np.ones((4, 4), dtype=np.uint16)
    payload = validate.block_frame_payload(
        frame=frame, shape="s", block=0, block_index=1, blocks_total=1,
        iteration=1, iterations_total=1, remaining=0,
    )
    frame[:] = 99
    assert payload["frame"].max() == 1


def test_no_observer_or_no_capture_row_means_no_callback(capture):
    seen: list[dict] = []
    _emit(None, capture)
    _emit(seen.append, capture, capture_row=None)
    assert seen == []


def test_a_raising_observer_does_not_escape(capture):
    def boom(_payload):
        raise RuntimeError("GUI 线程炸了")

    # 不抛 = 通过; 会话不能因为显示端的问题中断采集。
    _emit(boom, capture)


def test_a_broken_capture_is_skipped_not_raised(tmp_path):
    seen: list[dict] = []
    _emit(seen.append, (tmp_path, "does_not_exist.npz", None))
    assert seen == []


def test_run_defaults_on_frame_to_none():
    """签名默认值必须是 None —— 不传观察者的老调用行为不变。"""
    import inspect

    signature = inspect.signature(validate.run)
    assert signature.parameters["on_frame"].default is None


def test_run_resolves_observer_from_args_when_not_passed():
    """``args.on_frame`` 与关键字参数等价, 让 SimpleNamespace 调用点两种都能用。"""
    assert validate._resolve_observer(SimpleNamespace(on_frame=None), None) is None
    marker = object()
    assert validate._resolve_observer(SimpleNamespace(on_frame=marker), None) is marker
    assert validate._resolve_observer(SimpleNamespace(), marker) is marker
    other = object()
    assert validate._resolve_observer(SimpleNamespace(on_frame=marker), other) is other
