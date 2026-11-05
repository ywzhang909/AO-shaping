"""逐帧的 live ``on_frame`` 回调契约 (后端 → GUI 的唯一数据通道)。

钉住五件事:

* ``block_frame_payload`` 产生 plain-data 负载 (ndarray + int), 不含 Qt 对象;
* ``frame_payload`` 是逐帧负载: 拷贝、纯 int、不算剖面;
* ``emit_block_frame`` 每块发一次, 额外带上该块的优化器进度;
* **每一次 CCD 读出都发一帧** —— 取图循环内与曝光阶梯内各有一个 emit;
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
    raw = np.stack([np.full((8, 10), v, dtype=np.uint16) for v in (10, 20, 30)])
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
    assert (payload["block"], payload["block_index"], payload["blocks_total"]) == (
        2,
        3,
        4,
    )
    assert (payload["iteration"], payload["iterations_total"]) == (12, 12)
    assert payload["remaining"] == 12
    assert payload["label"] == "验收采集"
    for key in (
        "block",
        "block_index",
        "blocks_total",
        "iteration",
        "iterations_total",
        "remaining",
    ):
        assert isinstance(payload[key], int)
    # 剖面是纯 numpy, GUI 可以直接画, 不必接触任何 Qt 对象。
    np.testing.assert_allclose(payload["profile_x"], raw[-1].mean(axis=0))
    np.testing.assert_allclose(payload["profile_y"], raw[-1].mean(axis=1))


def test_payload_copies_the_frame_so_reused_buffers_cannot_mutate_it():
    frame = np.ones((4, 4), dtype=np.uint16)
    payload = validate.block_frame_payload(
        frame=frame,
        shape="s",
        block=0,
        block_index=1,
        blocks_total=1,
        iteration=1,
        iterations_total=1,
        remaining=0,
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


# ---------------------------------------------------------------------------
# 逐帧刷新 (2026-11-03): 每一次 CCD 读出都发一帧, 不再是每块一张快照
# ---------------------------------------------------------------------------


def test_frame_payload_is_a_copied_plain_dict_without_profiles():
    """逐帧负载: 拷贝 (相机缓冲区会复用)、纯 int、不算剖面。

    ``profile_x`` / ``profile_y`` 在整幅帧上是 8.7 ms, 而 ``LiveView`` 从图像自己
    重算 —— 逐帧发帧时这笔开销不再值得。
    """
    from ao_shaping.tools.slm.transfer.live_frame import frame_payload

    raw = np.arange(12, dtype=np.uint16).reshape(3, 4)
    payload = frame_payload(
        frame=raw, label="探针 12", phase="capture", frame_index=2, frames_total=3
    )
    raw[:] = 0  # 相机复用缓冲区
    assert payload["frame"].max() == 11
    assert payload["label"] == "探针 12"
    assert payload["phase"] == "capture"
    assert (payload["frame_index"], payload["frames_total"]) == (2, 3)
    assert "profile_x" not in payload and "profile_y" not in payload

    # 3 通道取第一通道; 真正的多维输入要报错而不是悄悄拍平。
    assert frame_payload(frame=np.zeros((2, 3, 3), np.uint8))["frame"].shape == (2, 3)
    with pytest.raises(ValueError):
        frame_payload(frame=np.zeros((2, 3, 4, 5), np.uint8))


def _loop_body(source: str, header: str) -> list[str]:
    """``source`` 里以 ``header`` 开头那个块的循环体 (按缩进判断块边界)。"""
    start = source.index(header)
    lines = source[start:].splitlines()[1:]
    body: list[str] = []
    indent = None
    for line in lines:
        if not line.strip():
            continue
        current = len(line) - len(line.lstrip())
        if indent is None:
            indent = current
        elif current < indent:
            break
        body.append(line)
    return body


def test_validate_emits_a_frame_inside_the_per_frame_read_loop():
    """取图循环里的每一次读出都要发帧 —— 这是本次改动的全部要点。

    用源码级的钉子而不是跑一遍设备: 会话段需要真相机/真 SLM, 而这里要钉的是
    「emit 在循环体内、不在循环后」这个结构性质。

    #205 第二十六批把会话段从 ``run`` 提为 ``_run_validate_session`` (上游一次
    合并把早先的拆分压回了 ``run``), 取图循环随之搬进该函数的嵌套 ``capture``。
    钉子因此改钉新的归属者, 并额外钉住 ``run`` 仍然委派给它 —— 否则归属者再被
    搬走时, 这个测试会静默地钉在一条 ``run`` 根本不再经过的路径上。
    """
    import inspect

    source = inspect.getsource(validate._run_validate_session)
    assert "_run_validate_session(" in inspect.getsource(validate.run)
    body = _loop_body(source, "for index in range(args.frames):")
    assert any("notify_frame(observer, frame_payload(" in line for line in body), body
    assert any('phase="validate"' in line for line in body)

    # 曝光阶梯的每一档也是一次读出, 同样逐帧上屏。
    assert 'phase="exposure"' in source


def test_capture_emits_a_frame_from_its_single_read_funnel():
    """``capture.read()`` 是采集段唯一的取图漏斗: 光路检查/阶梯/探针都走它。"""
    import inspect

    from ao_shaping.tools.slm.transfer import capture as capture_mod

    source = inspect.getsource(capture_mod.capture)
    body = _loop_body(source, "def read():")
    assert any("notify_frame(observer, frame_payload(" in line for line in body), body
    # 标签必须在光路检查之前就位, 否则那些帧会带着上一段的标签上屏。
    assert source.index("live_label") < source.index("verify_light(")


def test_capture_read_funnel_threads_the_displayed_phase_grid():
    """采集段逐帧负载也要带 ``phase_grid`` —— 用户要"每次给 SLM 发相位都展示"。

    不可用源码钉子钉: ``read()`` 闭包把 ``live_phase`` 细胞取出来放进负载, 而
    ``live_phase`` 只在真的 ``slm.display_data`` 之前落值。这里钉三点结构性定律:
    (1) ``read()`` 负载里出现 ``phase_grid=live_phase[0]``; (2) 该细胞在读漏斗的
    作用域里先初始化 (读到它之前已定义); (3) 探针循环里 ``slm.display_data(gray)``
    之后紧跟落值 —— 相位写进 SLM 的那一帧起就带上它, 光路检查的平场也走同一条路。
    """
    import inspect

    from ao_shaping.tools.slm.transfer import capture as capture_mod

    source = inspect.getsource(capture_mod.capture)
    # (1) read() 闭包把相位细胞带进负载 (多行调用, 逐行搜)。
    read_body = _loop_body(source, "def read():")
    read_text = "\n".join(read_body)
    assert "phase_grid=live_phase[0]" in read_text
    # (2) 细胞在 read 定义之前就位 (同作用域, 不含 probe_read 那种子闭包)。
    assert source.index("live_phase = [None]") < source.index("def read():")
    # (3) 光路检查与每条探针在写 SLM 之后落值; 平场是第一份相位。
    assert "live_phase[0] = np.zeros" in source
    assert "live_phase[0] = grid" in source
