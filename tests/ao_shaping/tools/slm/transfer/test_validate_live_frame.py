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


# ---------------------------------------------------------------------------
# 目标形状线框 (2026-11-07): 实时/预测面板叠加 target 轮廓
# ---------------------------------------------------------------------------


def _star_mask() -> np.ndarray:
    """一块 40x40 目标掩膜: 中央竖直条, 方便断言轮廓映射。"""
    mask = np.zeros((40, 40), np.float32)
    mask[10:30, 18:22] = 1.0
    return mask


def test_frame_payload_carries_a_copied_target_mask():
    from ao_shaping.tools.slm.transfer.live_frame import frame_payload

    target = _star_mask()
    payload = frame_payload(frame=np.zeros((8, 8), np.uint8), label="x", target=target)
    assert payload["target"] is not None
    assert payload["target"].shape == target.shape
    assert payload["target"] is not target  # 拷贝, 不共享后端缓冲区
    target[:] = 0
    assert payload["target"].any()
    # 不传 target 时键为 None (旧调用不变)。
    assert frame_payload(frame=np.zeros((8, 8), np.uint8))["target"] is None


def test_block_frame_payload_carries_a_copied_target_mask():
    target = _star_mask()
    payload = validate.block_frame_payload(
        frame=np.zeros((8, 8), np.uint8), shape="star", block=0,
        block_index=1, blocks_total=1, iteration=1, iterations_total=1, remaining=0,
        target=target,
    )
    assert payload["target"] is not None
    assert payload["target"].shape == target.shape
    assert payload["target"] is not target
    assert payload["phase"] == "validate"
    # 省略 target 时保持 None (无轮廓)。
    plain = validate.block_frame_payload(
        frame=np.zeros((8, 8), np.uint8), shape="star", block=0,
        block_index=1, blocks_total=1, iteration=1, iterations_total=1, remaining=0,
    )
    assert plain["target"] is None


def test_emit_block_frame_loads_target_from_npz_when_omitted(capture):
    """块级负载的 target 缺省从采集 npz 的 ``target`` 字段加载。"""
    directory, name, _ = capture
    target = _star_mask()
    np.savez_compressed(directory / name, raw_frames=np.stack(
        [np.full((8, 10), 5, np.uint16)]), target=target)
    validate._CAPTURE_FRAMES.pop(name, None)

    seen: list[dict] = []
    _emit(seen.append, (directory, name, None))
    assert len(seen) == 1
    assert seen[0]["target"] is not None
    np.testing.assert_array_equal(seen[0]["target"], target)

    # 显式传入的 target 优先, 不开文件。
    explicit = _star_mask() * 0.5
    seen2: list[dict] = []
    _emit(seen2.append, (directory, name, None), target=explicit)
    assert seen2[0]["target"] is not None
    np.testing.assert_array_equal(seen2[0]["target"], explicit)


def test_emit_block_frame_tolerates_missing_target_field(capture):
    directory, name, _ = capture
    # 老采集 npz 无 ``target`` 字段: 负载 target 为 None, 不报错。
    validate._CAPTURE_FRAMES.pop(name, None)
    seen: list[dict] = []
    _emit(seen.append, (directory, name, None))
    assert len(seen) == 1
    assert seen[0]["target"] is None


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


def _emits_payload(body: list[str]) -> bool:
    """循环体是否包含一次 ``notify_frame(observer, frame_payload(...))`` 调用。

    调用是**多行**的 (``notify_frame(`` 换行 ``observer,`` 再换行 ``frame_payload(``),
    所以不能按单行子串匹配。这里把循环体拼接成一段 (去掉缩进), 再用正则找
    ``notify_frame(`` 与紧随其后的 ``frame_payload(`` (中间允许换行/参数行)。
    """
    import re

    joined = "\n".join(body)
    return re.search(r"notify_frame\s*\([^)]*frame_payload\s*\(", joined) is not None


def test_validate_emits_a_frame_inside_the_per_frame_read_loop():
    """取图循环里的每一次读出都要发帧 —— 这是本次改动的全部要点。

    用源码级的钉子而不是跑一遍设备: 会话段需要真相机/真 SLM, 而这里要钉的是
    「emit 在循环体内、不在循环后」这个结构性质。

    #205 第二十六批把会话段从 ``run`` 提为 ``_run_validate_session`` (上游一次
    合并把早先的拆分压回了 ``run``), 取图循环随之搬进该函数的嵌套 ``capture``。
    钉子因此改钉新的归属者, 并额外钉住 ``run`` 仍然委派给它 —— 否则归属者再被
    搬走时, 这个测试会静默地钉在一条 ``run`` 根本不再经过的路径上。

    第三十九批把嵌套闭包 ``capture`` 升为模块级 ``_session_capture``
    (``functools.partial`` 在原处绑定), 归属者链变成
    ``run → _run_validate_session → _session_capture``: 先钉住这条链,
    循环体结构性质再钉在最终归属者上, 曝光阶梯钉子仍属于会话段。
    """
    import inspect

    assert "_run_validate_session(" in inspect.getsource(validate.run)
    session = inspect.getsource(validate._run_validate_session)
    assert "functools.partial(" in session and "_session_capture" in session
    source = inspect.getsource(validate._session_capture)
    body = _loop_body(source, "for index in range(args.frames):")
    assert _emits_payload(body), body
    assert any('phase="validate"' in line for line in body)

    # 曝光阶梯的每一档也是一次读出, 同样逐帧上屏。
    assert 'phase="exposure"' in session


def test_capture_emits_a_frame_from_its_single_read_funnel():
    """``capture`` 唯一的取图漏斗: 光路检查/阶梯/探针都走同一个 ``read()``。

    #205 第 74 批把 ``read()`` 闭包提为模块级工厂 ``_cap_live_reader``,
    归属者链变成 ``capture → _cap_live_reader``: 先钉委派边, 漏斗结构
    性质再钉在工厂上。
    """
    import inspect

    from ao_shaping.tools.slm.transfer import capture as capture_mod

    source = inspect.getsource(capture_mod.capture)
    assert "read = _cap_live_reader(" in source
    factory = inspect.getsource(capture_mod._cap_live_reader)
    body = _loop_body(factory, "def read():")
    assert _emits_payload(body), body
    # 标签必须在光路检查之前就位, 否则那些帧会带着上一段的标签上屏。
    assert source.index("live_label") < source.index("_cap_verify_light(")


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

    # #205 第 74 批: read 闭包搬进 _cap_live_reader 工厂, 细胞改为形参传入。
    source = inspect.getsource(capture_mod.capture)
    assert "read = _cap_live_reader(" in source
    factory = inspect.getsource(capture_mod._cap_live_reader)
    # (1) read() 闭包把相位细胞带进负载 (多行调用, 逐行搜)。
    read_body = _loop_body(factory, "def read():")
    read_text = "\n".join(read_body)
    assert "phase_grid=live_phase[0]" in read_text
    # (2) 细胞在交给漏斗之前先在母函数就位, 并以形参进工厂作用域。
    assert source.index("live_phase = [None]") < source.index("read = _cap_live_reader(")
    assert "live_phase" in inspect.signature(capture_mod._cap_live_reader).parameters
    # (3) 光路检查与每条探针在写 SLM 之后落值; 平场是第一份相位。
    assert "live_phase[0] = np.zeros" in source
    assert "live_phase[0] = grid" in source


# ---------------------------------------------------------------------------
# HDR 探测 (2026-11-07): payload 带 hdr 标志 + hdr_image, 前端色调映射
# ---------------------------------------------------------------------------


def test_frame_payload_hdr_keys_default_off():
    from ao_shaping.tools.slm.transfer.live_frame import frame_payload

    payload = frame_payload(frame=np.zeros((8, 8), np.uint8))
    assert payload["hdr"] is False
    assert payload["hdr_image"] is None


def test_frame_payload_hdr_image_is_copied():
    from ao_shaping.tools.slm.transfer.live_frame import frame_payload

    hdr = np.linspace(0, 5000.0, 16).reshape(4, 4)
    payload = frame_payload(
        frame=np.zeros((8, 8), np.uint8), hdr=True, hdr_image=hdr
    )
    assert payload["hdr"] is True
    assert payload["hdr_image"] is not None
    assert payload["hdr_image"] is not hdr  # 拷贝, 不共享后端缓冲区
    hdr[:] = 0
    assert payload["hdr_image"].max() > 0


def test_block_frame_payload_hdr_keys():
    payload = validate.block_frame_payload(
        frame=np.zeros((8, 8), np.uint8), shape="star", block=0,
        block_index=1, blocks_total=1, iteration=1, iterations_total=1, remaining=0,
        hdr=True, hdr_image=np.ones((4, 4)),
    )
    assert payload["hdr"] is True
    assert payload["hdr_image"] is not None
    assert payload["hdr_image"].shape == (4, 4)


def test_validate_capture_loop_passes_hdr_when_enabled():
    """采集循环里 HDR 开启时, 显示帧负载带 hdr=True 与 hdr_image。

    源码级钉子 (不跑设备): 取图循环体内必须同时出现 ``hdr=True`` 判定与
    ``hdr_image=hdr_image``, 且 HDR 判定用 ``getattr(args, "hdr", False)`` 容错
    (旧调用方不传 hdr 字段也能跑)。#205 第三十九批起循环体归属
    ``_session_capture`` (由 ``_run_validate_session`` 经 ``functools.partial``
    绑定), 钉子随归属者迁移。
    """
    import inspect

    source = inspect.getsource(validate._session_capture)
    body = _loop_body(source, "for index in range(args.frames):")
    text = "\n".join(body)
    assert 'getattr(args, "hdr", False)' in text
    assert "hdr_image=hdr_image" in text
    assert "hdr=hdr_on" in text
