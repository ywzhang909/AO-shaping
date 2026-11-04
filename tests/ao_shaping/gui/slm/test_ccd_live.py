"""``ccd_live`` 的离线单元测试。

全部离线 —— 永不打开真相机, ``create_camera`` 与相机方法一律用假替身。真实
``@st.fragment(run_every=...)`` 需要 streamlit 运行时, 测试通过把模块的 ``st`` 换成
最小记录替身驱动纯/离线逻辑; 直播实质内容经 :func:`_render_live_frame` 直接跑。

另用 AST 钉死两条渲染纪律 (scope 仅本文件):
* 源码无 ``st.divider()``;
* 源码无 ``type="primary"``。
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from ao_shaping.gui.slm import ccd_live as gui

SOURCE = Path(gui.__file__)


# ---------------------------------------------------------------------------
# streamlit 最小记录替身 (stub)
# ---------------------------------------------------------------------------


class _St:
    """只记录 ``session_state`` / info / warning / caption / image / button 的最小替身。

    省略 ``st.fragment`` 装饰器的实装由测试在驱动前以 no-op 关闭
    (``gui._FRAGMENT_ENABLED = False``), 避免触碰 streamlit 运行时。
    """

    def __init__(self, session_state: dict[str, Any] | None = None) -> None:
        self.session_state = session_state or {}
        self.infos: list[str] = []
        self.warnings: list[str] = []
        self.captions: list[str] = []
        self.images: list[np.ndarray] = []
        self.button_labels: list[str] = []
        self.button_values = {"ccd_live_toggle": False, "ccd_live_release": False}
        self.number_input_values: dict[str, float] = {}
        self._rerun_count = 0

    def set_page_config(self, **kwargs: Any) -> None:
        return None

    def caption(self, body: str, **kwargs: Any) -> None:
        self.captions.append(str(body))

    def info(self, body: str, **kwargs: Any) -> None:
        self.infos.append(str(body))

    def warning(self, body: str, **kwargs: Any) -> None:
        self.warnings.append(str(body))

    def image(self, image: np.ndarray, **kwargs: Any) -> None:
        self.images.append(np.asarray(image))

    def button(self, label: str, **kwargs: Any) -> bool:
        self.button_labels.append(str(label))
        return bool(self.button_values.get(kwargs.get("key", ""), False))

    def number_input(
        self,
        label: str,
        value: float,
        min_value: float,
        max_value: float,
        **kwargs: Any,
    ) -> float:
        return float(self.number_input_values.get(kwargs.get("key", ""), value))

    def rerun(self) -> None:
        self._rerun_count += 1


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> _St:
    recorder = _St()
    # 关掉真 fragment 装饰器: 测试直接跑实质渲染逻辑。
    monkeypatch.setattr(gui, "_FRAGMENT_ENABLED", False)
    monkeypatch.setattr(gui, "st", recorder)
    return recorder


@pytest.fixture
def fake_camera() -> Any:
    """一台假相机: open/close/stop_stream/set_exposure_ms/get_numpy_image 可记录。"""

    class _Fake:
        opened = False
        closed = False
        stopped = False
        exposure = 5.0
        frames = [np.arange(100, dtype=np.float64).reshape(10, 10)]
        set_exposures: list[float] = []

        def open(self) -> None:
            self.opened = True

        def close(self) -> None:
            self.closed = True

        def stop_stream(self) -> None:
            self.stopped = True

        def get_exposure_time_ms(self) -> float:
            return self.exposure

        def set_exposure_ms(self, ms: float) -> float:
            self.set_exposures.append(float(ms))
            self.exposure = float(ms)
            return self.exposure

        def get_numpy_image(
            self, n_sample: int = 1, skip_first: bool = True
        ) -> np.ndarray:
            return self.frames[0].copy()

    return _Fake()


def _tree() -> ast.Module:
    return ast.parse(SOURCE.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 缓存键
# ---------------------------------------------------------------------------


def test_cache_key_encodes_type_and_id() -> None:
    assert gui.cache_key_for("daheng", 0) != gui.cache_key_for("miicam", 0)
    assert gui.cache_key_for("daheng", 0) != gui.cache_key_for("daheng", 1)
    assert gui.cache_key_for("daheng", 1) == gui.cache_key_for("daheng", 1)


# ---------------------------------------------------------------------------
# 句柄生命周期
# ---------------------------------------------------------------------------


def test_first_call_fabricates_and_opens_camera(
    stub: _St, monkeypatch: pytest.MonkeyPatch
) -> None:
    created: list[tuple] = []
    fake = type(
        "Cam",
        (),
        {
            "opened": False,
            "closed": False,
            "open": lambda self: setattr(self, "opened", True),
            "close": lambda self: setattr(self, "closed", True),
        },
    )()

    def _create(cam_type: str, cam_id: int, exposure_time_ms: float = 5.0) -> Any:
        created.append((cam_type, cam_id, exposure_time_ms))
        return fake

    monkeypatch.setattr(gui, "create_camera", _create)
    cam = gui.ensure_camera_handle(stub, "daheng", 0)
    assert cam is fake
    assert created == [("daheng", 0, gui.BENCH_DEFAULT_EXPOSURE_MS)]
    assert fake.opened is True


def test_handle_rebuilt_when_key_changes(
    stub: _St, monkeypatch: pytest.MonkeyPatch
) -> None:
    fakes = [
        type(
            "C",
            (),
            {
                "opened": False,
                "closed": False,
                "open": lambda self: setattr(self, "opened", True),
                "close": lambda self: setattr(self, "closed", True),
            },
        )()
        for _ in range(2)
    ]

    def _create(cam_type: str, cam_id: int, exposure_time_ms: float = 5.0) -> Any:
        assert cam_type in ("daheng", "miicam")
        return fakes[0 if cam_type == "daheng" else 1]

    monkeypatch.setattr(gui, "create_camera", _create)
    first = gui.ensure_camera_handle(stub, "daheng", 0)
    second = gui.ensure_camera_handle(stub, "daheng", 0)  # 同键, 复用
    assert first is second

    rebuilt = gui.ensure_camera_handle(stub, "miicam", 0)  # 换键, 重建
    assert rebuilt is not second
    assert rebuilt is fakes[1]
    assert fakes[0].closed is True


def test_close_button_clears_handle_and_closes(
    stub: _St, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = type(
        "C",
        (),
        {
            "opened": False,
            "closed": False,
            "stopped": False,
            "open": lambda self: setattr(self, "opened", True),
            "close": lambda self: setattr(self, "closed", True),
            "stop_stream": lambda self: setattr(self, "stopped", True),
        },
    )()

    def _create(cam_type: str, cam_id: int, exposure_time_ms: float = 5.0) -> Any:
        return fake

    monkeypatch.setattr(gui, "create_camera", _create)
    gui.ensure_camera_handle(stub, "daheng", 0)
    gui.release_camera_handle(stub)
    assert fake.closed is True
    assert fake.stopped is True
    assert gui.cached_handle(stub) is None


def test_open_failure_returns_none_and_clears_cache(
    stub: _St, monkeypatch: pytest.MonkeyPatch
) -> None:
    closed = {"v": False}

    class _Broken:
        def open(self) -> None:
            raise RuntimeError("device busy")

        def close(self) -> None:
            closed["v"] = True

    def _create(cam_type: str, cam_id: int, exposure_time_ms: float = 5.0) -> Any:
        return _Broken()

    monkeypatch.setattr(gui, "create_camera", _create)
    assert gui.ensure_camera_handle(stub, "daheng", 0) is None
    assert closed["v"] is True
    assert gui.cached_handle(stub) is None


# ---------------------------------------------------------------------------
# 无相机时的空态: 不崩 + 点名修补办法
# ---------------------------------------------------------------------------


def test_render_without_camera_shows_info_naming_action(
    stub: _St, monkeypatch: pytest.MonkeyPatch
) -> None:
    # bench_camera 返回默认 (daheng, 0), 但 create 失败 → running 下的空态 warning。
    def _create(cam_type: str, cam_id: int, exposure_time_ms: float = 5.0) -> Any:
        raise ImportError("no SDK")

    stub.session_state["ccd_live_running"] = True
    monkeypatch.setattr(gui, "create_camera", _create)
    monkeypatch.setattr(gui, "bench_camera", lambda: ("daheng", 0))  # type: ignore[assignment]

    gui.render_ccd_live()  # 不应抛
    assert stub.warnings, "空态必须以 warning 点名修补办法"
    assert "「CCD参数设置」" in stub.warnings[0]
    assert "开始实时采集" in stub.warnings[0]


def test_paused_state_shows_info_with_start_button(stub: _St) -> None:
    stub.session_state["ccd_live_running"] = False
    gui.render_ccd_live()
    assert stub.infos, "暂停空态必须有指引"
    assert "开始实时采集" in stub.infos[0]


# ---------------------------------------------------------------------------
# 帧归一化
# ---------------------------------------------------------------------------


def test_normalize_frame_maps_float_counts_to_uint8() -> None:
    raw = (500.0 + np.linspace(0.0, 255.0, 9 * 9)).reshape(9, 9)
    out = gui.normalize_frame(raw)
    assert out.dtype == np.uint8
    assert out.shape == raw.shape
    assert out.min() >= 0
    assert out.max() <= 255


def test_normalize_frame_3d_first_channel_and_empty() -> None:
    rgb = np.zeros((4, 5, 3), dtype=np.float64)
    rgb[..., 0] = 100.0
    out = gui.normalize_frame(rgb)
    assert out.shape == (4, 5)
    assert out.dtype == np.uint8
    empty = gui.normalize_frame(np.zeros((0, 0), dtype=np.float64))
    assert empty.dtype == np.uint8


# ---------------------------------------------------------------------------
# 曝光控制: 按钮路径调用 setter, 显示生效值
# ---------------------------------------------------------------------------


def test_apply_exposure_calls_fake_setter(stub: _St, fake_camera: Any) -> None:
    returned = gui.apply_exposure_ms(fake_camera, 7.5)
    assert fake_camera.set_exposures == [7.5]
    assert returned == 7.5
    assert fake_camera.exposure == 7.5


def test_run_exposure_display_uses_effective_readback(
    stub: _St, fake_camera: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub.session_state["ccd_live_running"] = True
    stub.session_state["ccd_live_cam"] = fake_camera
    stub.session_state["ccd_live_cache_key"] = gui.cache_key_for("daheng", 0)
    monkeypatch.setattr(gui, "bench_camera", lambda: ("daheng", 0))  # type: ignore[assignment]

    gui.render_ccd_live()
    # caption 里带生效曝光 (5.0) 与范围。
    joined = " ".join(stub.captions)
    assert "5" in joined
    assert "ms" in joined


# ---------------------------------------------------------------------------
# 渲染纪律 (AST, scope 本文件)
# ---------------------------------------------------------------------------


def test_no_divider_in_source() -> None:
    assert not any(
        isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "divider"
        for n in ast.walk(_tree())
    )


def test_no_primary_button_in_source() -> None:
    assert not any(
        isinstance(n, ast.Call)
        and getattr(n.func, "attr", None) == "button"
        and any(
            isinstance(k, ast.keyword)
            and k.arg == "type"
            and isinstance(k.value, ast.Constant)
            and k.value.value == "primary"
            for k in n.keywords
        )
        for n in ast.walk(_tree())
    )
