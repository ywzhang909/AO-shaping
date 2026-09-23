# -*- coding: utf-8 -*-
"""fouriergsnet_optimize.py 的离线 mock-SLM 测试 (不打开任何硬件).

该脚本是独立 CLI (不属于 ao_shaping 包), 因此用 importlib 从仓库根加载, 并
通过 FakeSLM/FakeCCD 驱动其显示助手与 ShapingSystem, 断言 AGENTS.md 中的设备契约:

* 每次写入都做内存槽轮换 (2..125) 且使用 MEMORY_MODE_INTERNAL;
* 平场 / 自定义 LUT 通道发送 **原始 uint16 灰度**, 绝不走弧度转换;
* ShapingSystem 使用 ``Santec.Panel_Res`` 解析面板尺寸
  (回归: 旧代码引用了不存在的 ``SantecSLM200``).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "fouriergsnet_optimize.py"
_SRC = _REPO_ROOT / "src"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ao_shaping.drivers.slm.santec import MEMORY_MODE_INTERNAL  # noqa: E402
from ao_shaping.utils.slm import phase_display  # noqa: E402


def _load_module():
    """从磁盘加载独立脚本 (不加入 sys.modules 缓存之外的位置)."""
    spec = importlib.util.spec_from_file_location("fouriergsnet_optimize", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fg = _load_module()


class FakeSLM:
    """Santec 最小替身: 记录每次显示调用, 不接触硬件."""

    Panel_Res = (1920, 1200)

    def __init__(self):
        self.displayed_memory_number: int | None = 7
        self.phase_calls: list[dict] = []
        self.data_calls: list[dict] = []

    def get_displayed_memory_number(self) -> int:
        return int(self.displayed_memory_number or 1)

    def display_phase(self, phase, wait_time_s=None, memory_number=None,
                      memory_mode=None) -> int:
        self.phase_calls.append(dict(phase=np.array(phase), wait_time_s=wait_time_s,
                                     memory_number=memory_number,
                                     memory_mode=memory_mode))
        self.displayed_memory_number = memory_number
        return 0

    def display_data(self, data, wait_time_s=None, memory_number=None,
                     memory_mode=None) -> int:
        self.data_calls.append(dict(data=np.array(data), wait_time_s=wait_time_s,
                                    memory_number=memory_number,
                                    memory_mode=memory_mode))
        self.displayed_memory_number = memory_number
        return 0


class FakeCCD:
    def get_numpy_image(self, n_sample: int = 1) -> np.ndarray:
        return np.zeros((64, 64), np.float64)


def _calib() -> dict:
    """ShapingSystem 构造 + display() 所需的最小标定字典."""
    return dict(
        Kx=80.0, Ky=80.0,
        center=np.array([60.0, 60.0]),
        crop_side=120,
        rotation_deg=0.0,
        beam_center=np.array([600.0, 960.0]),
    )


def _lut_result() -> dict:
    return dict(phase_range=2.0 * np.pi,
                gray_of_phase=np.linspace(0.0, 255.0, 256))


def _freeze(monkeypatch, slot: int) -> None:
    """固定槽位 + 消除 settle 睡眠, 让测试确定且快速."""
    monkeypatch.setattr(phase_display, "pick_slm_slot", lambda slm: slot)
    monkeypatch.setattr(fg.time, "sleep", lambda *_: None)


# --------------------------------------------------------------------------
# 显示助手契约
# --------------------------------------------------------------------------
def test_display_radians_rotates_slot_and_uses_memory_mode(monkeypatch):
    _freeze(monkeypatch, 42)
    slm = FakeSLM()
    phase = np.linspace(0.0, 2 * np.pi, 64, dtype=np.float32).reshape(8, 8)

    slot = fg._display_radians(slm, phase, 0.0)

    assert slot == 42
    assert len(slm.phase_calls) == 1
    assert slm.data_calls == []
    call = slm.phase_calls[0]
    assert call["memory_number"] == 42
    assert call["memory_mode"] == MEMORY_MODE_INTERNAL == 0
    assert call["wait_time_s"] == 0.0
    np.testing.assert_allclose(call["phase"], phase)


def test_display_grayscale_sends_raw_uint16(monkeypatch):
    _freeze(monkeypatch, 55)
    slm = FakeSLM()
    gray = np.arange(16, dtype=np.uint16).reshape(4, 4)

    slot = fg._display_grayscale(slm, gray, 0.0)

    assert slot == 55
    assert len(slm.data_calls) == 1
    assert slm.phase_calls == []
    call = slm.data_calls[0]
    assert call["data"].dtype == np.uint16
    np.testing.assert_array_equal(call["data"], gray)
    assert call["memory_number"] == 55
    assert call["memory_mode"] == MEMORY_MODE_INTERNAL
    assert call["wait_time_s"] == 0.0


def test_show_flat_uses_grayscale_path(monkeypatch):
    _freeze(monkeypatch, 60)
    slm = FakeSLM()
    cal = fg.SLMCCDCalibrator(slm, FakeCCD(), lambda ccd: np.zeros((64, 64)),
                              slm.Panel_Res, settle_s=0.0)

    cal._show_flat()

    assert len(slm.data_calls) == 1
    assert slm.phase_calls == []
    data = slm.data_calls[0]["data"]
    assert data.dtype == np.uint16
    assert data.shape == tuple(slm.Panel_Res)
    assert not data.any()


# --------------------------------------------------------------------------
# ShapingSystem 契约
# --------------------------------------------------------------------------
def test_shaping_system_display_uses_lut_grayscale(monkeypatch):
    _freeze(monkeypatch, 70)
    slm = FakeSLM()
    sys_ = fg.ShapingSystem(slm, FakeCCD(), lambda ccd: np.zeros((64, 64)),
                            _calib(), _lut_result())
    phi = fg.torch.full((fg.N, fg.N), np.pi, device=fg.DEV)

    sys_.display(phi)

    assert len(slm.data_calls) == 1
    assert slm.phase_calls == []
    data = slm.data_calls[0]["data"]
    assert data.dtype == np.uint16
    assert data.shape == tuple(slm.Panel_Res)
    assert 0 < int(data.max()) <= 255
    assert slm.data_calls[0]["memory_mode"] == MEMORY_MODE_INTERNAL


def test_shaping_system_uses_santec_panel_res():
    from ao_shaping.drivers.slm import Santec

    slm = FakeSLM()
    sys_ = fg.ShapingSystem(slm, FakeCCD(), lambda ccd: np.zeros((64, 64)),
                            _calib(), _lut_result())

    assert tuple(Santec.Panel_Res) == sys_.geo.panel_res
    assert tuple(Santec.Panel_Res) == sys_.lut.panel_res


def test_run_final_append_has_uniformity(monkeypatch):
    """回归: run() 收尾 append 必须含 uniformity/encircled (Recorder mark 断言).

    Recorder("uniformity") 的 append 断言 mark 键必须存在 (utils/io/file.py);
    旧代码只追加 compute_metrics 的 {mse, correlation, efficiency}, 缺少
    "uniformity" 触发 AssertionError. 修复后由 _append_final_record 补上
    uniformity/encircled 两个键.
    """
    _freeze(monkeypatch, 80)
    sys_ = fg.ShapingSystem(FakeSLM(), FakeCCD(), lambda ccd: np.zeros((64, 64)),
                            _calib(), _lut_result())
    recorder = fg.Recorder("uniformity", "max")
    acquire = lambda ccd: np.zeros((64, 64))
    final = sys_.geo.workzone(np.asarray(FakeCCD().get_numpy_image(1), np.float64),
                              fg.N)

    fg._append_final_record(sys_, recorder, acquire, FakeCCD())

    last = recorder.history[-1]
    assert "uniformity" in last
    assert "encircled" in last
    assert "mse" in last
    np.testing.assert_array_equal(last["ccd"], final)


# --------------------------------------------------------------------------
# 目标形状构建器 (square / circle / gaussian) 契约
# --------------------------------------------------------------------------
def _ring_mean(I, center: int, d: int) -> float:
    """I 上距 center 欧氏距离恰为 d 的像素均值 (网格索引空间)."""
    n = I.shape[0]
    idx = fg.torch.arange(n, device=I.device)
    dx = idx - center
    mask = (dx[:, None] ** 2 + dx[None, :] ** 2) == d * d
    return I[mask].mean().item()


def test_make_circle_target_unit_energy_and_mask():
    n, half = 64, 0.5
    I = fg.make_circle_target(n, half, fg.DEV)

    assert I.dtype == fg.torch.float32
    assert I.shape == (n, n)
    np.testing.assert_allclose(I.sum().item(), 1.0, rtol=1e-5)

    # top-hat: 值只取 {0, 1/圆内像素数}
    count = int((I > 0).sum().item())
    vals = fg.torch.unique(I)
    assert len(vals) == 2
    assert vals[0].item() == 0.0
    np.testing.assert_allclose(vals[1].item(), 1.0 / count, rtol=1e-6)

    # 圆面积 ≈ π(half·n/2)² (网格像素尺寸≈2/n; 实测 788 vs 804, 偏差<3%)
    expected = np.pi * half**2 * n**2 / 4
    assert abs(count - expected) / expected < 0.15


def test_make_gaussian_target_unit_energy_and_peak():
    n, sigma = 64, 0.3
    I = fg.make_gaussian_target(n, sigma, fg.DEV)

    assert I.dtype == fg.torch.float32
    assert I.shape == (n, n)
    np.testing.assert_allclose(I.sum().item(), 1.0, rtol=1e-5)

    # 峰值在网格中心 (n//2, n//2); 网格最接近原点的点即中心块
    assert I[n // 2, n // 2].item() == I.max().item()

    # 严格正 + 沿半径单调递减 (采样 3 个环)
    assert bool((I > 0).all())
    center = n // 2
    rings = [_ring_mean(I, center, d) for d in (8, 16, 24)]
    assert rings[0] > rings[1] > rings[2]


def test_shaping_system_target_fn_circle():
    slm = FakeSLM()
    sys_ = fg.ShapingSystem(slm, FakeCCD(), lambda ccd: np.zeros((64, 64)),
                            _calib(), _lut_result(), half=0.5,
                            target_fn=fg.make_circle_target)
    I = sys_.I_tgt

    # 圆形掩码: 值只取 {0, v>0}, 非方形足迹 (半径内像素数≈π(half·n/2)²)
    vals = fg.torch.unique(I)
    assert len(vals) == 2
    assert vals[0].item() == 0.0
    assert vals[1].item() > 0.0
    count = int((I > 0).sum().item())
    expected = np.pi * 0.5**2 * fg.N**2 / 4
    assert abs(count - expected) / expected < 0.15

    assert fg.torch.equal(sys_.roi, (I > 0).float())


def test_shaping_system_default_target_is_square():
    slm = FakeSLM()
    sys_ = fg.ShapingSystem(slm, FakeCCD(), lambda ccd: np.zeros((64, 64)),
                            _calib(), _lut_result())

    expected = fg.make_square_target(fg.N, fg.HALF, fg.DEV)
    assert fg.torch.equal(sys_.I_tgt, expected)


def test_shaping_system_target_fn_gaussian_roi():
    slm = FakeSLM()
    sys_ = fg.ShapingSystem(slm, FakeCCD(), lambda ccd: np.zeros((64, 64)),
                            _calib(), _lut_result(),
                            target_fn=fg.make_gaussian_target)
    I = sys_.I_tgt

    assert bool((I > 0).all())
    assert fg.torch.equal(sys_.roi, fg.torch.ones_like(I))
    assert fg.torch.equal(sys_.A_tgt, I.sqrt())
