"""中心口径偏好在 GUI 写入端与整形读取端之间的交接, 以及轴序约定。"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pytest

import ao_shaping
from ao_shaping.optimizer.wfless._measure import locate_zero_order
from ao_shaping.utils.image import center_preference
from ao_shaping.utils.image.zero_order import zero_order_center_by_method


@pytest.fixture(autouse=True)
def _reset_preferences():
    """模块级状态不跨用例泄漏。"""
    center_preference.reset()
    yield
    center_preference.reset()


def _analyzer_keys() -> list[str]:
    """从 ``analyzer.py`` 读出 ``CENTER_METHODS`` 的键 (不 import, 免拉 Qt)。"""
    root = Path(ao_shaping.__file__).resolve().parent
    path = root / "gui" / "ccd" / "beam_quality" / "analyzer.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        target = node.target if isinstance(node, ast.AnnAssign) else None
        if isinstance(target, ast.Name) and target.id == "CENTER_METHODS":
            row = node.value
            assert isinstance(row, ast.Tuple)
            return [pair.elts[0].value for pair in row.elts if isinstance(pair, ast.Tuple)]
    raise AssertionError("CENTER_METHODS not found in analyzer.py")


def _asymmetric_frame() -> np.ndarray:
    """主瓣在 ``(row=20, col=30)``, 旁瓣把质心拉向 ``(60, 80)``。"""
    yy, xx = np.mgrid[0:80, 0:100]
    main = np.exp(-(((yy - 20.0) ** 2 + (xx - 30.0) ** 2) / (2.0 * 2.0**2)))
    tail = 0.35 * np.exp(-(((yy - 60.0) ** 2 + (xx - 80.0) ** 2) / (2.0 * 6.0**2)))
    return 100.0 * (main + tail)


def test_default_center_method_is_ellipse() -> None:
    assert center_preference.get_center_method() == "ellipse"
    assert center_preference.get_center_method() == center_preference.DEFAULT_CENTER_METHOD


def test_set_center_method_round_trips() -> None:
    center_preference.set_center_method("centroid")
    assert center_preference.get_center_method() == "centroid"
    center_preference.set_center_method("peak")
    assert center_preference.get_center_method() == "peak"


def test_resolve_zero_order_center_uses_selected_method() -> None:
    frame = _asymmetric_frame()
    center_preference.set_center_method("centroid")

    assert center_preference.resolve_zero_order_center(frame) == (
        zero_order_center_by_method(frame, "centroid")
    )


def test_peak_analyzer_key_maps_to_argmax(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def _fake(frame: np.ndarray, method: str = "ellipse", **_kw: object):
        seen.append(method)
        return (7, 9)

    monkeypatch.setattr(center_preference, "zero_order_center_by_method", _fake)
    center_preference.set_center_method("peak")

    assert center_preference.resolve_zero_order_center(_asymmetric_frame()) == (7, 9)
    assert seen == ["argmax"], "界面键 peak 必须映射为 argmax"


def test_invalid_center_method_raises() -> None:
    with pytest.raises(ValueError, match="unknown center method"):
        center_preference.set_center_method("argmax")  # 界面侧没有这个键


def test_opened_camera_lifecycle() -> None:
    assert center_preference.get_opened_camera() is None

    center_preference.set_opened_camera("daheng", 1)
    assert center_preference.get_opened_camera() == ("daheng", 1)

    center_preference.reset()
    assert center_preference.get_opened_camera() is None
    assert center_preference.get_center_method() == "ellipse"


def test_resolve_zero_order_center_is_x_first() -> None:
    """argmax 口径下返回 ``(x, y)``: 帧里峰值在 (row=20, col=30) ⇒ (30, 20)。"""
    frame = _asymmetric_frame()
    center_preference.set_center_method("peak")

    assert zero_order_center_by_method(frame, "argmax") == (30, 20)
    assert center_preference.resolve_zero_order_center(frame) == (30, 20)
    assert locate_zero_order(frame) == (20, 30)


def test_locate_zero_order_preferred_is_row_first_and_honors_method() -> None:
    from ao_shaping.optimizer.wfless._measure import locate_zero_order_preferred

    frame = _asymmetric_frame()
    center_preference.set_center_method("peak")
    argmax_center = locate_zero_order_preferred(frame)
    assert argmax_center == locate_zero_order(frame), "默认峰值口径应与历史 argmax 一致"

    center_preference.set_center_method("centroid")
    centroid_center = locate_zero_order_preferred(frame)
    x, y = center_preference.resolve_zero_order_center(frame)
    assert centroid_center == (y, x), "行优先: (cy, cx) 是 (x, y) 的换位"
    assert centroid_center != argmax_center, "非默认口径必须真的改变定位结果"


def test_analyzer_contract_does_not_drift() -> None:
    """界面下拉框的键集/默认值必须与偏好库、zero_order 口径一致。"""
    from ao_shaping.utils.image.zero_order import DEFAULT_ZERO_ORDER_METHOD

    assert center_preference.DEFAULT_CENTER_METHOD == DEFAULT_ZERO_ORDER_METHOD
    assert set(_analyzer_keys()) == {"centroid", "box_centroid", "ellipse", "peak"}


def test_shaping_call_sites_use_the_preference_store() -> None:
    """读取端已接线: 旧 ``zero_order_center`` / ``locate_zero_order`` 调用不再残留。"""
    root = Path(ao_shaping.__file__).resolve().parent
    wfless = root / "optimizer" / "wfless"
    legacy_center = re.compile(r"(?<!resolve_)zero_order_center\(")

    mil = (wfless / "model_in_loop_shaping.py").read_text(encoding="utf-8")
    assert not legacy_center.search(mil)
    assert mil.count("resolve_zero_order_center(") >= 2

    smil = (wfless / "slm_model_in_loop.py").read_text(encoding="utf-8")
    assert not legacy_center.search(smil)
    assert smil.count("resolve_zero_order_center(") >= 2

    gsr = (wfless / "slm_gs_refine.py").read_text(encoding="utf-8")
    assert gsr.count("locate_zero_order_preferred(") == 3
    assert "locate_zero_order(frame0)" not in gsr

    measure = (wfless / "_measure.py").read_text(encoding="utf-8")
    assert "def locate_zero_order_preferred" in measure


def _transfer_page_functions() -> dict[str, str]:
    """``page.py`` 里各 ``TransferPage`` 方法反编译成源码文本 (不 import: 顶层拉 Qt)。"""
    root = Path(ao_shaping.__file__).resolve().parent
    path = root / "gui" / "slm" / "transfer_desktop" / "page.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name: ast.unparse(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }


def test_transfer_page_seeds_camera_from_opened_source() -> None:
    """① 的初值消费 ``get_opened_camera()``; 且它是初值, 不是锁。"""
    path = (
        Path(ao_shaping.__file__).resolve().parent
        / "gui"
        / "slm"
        / "transfer_desktop"
        / "page.py"
    )
    text = path.read_text(encoding="utf-8")
    assert (
        "from ao_shaping.utils.image.center_preference import get_opened_camera"
        in text
    )

    functions = _transfer_page_functions()
    seed = functions["_seed_camera_from_beam_quality"]
    assert "get_opened_camera()" in seed
    assert "CAMERA_CHOICES" in seed, "选项外的键必须忽略而不是崩溃"
    assert "_apply_ccd_choice" in seed, "控件要经既有的无信号写入口落值"

    init = functions["__init__"]
    assert init.index("self._build()") < init.index("self._seed_camera_from_beam_quality()")

    detected = functions["_on_ccd_detected"]
    assert "_camera_seeded_from_beam_quality" in detected, (
        "自动探测不得覆盖 Beam Quality 里真打开过的相机"
    )
