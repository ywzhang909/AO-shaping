"""离线测试: Daheng reset_window 的 GenICam [min, max, inc] 网格钳制。

背景坑 (2026-09-23): gxipy 的 ``IntFeature.set`` 对越界值**不抛异常**,
只 print "IntFeature.set: int_value out of bounds, xxx.range=[min,max,inc]"
然后**静默保留旧窗口** — 调用方拿到的是上一次的窗口而无从知晓
(开窗设置/关闭时的缓冲报错, 窗口被钳到 930×632)。修复后 reset_window
先把请求值钳制到相机合法网格再 set, 并 readback 校验返回真实生效值。
"""

from unittest.mock import MagicMock, patch

W_RANGE = {"min": 4, "max": 1680, "inc": 4}
H_RANGE = {"min": 2, "max": 1138, "inc": 2}
OX_RANGE = {"min": 0, "max": 800, "inc": 4}
OY_RANGE = {"min": 0, "max": 400, "inc": 2}


class TestClampInt:
    @patch("ao_shaping.drivers.ccd.daheng.driver.gx", create=True)
    def test_snaps_down_to_inc_grid(self, _mock_gx):
        from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

        assert DahengCamera._clamp_int(931, H_RANGE) == 930  # 931 → 偶数网格
        assert DahengCamera._clamp_int(635, W_RANGE) == 632  # 635 → 4 的倍数网格
        assert DahengCamera._clamp_int(632, W_RANGE) == 632  # 已在网格上
        assert DahengCamera._clamp_int(931, {"min": 2, "max": 1138, "inc": 2}) == 930

    @patch("ao_shaping.drivers.ccd.daheng.driver.gx", create=True)
    def test_clamps_below_min(self, _mock_gx):
        from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

        assert DahengCamera._clamp_int(1, H_RANGE) == 2
        assert DahengCamera._clamp_int(0, H_RANGE) == 2
        assert DahengCamera._clamp_int(-540, {"min": 0, "max": 800, "inc": 4}) == 0

    @patch("ao_shaping.drivers.ccd.daheng.driver.gx", create=True)
    def test_clamps_above_max(self, _mock_gx):
        from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

        assert DahengCamera._clamp_int(2000, W_RANGE) == 1680
        assert DahengCamera._clamp_int(9999, H_RANGE) == 1138

    @patch("ao_shaping.drivers.ccd.daheng.driver.gx", create=True)
    def test_none_range_passthrough(self, _mock_gx):
        from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

        assert DahengCamera._clamp_int(123, None) == 123


def _make_mock_cam(readback_size=(632, 930)):
    """构造带 GenICam range/readback 的 MagicMock 相机。

    readback_size: SDK 在 set 之后通过 Width.get()/Height.get() 读回的值
        (等于 set 值时表示 set 生效; 不同于 set 值时表示 SDK 静默保留旧窗口)。
    """
    cam = MagicMock()
    cam.stream_off = MagicMock()
    cam.stream_on = MagicMock()
    cam.Width.get_range.return_value = W_RANGE
    cam.Height.get_range.return_value = H_RANGE
    cam.OffsetX.get_range.return_value = OX_RANGE
    cam.OffsetY.get_range.return_value = OY_RANGE
    cam.Width.get.return_value = readback_size[0]
    cam.Height.get.return_value = readback_size[1]
    cam.OffsetX.get.return_value = None  # None → 回退到请求值
    cam.OffsetY.get.return_value = None
    return cam


class TestResetWindowClamp:
    def _make_cam(self, monkeypatch, readback_size=(632, 930)):
        """构造带日志捕获的 DahengCamera 实例, 返回 (cam, mock_cam, warnings)。"""
        from ao_shaping.drivers.ccd.daheng import driver as daheng_driver

        cam = daheng_driver.DahengCamera(cam_id=0, exposure_time_ms=10.0)
        mock_cam = _make_mock_cam(readback_size=readback_size)
        cam.cam = mock_cam
        warnings = []
        monkeypatch.setattr(
            daheng_driver.logger, "warning", lambda *a, **k: warnings.append(a)
        )
        return cam, mock_cam, warnings

    def test_no_clamp_when_in_range(self, monkeypatch):
        # 请求 (632, 930) 已在 [min,max,inc] 网格上, 中心落在可偏移范围
        cam, mock_cam, warnings = self._make_cam(monkeypatch)
        result = cam.reset_window(center=(632, 673), size=(632, 930))

        mock_cam.Width.set.assert_called_once_with(632)
        mock_cam.Height.set.assert_called_once_with(930)
        mock_cam.OffsetX.set.assert_called_once_with(316)  # 632 - 316
        mock_cam.OffsetY.set.assert_called_once_with(208)  # 673 - 465
        assert warnings == [], f"不应有 warning, got {warnings}"
        assert result == ((632, 930), (632, 673))

    def test_size_clamped_to_grid_with_warning(self, monkeypatch):
        # 请求 (631, 931) 违反 Height inc=2 / Width inc=4 -> 钳到 (628, 930)
        cam, mock_cam, warnings = self._make_cam(
            monkeypatch, readback_size=(628, 930)
        )
        result = cam.reset_window(center=(632, 673), size=(631, 931))

        mock_cam.Width.set.assert_called_once_with(628)
        mock_cam.Height.set.assert_called_once_with(930)
        assert "clamped" in str(warnings), f"缺失 size 钳制 warning, got {warnings}"
        # 读回 = 生效值; 返回中心 = offset + 生效宽高一半
        assert result == ((628, 930), (316 + 314, 208 + 465))
        assert result == ((628, 930), (630, 673))

    def test_offset_clamped_to_grid_with_warning(self, monkeypatch):
        # size 取最大值且中心在角落 -> 负偏移被钳到 0 (旧代码这里会 assert 崩溃)
        cam, mock_cam, warnings = self._make_cam(
            monkeypatch, readback_size=(1680, 1138)
        )
        result = cam.reset_window(center=(300, 300), size=(1680, 1138))

        mock_cam.Width.set.assert_called_once_with(1680)
        mock_cam.Height.set.assert_called_once_with(1138)
        mock_cam.OffsetX.set.assert_called_once_with(0)
        mock_cam.OffsetY.set.assert_called_once_with(0)
        assert "clamped" in str(warnings), f"缺失 offset 钳制 warning, got {warnings}"
        assert result == ((1680, 1138), (840, 569))

    def test_offset_clamp_below_min_does_not_crash(self, monkeypatch):
        # 负偏移钳制到 0, 不崩溃
        cam, mock_cam, _warnings = self._make_cam(
            monkeypatch, readback_size=(1680, 1138)
        )
        result = cam.reset_window(center=(50, 50), size=(1680, 1138))
        assert mock_cam.OffsetX.set.call_args[0][0] == 0
        assert result == ((1680, 1138), (840, 569))

    def test_warns_and_uses_readback_when_sdk_kept_old_window(self, monkeypatch):
        # SDK 静默保留旧窗口 (Width.get != set 值) -> warning + 返回读回值
        cam, _mock_cam, warnings = self._make_cam(
            monkeypatch, readback_size=(500, 480)
        )
        result = cam.reset_window(center=(632, 673), size=(632, 930))

        assert "kept the previous window" in str(warnings), f"{warnings}"
        # 返回真实生效窗口; 中心 = 请求 offset + 真实宽高一半
        assert result == ((500, 480), (316 + 250, 208 + 240))
        assert result == ((500, 480), (566, 448))

    def test_full_frame_branch(self, monkeypatch):
        # size=(0,0) 全帧分支: WidthMax/HeightMax, 偏移 0, 不触发任何钳制
        cam, mock_cam, warnings = self._make_cam(
            monkeypatch, readback_size=(1680, 1138)
        )
        mock_cam.WidthMax.get.return_value = 1680
        mock_cam.HeightMax.get.return_value = 1138
        result = cam.reset_window(center=(840, 569), size=(0, 0))

        mock_cam.OffsetX.set.assert_called_once_with(0)
        mock_cam.OffsetY.set.assert_called_once_with(0)
        assert warnings == []
        assert result == ((1680, 1138), (840, 569))