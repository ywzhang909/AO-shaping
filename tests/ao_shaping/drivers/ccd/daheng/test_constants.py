import pytest
from unittest.mock import MagicMock, patch

import numpy as np

from ao_shaping.drivers.ccd.daheng import constants


class TestGxAutoEntryValues:
    def test_off_value(self):
        assert constants.GxAutoEntry.OFF.value == 0

    def test_continuous_value(self):
        assert constants.GxAutoEntry.CONTINUOUS.value == 1

    def test_once_value(self):
        assert constants.GxAutoEntry.ONCE.value == 2

    def test_values_are_integers(self):
        for member in constants.GxAutoEntry:
            assert isinstance(member.value, int)

    def test_member_count(self):
        assert len(constants.GxAutoEntry) == 3


class TestGxPixelFormatEntry:
    def test_mono8(self):
        assert constants.GxPixelFormatEntry.MONO8.value == 0x01080001

    def test_all_values_integers(self):
        for m in constants.GxPixelFormatEntry:
            assert isinstance(m.value, int)


class TestGxTriggerActivationEntry:
    def test_falling_edge(self):
        assert constants.GxTriggerActivationEntry.FALLING_EDGE.value == 0

    def test_rising_edge(self):
        assert constants.GxTriggerActivationEntry.RISING_EDGE.value == 1


class TestGxFrameStatusList:
    def test_success(self):
        assert constants.GxFrameStatusList.SUCCESS.value == 0

    def test_incomplete(self):
        assert constants.GxFrameStatusList.INCOMPLETE.value == -1


class TestExposureAutoConstantValues:
    def test_off_is_zero(self):
        assert constants.GxAutoEntry.OFF.value == 0

    def test_continuous_is_one(self):
        assert constants.GxAutoEntry.CONTINUOUS.value == 1

    def test_once_is_two(self):
        assert constants.GxAutoEntry.ONCE.value == 2


class TestEnableAutoExposureUsesConstants:
    @patch("ao_shaping.drivers.ccd.daheng.driver.gx", create=True)
    def test_enable_uses_continuous(self, mock_gx):
        from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

        mock_cam = MagicMock()

        cam = DahengCamera(cam_id=0, exposure_time_ms=10.0)
        cam.cam = mock_cam

        result = cam.enable_auto_exposure(enable=True)

        assert result is True
        mock_cam.ExposureAuto.set.assert_called_once_with(
            constants.GxAutoEntry.CONTINUOUS.value
        )

    @patch("ao_shaping.drivers.ccd.daheng.driver.gx", create=True)
    def test_disable_uses_off(self, mock_gx):
        from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

        mock_cam = MagicMock()

        cam = DahengCamera(cam_id=0, exposure_time_ms=10.0)
        cam.cam = mock_cam

        result = cam.enable_auto_exposure(enable=False)

        assert result is True
        mock_cam.ExposureAuto.set.assert_called_once_with(
            constants.GxAutoEntry.OFF.value
        )


class TestAutoExposureUsesConstants:
    @patch("ao_shaping.drivers.ccd.daheng.driver.gx", create=True)
    def test_sdk_auto_exposure_uses_once_then_off(self, mock_gx):
        from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

        mock_cam = MagicMock()
        mock_cam.ExposureTime.get_range.return_value = {"min": 0.001, "max": 1000.0}
        mock_cam.WidthMax.get.return_value = 1920
        mock_cam.HeightMax.get.return_value = 1080
        mock_cam.data_stream = [MagicMock()]
        mock_gx.GxFrameStatusList.SUCCESS = 0

        cam = DahengCamera(cam_id=0, exposure_time_ms=10.0)
        cam.cam = mock_cam
        cam.cam_width = 1920
        cam.cam_height = 1080
        cam._DahengCamera__take_one_shot = MagicMock(
            return_value=np.zeros((1080, 1920), dtype=np.uint8)
        )
        cam.reset_exposure_time = MagicMock()

        result = cam.auto_exposure(use_sdk_auto=True, sdk_settle_frames=2)

        calls = mock_cam.ExposureAuto.set.call_args_list
        assert constants.GxAutoEntry.ONCE.value in [c.args[0] for c in calls]
        assert constants.GxAutoEntry.OFF.value in [c.args[0] for c in calls]

    @patch("ao_shaping.drivers.ccd.daheng.driver.gx", create=True)
    def test_initialize_disables_auto_exposure(self, mock_gx):
        from ao_shaping.drivers.ccd.daheng.driver import DahengCamera

        mock_cam = MagicMock()
        mock_cam.ExposureTime.get_range.return_value = {"min": 0.001, "max": 1000.0}
        mock_cam.WidthMax.get.return_value = 1920
        mock_cam.HeightMax.get.return_value = 1080
        mock_cam.data_stream = [MagicMock()]
        mock_raw = MagicMock()
        mock_raw.get_status.return_value = 0
        mock_raw.get_numpy_array.return_value = MagicMock()
        mock_cam.data_stream[0].get_image.return_value = mock_raw
        mock_gx.GxFrameStatusList.SUCCESS = 0
        mock_gx.GxPixelFormatEntry.MONO8 = 0x01080001

        mock_device_manager = MagicMock()
        mock_gx.DeviceManager.return_value = mock_device_manager
        mock_device_manager.update_device_list.return_value = (1, [{"sn": "TEST001"}])
        mock_device_manager.open_device_by_sn.return_value = mock_cam

        cam = DahengCamera(cam_id=0, exposure_time_ms=10.0)
        cam.open()

        mock_cam.ExposureAuto.set.assert_any_call(constants.GxAutoEntry.OFF.value)


class TestAllEnumValuesAreIntegers:
    _enums_to_check = [
        constants.GxDeviceClassList,
        constants.GxAccessStatus,
        constants.GxAccessMode,
        constants.GxPixelFormatEntry,
        constants.GxFrameStatusList,
        constants.GxDeviceTemperatureSelectorEntry,
        constants.GxPixelSizeEntry,
        constants.GxPixelColorFilterEntry,
        constants.GxAcquisitionModeEntry,
        constants.GxTriggerSourceEntry,
        constants.GxTriggerActivationEntry,
        constants.GxExposureModeEntry,
        constants.GxUserOutputSelectorEntry,
        constants.GxUserOutputModeEntry,
        constants.GxGainSelectorEntry,
        constants.GxBlackLevelSelectEntry,
        constants.GxBalanceRatioSelectorEntry,
        constants.GxAALightEnvironmentEntry,
        constants.GxUserSetEntry,
        constants.GxAWBLampHouseEntry,
        constants.GxUserDataFieldSelectorEntry,
        constants.GxTestPatternEntry,
        constants.GxTriggerSelectorEntry,
        constants.GxLineSelectorEntry,
        constants.GxLineModeEntry,
        constants.GxLineSourceEntry,
        constants.GxEventSelectorEntry,
        constants.GxLutSelectorEntry,
        constants.GxTransferControlModeEntry,
        constants.GxTransferOperationModeEntry,
        constants.GxTestPatternGeneratorSelectorEntry,
        constants.GxChunkSelectorEntry,
        constants.GxBinningHorizontalModeEntry,
        constants.GxBinningVerticalModeEntry,
        constants.GxSensorShutterModeEntry,
        constants.GxAcquisitionStatusSelectorEntry,
        constants.GxExposureTimeModeEntry,
        constants.GxGammaModeEntry,
        constants.GxLightSourcePresetEntry,
        constants.GxColorTransformationModeEntry,
        constants.GxColorTransformationValueSelectorEntry,
        constants.GxAutoEntry,
        constants.GxSwitchEntry,
        constants.GxRegionSendModeEntry,
        constants.GxRegionSelectorEntry,
        constants.GxTimerSelectorEntry,
        constants.GxTimerTriggerSourceEntry,
        constants.GxCounterSelectorEntry,
        constants.GxCounterEventSourceEntry,
        constants.GxCounterResetSourceEntry,
        constants.GxCounterResetActivationEntry,
        constants.GxCounterTriggerSourceEntry,
        constants.GxTimerTriggerActivationEntry,
        constants.GxStopAcquisitionModeEntry,
        constants.GxDSStreamBufferHandlingModeEntry,
        constants.GxResetDeviceModeEntry,
        constants.DxBayerConvertType,
        constants.DxValidBit,
        constants.DxImageMirrorMode,
        constants.DxRGBChannelOrder,
    ]

    @pytest.mark.parametrize(
        "enum_cls", _enums_to_check, ids=[e.__name__ for e in _enums_to_check]
    )
    def test_all_values_are_integers(self, enum_cls):
        for member in enum_cls:
            assert isinstance(member.value, int)
