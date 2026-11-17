"""MIICAM output bit-depth and pixel-format setup."""

from __future__ import annotations

from loguru import logger


def _miicam_sdk():
    """Resolve the driver module's current SDK binding, including test overrides."""
    from ao_shaping.drivers.ccd.miicam import driver

    return driver.miicam


class BitDepthMixin:
    def _init_bit_depth(self) -> None:
        """查询最大位深并配置输出位深。

        契约属性 ``bit_depth`` 按**设备回读**写在这里: 位深是满量程的唯一来源
        (issue #246), 调用方读 ``cam.bit_depth`` 定满量程; 不写契约属性的话,
        满量程只能退回 dtype 兜底, 高位置深模式下 ``uint16`` 容器一律按
        65535 折算, 与真实位深不符。
        """
        self._max_bit_depth = self.cam.MaxBitDepth()

        if self._bit_depth == 8:
            output_bitdepth = 0
            self.bit_depth = 8
        elif self._max_bit_depth > 8:
            output_bitdepth = 1
        else:
            output_bitdepth = 0
            logger.warning(
                f"Camera max bit depth is {self._max_bit_depth}, "
                f"requested {self._bit_depth}-bit. Falling back to 8-bit."
            )
            self._bit_depth = 8
            self.bit_depth = 8

        try:
            self.cam.put_Option(_miicam_sdk().MIICAM_OPTION_BITDEPTH, output_bitdepth)
        except _miicam_sdk().HRESULTException:
            logger.warning(f"Could not set bit depth to {self._bit_depth}, using 8-bit")
            self._bit_depth = 8
            self.bit_depth = 8
        else:
            if output_bitdepth == 1:
                #: 高位深输出已启用: 生效位深取 ``MaxBitDepth()`` 的**回读真值**;
                #: 回读值不在 9..16 则退回请求值, 不留虚报档位。
                reported = int(self._max_bit_depth)
                self.bit_depth = reported if 9 <= reported <= 16 else int(self._bit_depth)
            else:
                self.bit_depth = 8
    def _init_pixel_format(self) -> None:
        """按位深设置像素格式。"""
        if self._bit_depth == 8:
            self._pixel_format = "MONO8"
            self._set_mono8_format()
            return

        self._pixel_format = "MONO16"
        if not self._try_set_option(_miicam_sdk().MIICAM_OPTION_RAW, 1):
            if not self._try_set_option(_miicam_sdk().MIICAM_OPTION_RGB, 4):
                logger.warning(
                    "Could not set high bit depth mode (RAW or 16-bit Grey), "
                    "falling back to 8-bit"
                )
                self._pixel_format = "MONO8"
                self._bit_depth = 8
                # 契约属性同步回落到 8: 像素格式退回 8-bit, 满量程随之是 255。
                self.bit_depth = 8
                self._set_mono8_format()

    def _set_mono8_format(self) -> None:
        """强制 MONO8 格式并关闭 RAW 模式。"""
        self._try_set_option(_miicam_sdk().MIICAM_OPTION_RGB, 3)
        self._try_set_option(_miicam_sdk().MIICAM_OPTION_RAW, 0)

    def _try_set_option(self, option: int, value: int) -> bool:
        """尝试设置某个 SDK 选项, 成功返回 True。"""
        try:
            self.cam.put_Option(option, value)
            return True
        except _miicam_sdk().HRESULTException:
            return False
    def _detect_raw_format(self) -> None:
        """检测实际 raw 格式, 必要时调整像素格式。"""
        raw_fmt, _ = self.cam.get_RawFormat()
        raw_fmt_str = (
            raw_fmt.decode("ascii", errors="replace")
            if isinstance(raw_fmt, bytes)
            else raw_fmt
        )

        if raw_fmt_str in ("YUYV", "VUYY", "UYVY"):
            self._pixel_format = "YUV422"
            logger.info(f"Camera using YUV422 format (raw: {raw_fmt_str})")
        elif raw_fmt_str in ("YMono",):
            logger.info(
                f"Camera using MONO{self._bit_depth} format (raw: {raw_fmt_str})"
            )
        elif raw_fmt_str in ("BGGR", "RGGB", "GRBG", "GBRG"):
            logger.info(
                f"Camera using Bayer {raw_fmt_str} format, treating as MONO{self._bit_depth}"
            )
        else:
            logger.info(
                f"Camera using format: {raw_fmt_str}, treating as MONO{self._bit_depth}"
            )
