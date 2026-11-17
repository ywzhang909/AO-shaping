"""Daheng camera auto-exposure operations."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from loguru import logger

from ao_shaping.drivers.ccd.base import BaseCamera, CameraError
from ao_shaping.drivers.ccd.daheng import constants, discovery


class AutoExposureMixin:
    def auto_exposure(
        self,
        target_max: float = BaseCamera.AUTO_EXPOSURE_TARGET_MAX,
        tolerance: float = BaseCamera.AUTO_EXPOSURE_TOLERANCE,
        twice_valid: bool = True,
        max_iterations: int = BaseCamera.AUTO_EXPOSURE_MAX_ITERATIONS,
        n_sample: int = BaseCamera.AUTO_EXPOSURE_N_SAMPLE,
        use_sdk_auto: bool = True,
        sdk_settle_frames: int = 3,
    ) -> npt.NDArray[np.uint8]:
        """自动曝光调整 - 优先大恒 SDK 原生自动曝光, 再比例迭代修正。

        ``target_max``/``tolerance`` 均为 **0-255 灰度** (与 SDK ``ExpectedGrayValue``
        一致)。阶段 1 (use_sdk_auto): ExposureAuto="Once"+ExpectedGrayValue=target →
        采 sdk_settle_frames 帧让 SDK 收敛 → 读回曝光作初值 → 切手动。阶段 2 比例迭代:
        拍图取峰值, 落 ``target±tolerance`` (twice_valid 需连续两次) 即停; 否则
        ``new_exp = exp*target/peak`` (单步放大上限 3×) 裁剪到 [min,max], 处理已到
        边界仍不达标, 重复至收敛或 max_iterations。
        """
        self._ensure_open()
        if tolerance <= 0:
            raise ValueError("tolerance must be > 0")

        target_val = float(target_max)
        low = int(max(target_val - tolerance, 10))
        high = int(min(target_val + tolerance, 254))
        min_exp, max_exp = self.get_exposure_range()

        # ── 阶段 1: SDK 原生自动曝光 ──
        if use_sdk_auto:
            sdk_auto_enabled = False
            try:
                sdk_target = int(max(0, min(255, target_val)))
                self.cam.ExposureAuto.set(constants.GxAutoEntry.ONCE.value)
                sdk_auto_enabled = True
                self.cam.ExpectedGrayValue.set(sdk_target)
                logger.info(
                    "[SDK auto-exposure] enabled, target={}, waiting {} frames "
                    "for convergence...",
                    sdk_target,
                    sdk_settle_frames,
                )
                # 采集若干帧让 SDK 内部自动曝光收敛
                for _ in range(sdk_settle_frames):
                    self._DahengCamera__take_one_shot()
                logger.info(
                    "[SDK auto-exposure] converged: exp={:.2f}ms",
                    self.get_exposure_time_ms(),
                )
            except Exception as e:  # 保留宽泛捕获: gxipy 第三方 Feature 访问, 不支持时抛出的类型不可枚举 (UnexpectedError/FeatureTypeError/ParameterTypeError 各自独立继承 Exception, 无公共基类)
                logger.warning(
                    "[SDK auto-exposure] not available, falling back to manual: {}",
                    e,
                )
            finally:
                # SDK auto-exposure must never stay enabled when falling through to the
                # manual proportional loop, including when settle raised mid-run.
                if sdk_auto_enabled:
                    try:
                        self.cam.ExposureAuto.set(constants.GxAutoEntry.OFF.value)
                    except Exception as disable_exc:  # 保留宽泛捕获: gxipy 第三方 Feature 访问, 不支持时抛出的类型不可枚举 (UnexpectedError/FeatureTypeError/ParameterTypeError 各自独立继承 Exception, 无公共基类)
                        logger.warning(
                            "[SDK auto-exposure] disable failed: {}", disable_exc
                        )

        logger.info(
            "Auto exposure start: target={:.0f}±{:.0f} (range=[{}, {}]ms, max_iter={})",
            target_val,
            tolerance,
            min_exp,
            max_exp,
            max_iterations,
        )

        # ── 阶段 2: 比例迭代修正 ──
        twice_ok = False
        img = self.get_numpy_image(n_sample, skip_first=True)

        for i in range(max(1, int(max_iterations))):
            # 0-255 灰度量纲 (与基类同一入口): 12/16-bit 帧满量程不是 255
            peak = self._auto_exposure_peak(img)

            if low <= peak <= high:
                if twice_ok or not twice_valid:
                    logger.info(
                        "Auto exposure converged at iter {}: exp={:.3f}ms, max={:.1f}",
                        i + 1,
                        self.get_exposure_time_ms(),
                        peak,
                    )
                    return img
                twice_ok = True
            else:
                twice_ok = False
                current = self.get_exposure_time_ms()
                if current <= 0:
                    break
                ratio = min(target_val / max(peak, 1.0), self.AUTO_EXPOSURE_MAX_RATIO)
                new_exp = float(np.clip(current * ratio, min_exp, max_exp))
                if abs(new_exp - current) < 1e-9:
                    break
                # 用**生效值**判边界: reset_exposure_time 会把请求钳到设备量程,
                # 拿请求值比较会在「设备拒绝该值」时一直转到 max_iterations。
                applied = float(self.reset_exposure_time(new_exp))
                if applied <= min_exp and peak > high:
                    logger.warning(
                        "target brightness {:.0f} unreachable (peak {:.1f}); "
                        "exposure forced to min",
                        target_val,
                        peak,
                    )
                    break
                if applied >= max_exp and peak < low:
                    logger.warning(
                        "target brightness {:.0f} unreachable (peak {:.1f}); "
                        "exposure forced to max",
                        target_val,
                        peak,
                    )
                    break

            img = self.get_numpy_image(n_sample, skip_first=True)

        logger.info(
            "Auto exposure finished: exp={:.3f}ms, max={:.1f}",
            self.get_exposure_time_ms(),
            self._auto_exposure_peak(img),
        )
        return img
    @staticmethod
    def get_cam_list() -> list[Any]:
        """获取可用相机列表（大恒返回 SDK 的设备信息对象）；枚举统一走 discovery 漏斗, 不 ping 固定 IP。"""
        return discovery.enumerate_devices().devices
    def _update_exposure_range(self) -> tuple[float, float]:
        """从设备回读曝光范围, 刷新本地 min/max 缓存并返回 (min_ms, max_ms)。

        范围原先只在 ``initialize()`` 里采一次; 但 ``ExposureTime.ms`` 的 setter 对
        越界值**抛 ValueError**, 而 ``reset_exposure_time`` 又把**设备回读值**写回缓存,
        ``auto_exposure`` 还拿这个范围当裁剪边界。缓存一旦与设备脱节 (initialize 时
        ``get_range()`` 返回空而沿用了 dataclass 默认的 0~100000 ms、SDK 侧改过量程、
        配置文件覆盖了曝光), 一次合法的改曝光就会在写缓存时炸掉, 或者把曝光裁到错误
        边界。故凡把范围当真值用的入口都先回读。

        读不到 (``get_range()`` 返回空 / 不是字典 / 缺键 / ``max <= min``) 时**沿用上次
        缓存**并记 warning, 不抛错: ``get_exposure_range`` 是查询入口, ``auto_exposure``
        每轮都调它, 一个读不出来的量程不该把整个自动曝光打死 —— 退回缓存的行为与
        修复前一致, 而设备**答得上来**时缓存就再也不会陈旧。
        """
        self._ensure_open()
        cached = (
            float(self._DahengCamera__exposure_time_ms.min),
            float(self._DahengCamera__exposure_time_ms.max),
        )
        float_range = self.cam.ExposureTime.get_range()
        parsed = self._parse_exposure_range(float_range)
        if parsed is None:
            logger.warning(
                f"Daheng ExposureTime.get_range() 不可用 ({float_range!r}); "
                f"沿用上次缓存的曝光范围 [{cached[0]}, {cached[1]}] ms"
            )
            return cached
        lo, hi = parsed
        self._DahengCamera__exposure_time_ms.min = lo
        self._DahengCamera__exposure_time_ms.max = hi
        return lo, hi

    @staticmethod
    def _parse_exposure_range(float_range: Any) -> tuple[float, float] | None:
        """把 SDK 的 ``ExposureTime.get_range()`` (µs) 解析成 ``(min_ms, max_ms)``。

        不可用 (空 / 不是字典 / 缺键 / 值不是数 / ``max <= min``) 一律返回 ``None``
        = 「读不到」, 由调用方退回缓存。**这里不抛**: 量程读不到是降级而非致命错误。
        ``isinstance(..., dict)`` 这道闸是必要的 —— 裸 ``MagicMock`` 的 ``float()``
        会安静地给出 1.0, 让"读不到"伪装成"读到一个 1 µs 的荒谬量程"。
        """
        if not isinstance(float_range, dict):
            return None
        try:
            lo = float(float_range["min"]) / 1000.0  # SDK 单位 µs → 全驱动统一 ms
            hi = float(float_range["max"]) / 1000.0
        except (KeyError, TypeError, ValueError):
            return None
        if not hi > lo >= 0:
            return None
        return lo, hi
    @staticmethod
    def _int_from_readback(raw: Any) -> int | None:
        """把 GenICam 特征回读值归一化成 int; 认不出来返回 ``None`` (= 校验失败)。

        gxipy 的 ``get()`` 形态不唯一: 纯 int/float、``(value, name)`` 元组、枚举成员
        都出现过 (``initialize`` 里 ``StreamBufferHandlingMode.get()[0]``、
        ``_bit_depth.bit_depth_of_format`` 处理 ``PixelFormat`` 时同源)。
        ``None`` 一律当**失败**而不是「跳过校验」: 一个认不出的回读值 (包括测试里的
        裸 ``MagicMock``) 绝不能被当成「设备已确认」, 否则 M1 这道校验形同虚设。
        """
        if isinstance(raw, (bool, int, float)):
            return int(raw)
        if isinstance(raw, tuple) and raw:
            return AutoExposureMixin._int_from_readback(raw[0])
        value = getattr(raw, "value", None)
        if isinstance(value, (bool, int, float)):
            return int(value)
        return None

    def _verify_int_feature(self, name: str, expected: int, raw: Any) -> None:
        """校验一个整数特征的回读值; 不符 (含认不出) 抛 :class:`CameraError`。

        与 :meth:`set_bit_depth` 的 ``PixelFormat`` 回读、:meth:`reset_window` 的窗口
        回读同一口径: **设备真值优先**, 请求值不算数。
        """
        got = self._int_from_readback(raw)
        if got != int(expected):
            raise CameraError(
                f"Daheng {name} readback mismatch: requested {int(expected)}, "
                f"device reports {raw!r} (parsed={got})"
            )
    def enable_auto_exposure(self, enable: bool = True, mode: int = 1) -> bool:
        """启用或关闭自动曝光 (大恒 SDK ``ExposureAuto``), **写后回读校验**。

        ``mode`` 仅 enable=True 时有意义 (大恒 SDK 无单独模式枚举, 始终用连续模式)。

        返回:
            bool: 成功 True; 设备根本没有该特征 (不支持) False。

        抛出:
            CameraError: 特征存在但写入未生效 (回读与请求不符)。「不支持」与「写了没生效」
                必须分开: 前者调用方可以退回比例迭代, 后者必须让 ``auto_exposure`` 的
                阶段 1 失败 —— 否则它会拿**手动曝光**的帧当「SDK 已收敛」的结果,
                把曝光一路推到边界 (与 12-bit 满量程误判同型的静默错误)。
        """
        self._ensure_open()
        entry = (
            constants.GxAutoEntry.CONTINUOUS if enable else constants.GxAutoEntry.OFF
        )
        try:
            self.cam.ExposureAuto.set(entry.value)
            raw = self.cam.ExposureAuto.get()
        except Exception as e:  # gxipy Feature 访问, 不支持时抛出的类型不可枚举
            logger.warning(f"Auto exposure not supported on Daheng camera: {e}")
            return False
        self._verify_int_feature("ExposureAuto", entry.value, raw)
        return True
    def get_auto_exposure_state(self) -> dict:
        """获取当前自动曝光状态 (SDK ``ExposureAuto`` + ``ExpectedGrayValue``)。

        返回: enabled / mode (1=连续 0=禁用) / target / exposure_time_ms (ms) 字典。
        """
        self._ensure_open()
        state: dict = {
            "enabled": False,
            "mode": 0,
            "target": 120,
            "exposure_time_ms": 0.0,
        }
        try:
            _val, auto_key = self.cam.ExposureAuto.get()
            state["enabled"] = auto_key == "On"
            state["mode"] = 1 if auto_key == "On" else 0
            try:
                state["target"] = self.cam.ExpectedGrayValue.get()
            # policy: the auto-exposure target is an optional register; a state snapshot records only what this firmware exposes
            except Exception:  # 同上: gxipy Feature 访问, 不支持时抛出的类型不可枚举
                pass
            state["exposure_time_ms"] = self.exposure_time
        except Exception as e:  # 同上: gxipy Feature 访问, 不支持时抛出的类型不可枚举
            logger.warning(f"Auto exposure state not supported: {e}")
        return state
    def set_auto_exposure_range(
        self,
        max_time_ms: int = 350,
        min_time_ms: int = 0,
        max_gain: int = 300,
        min_gain: int = 100,
    ) -> bool:
        """设置自动曝光的时间与增益范围, 返回成功 True / 不支持 False。

        对应大恒 SDK ``AutoExposureTimeMin/Max`` (FloatFeature, µs) 与
        ``AutoGainMin/Max``, 限定自动曝光/增益的搜索空间。
        """
        self._ensure_open()
        try:
            self.cam.AutoExposureTimeMin.set(min_time_ms * 1000)
            self.cam.AutoExposureTimeMax.set(max_time_ms * 1000)
            self.cam.AutoGainMin.set(min_gain)
            self.cam.AutoGainMax.set(max_gain)
            logger.info(
                f"Auto exposure range set: time=[{min_time_ms}, {max_time_ms}]ms, "
                f"gain=[{min_gain}, {max_gain}]"
            )
            return True
        except Exception as e:  # 同上: gxipy Feature 访问, 不支持时抛出的类型不可枚举
            logger.warning(f"Auto exposure range not supported: {e}")
            return False
    @property
    def auto_exposure_value(self) -> float:
        """当前自动曝光的数值 (曝光时间, ms); 启用时返回 SDK 调到的值, 关闭时返回手动值。"""
        return self.exposure_time
