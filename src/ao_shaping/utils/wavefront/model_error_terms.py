"""正向模型的误差项容器 (命令相位 → 实际相位 的 LUT 非线性 + 装调误差)。

数字孪生 (:mod:`ao_shaping.drivers.sim.bench_twin`) 把实验系统的非理想因素
分成三族: SLM 的 LUT 非线性、面板静态相差、装调 (离焦 / 倾斜)。model-in-the-loop
(MIL) 的正向模型 (:class:`~ao_shaping.algorithm.signal_processing.
zernike_coefficient_optimizer.ZernikeCoefficientOptimizer`) 只描述了**理想**
系统 —— 「命令相位 = 实际相位」。本模块把孪生有、MIL 没有的那几级误差搬进
MIL, 并给出一个可序列化的容器, 让标定产物能被装载、被复用。

误差项与自由度的对应 (实现时不要重复计入)
-------------------------------------------
| 误差项 | 作用位置 | 与谁简并 / 被谁吸收 |
|---|---|---|
| ``lut_gamma`` | **瞳面相位之前** (命令 → 实际相位) | 非冗余: MIL 唯一真正缺的误差级 |
| ``camera_defocus_m`` | **焦面复场**上, ``|·|²`` 之前 | 与 MIL 模型的 **Noll 4 (离焦) 系数**简并 —— 孪生实测二者逐位等价 (焦面复场的频谱就是瞳面, 所以焦面角谱 ≡ 瞳面二次相位) |
| ``tilt_x_rad`` / ``tilt_y_rad`` | 焦面图样平移 ``f · tilt`` | 在硬件上被**几何标定**吸收 (ROI 中心按 0 级实测); 孪生保留该级只是为了仿真路径完整 |

⇒ **不要既加「面板 Zernike」项又加「相机离焦」项去双重表达同一个自由度**:
Step A 拟合出来的 Noll 4 系数会把相机离焦吸收掉。

简并 / 吸收关系的实测出处见 ``src/ao_shaping/drivers/sim/bench_twin.py``
(``apply_defocus`` 的 docstring 与 ``apply_tilt`` 的 docstring)。

容器是 **frozen dataclass**, 且刻意带齐**评价误差项所需的几何常数**
(``wavelength_m`` / ``focal_length_m`` / ``far_field_pitch_m``): 离焦以米计、
倾斜以焦面平移量计, 它们只有在同一套光学处方下才有意义。把它们放进同一个
不可变对象, 就不可能把一台实验系统的离焦量配到另一台实验系统的焦面节距上。

本模块**不含 torch** (``utils`` 是零 torch 叶子包), 也不 import 任何驱动。
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

__all__ = [
    "ModelErrorTerms",
    "load_error_terms",
    "save_error_terms",
]

#: 满量程相位 (rad)。SLM 是 2π 周期器件, 故 2π = 满灰度。
DEFAULT_PHI_MAX_RAD = 2.0 * math.pi


@dataclass(frozen=True)
class ModelErrorTerms:
    """MIL 正向模型的误差项。

    Attributes:
        lut_gamma: **命令相位 → 实际相位** 的 Gamma 指数。``1.0`` = 理想线性
            (恒等映射)。语义与孪生 ``apply_lut`` 一致:
            ``actual = phi_max · (wrap(cmd) / 2π) ** gamma``。
        camera_defocus_m: 相机离焦量 (m, 正负皆可)。作用在**焦面复场**上的角谱
            传递函数; 倏逝分量置 0, 不产生 NaN。
        tilt_x_rad: 光路在 x 方向的倾斜 (rad)。焦面图样平移 ``f · tilt_x_rad``。
        tilt_y_rad: 光路在 y 方向的倾斜 (rad)。
        phi_max_rad: 满灰度对应的相位 (rad), 默认 2π。
        wavelength_m: 工作波长 (m)。评价离焦的角谱传递函数时需要。
        focal_length_m: 2f 透镜焦距 (m)。把倾斜角换算成焦面平移量时需要。
        far_field_pitch_m: 焦面采样间距 (m) = ``λ·f / (N_pad · d_pupil)``。
            离焦与倾斜都在焦面网格上评价, 所以需要它。

    默认值 = **理想系统**: ``lut_gamma = 1`` 且三项装调误差为 0。此时
    :attr:`is_identity` 为 ``True``, 正向模型与本模块引入前的输出**逐位相同**
    (γ = 1 时 LUT 是恒等映射, 连 2π 包裹都不引入 —— 包裹会改变最后几个比特,
    而既有输出必须一字节不差)。
    """

    lut_gamma: float = 1.0
    camera_defocus_m: float = 0.0
    tilt_x_rad: float = 0.0
    tilt_y_rad: float = 0.0
    phi_max_rad: float = DEFAULT_PHI_MAX_RAD
    wavelength_m: float = 0.0
    focal_length_m: float = 0.0
    far_field_pitch_m: float = 0.0

    def __post_init__(self) -> None:
        """校验标量参数的取值域。"""
        if not math.isfinite(self.lut_gamma) or self.lut_gamma <= 0.0:
            raise ValueError(
                f"lut_gamma must be finite and > 0, got {self.lut_gamma!r}"
            )
        if not math.isfinite(self.phi_max_rad) or self.phi_max_rad <= 0.0:
            raise ValueError(
                f"phi_max_rad must be finite and > 0, got {self.phi_max_rad!r}"
            )
        for name in ("camera_defocus_m", "tilt_x_rad", "tilt_y_rad"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")
        for name in ("wavelength_m", "focal_length_m", "far_field_pitch_m"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0, got {value!r}")

    # ------------------------------------------------------------------
    # 性质
    # ------------------------------------------------------------------

    @property
    def is_identity(self) -> bool:
        """``True`` 当且仅当三项误差全为零 (理想系统)。"""
        return (
            float(self.lut_gamma) == 1.0
            and float(self.camera_defocus_m) == 0.0
            and float(self.tilt_x_rad) == 0.0
            and float(self.tilt_y_rad) == 0.0
        )

    @property
    def needs_focal_geometry(self) -> bool:
        """``True`` 当离焦或倾斜非零 —— 它们只能在焦面网格上评价。"""
        return (
            float(self.camera_defocus_m) != 0.0
            or float(self.tilt_x_rad) != 0.0
            or float(self.tilt_y_rad) != 0.0
        )

    def validate_focal_geometry(self) -> None:
        """确认评价离焦 / 倾斜所需的几何常数已经就位。

        Raises:
            ValueError: 离焦缺波长或焦面节距, 或倾斜缺焦距或焦面节距时。
        """
        if float(self.camera_defocus_m) != 0.0:
            if float(self.wavelength_m) <= 0.0:
                raise ValueError(
                    "camera_defocus_m needs a wavelength: the defocus transfer "
                    "function is an angular-spectrum kernel, so lambda is not "
                    "optional"
                )
            if float(self.far_field_pitch_m) <= 0.0:
                raise ValueError(
                    "camera_defocus_m needs the far-field sample spacing "
                    "(far_field_pitch_m): the transfer function is evaluated on "
                    "the focal-plane frequency grid"
                )
        if float(self.tilt_x_rad) != 0.0 or float(self.tilt_y_rad) != 0.0:
            if float(self.focal_length_m) <= 0.0:
                raise ValueError(
                    "a tilt needs the focal length: the pattern shift is "
                    "f * tilt, so f is not optional"
                )
            if float(self.far_field_pitch_m) <= 0.0:
                raise ValueError(
                    "a tilt needs the far-field sample spacing "
                    "(far_field_pitch_m): the shift is expressed in metres"
                )

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, float]:
        """返回 JSON 可序列化的字典 (字段名即 dataclass 字段名)。"""
        return {key: float(value) for key, value in asdict(self).items()}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelErrorTerms":
        """从 :meth:`to_dict` 的输出 (或等价字典) 重建。

        未知键被忽略, 缺失的键取默认值 —— 这样一份只带误差标量的文件也能读。
        """
        known = {
            "lut_gamma",
            "camera_defocus_m",
            "tilt_x_rad",
            "tilt_y_rad",
            "phi_max_rad",
            "wavelength_m",
            "focal_length_m",
            "far_field_pitch_m",
        }
        kwargs = {
            key: float(value)
            for key, value in dict(payload).items()
            if key in known
        }
        return cls(**kwargs)

    def with_optics(
        self,
        *,
        wavelength_m: float,
        focal_length_m: float,
        far_field_pitch_m: float,
    ) -> "ModelErrorTerms":
        """返回换上另一套光学处方后的副本。

        离焦 (m) 与倾斜 (rad) 只有在对应的波长 / 焦距 / 焦面节距下才有意义。
        标定产物里的几何属于**标定它的那台实验系统**; 当 MIL 运行在另一套
        网格上时, 必须用本次运行的几何替换它 —— 这正是本方法的用途。

        Args:
            wavelength_m: 本次运行的工作波长 (m)。
            focal_length_m: 本次运行的 2f 透镜焦距 (m)。
            far_field_pitch_m: 本次运行的焦面采样间距 (m)。

        Returns:
            换上几何后的新实例 (标量误差项原样保留)。
        """
        return replace(
            self,
            wavelength_m=float(wavelength_m),
            focal_length_m=float(focal_length_m),
            far_field_pitch_m=float(far_field_pitch_m),
        )


def _twin_far_field_pitch_m(config: dict[str, Any]) -> float:
    """由孪生配置算出它的焦面采样间距 (m)。

    孪生的口径: ``p_fft = λ·f / (N_pad · d_pupil)``, 其中
    ``d_pupil = region_span_px / region · slm_pixel``。

    Args:
        config: 孪生配置字典 (``BenchTwinConfig.to_dict()`` 的输出)。

    Returns:
        焦面采样间距 (m); 配置缺字段时返回 0.0。
    """
    try:
        lam = float(config["wavelength_m"])
        focal = float(config["focal_length_m"])
        region = int(config["region"])
        span = float(config["region_span_px"])
        slm_um = float(config["slm_pixel_um"])
        padding = int(config["far_field_padding"])
    except (KeyError, TypeError, ValueError):
        return 0.0
    if region <= 0 or span <= 0 or slm_um <= 0 or lam <= 0 or focal <= 0:
        return 0.0
    pupil_pitch_m = span / region * slm_um * 1e-6
    n_pad = region * padding
    return lam * focal / (n_pad * pupil_pitch_m)


def load_error_terms(path: str | Path) -> ModelErrorTerms:
    """从标定产物装载误差项。

    接受两种 JSON:

    * **孪生反演结果** (``tools/slm/bench_twin_calibrate.py::save_fit`` 写的
      ``TwinFitResult.to_dict()``): 误差标量在 ``config.errors`` 里, 几何
      (``wavelength_m`` / ``focal_length_m`` / 焦面节距) 由 ``config`` 推出。
    * **本容器自己的字典** (:meth:`ModelErrorTerms.to_dict`): 直接读。

    孪生反演结果里的几何属于**标定它的那台实验系统**; MIL 运行在自己的网格
    上, 所以调用方应当用 :meth:`ModelErrorTerms.with_optics` 换上本次运行的
    几何。

    Args:
        path: JSON 文件路径。

    Returns:
        装载好的 :class:`ModelErrorTerms`。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: 文件内容不是可识别的误差项来源。
    """
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{source} does not contain an error-term payload")

    if "config" in payload and isinstance(payload["config"], dict):
        config = dict(payload["config"])
        raw_errors = dict(config.get("errors") or {})
    elif "lut_gamma" in payload or "errors" in payload:
        config = dict(payload)
        raw_errors = dict(payload.get("errors") or payload)
    else:
        raise ValueError(
            f"{source} does not contain error terms: expected a twin fit "
            "result (a 'config' key) or a ModelErrorTerms dictionary"
        )

    def pick(*names: str, default: float = 0.0) -> float:
        """The first present name among ``names`` (the twin's and ours)."""
        for name in names:
            if name in raw_errors:
                return float(raw_errors[name])
        return float(default)

    terms = ModelErrorTerms(
        lut_gamma=pick("lut_gamma", default=1.0),
        camera_defocus_m=pick("camera_defocus_m", "defocus_m"),
        tilt_x_rad=pick("tilt_x_rad"),
        tilt_y_rad=pick("tilt_y_rad"),
        phi_max_rad=pick(
            "phi_max_rad", "lut_phase_max_rad", default=DEFAULT_PHI_MAX_RAD
        ),
        wavelength_m=float(config.get("wavelength_m", 0.0) or 0.0),
        focal_length_m=float(config.get("focal_length_m", 0.0) or 0.0),
        far_field_pitch_m=pick(
            "far_field_pitch_m", default=_twin_far_field_pitch_m(config)
        ),
    )
    return terms


def save_error_terms(terms: ModelErrorTerms, path: str | Path) -> Path:
    """把误差项写成 JSON (父目录会自动创建)。

    Args:
        terms: 要落盘的误差项。
        path: 目标文件路径。

    Returns:
        写入的路径。
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(terms.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return target
