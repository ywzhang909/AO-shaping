"""自标定量: 瞳面仿射 τ_θ 与行程缩放 γ。

依据 GRAPHYCS (Cho 等 2026, "Graph-based modeling of optical system enables
adaptive optics on dynamic samples with self-calibration") 的思路: 前向模型是
**经典 Fourier 光学物理模型** (可微计算图), 可学习量除物体与像差系数外,
还有两类**自标定量** —— 它们嵌在正向模型**内部**, 不直接进损失:

1. **行程缩放 γ** (逐模式族): 作用在**指令**相位上, 不作用在待估像差上。
   论文写作 ``scale = 1 + γ``, 并逐步 clamp ``γ ≥ −1`` (即缩放因子非负)。
2. **瞳面平面内仿射 τ_θ** (6 个标量 ``[amp_x, amp_y, off_x, off_y, rot,
   grid]``): ``off`` 平移只作用于**振幅包络** (归一化坐标), ``amp`` 各向异性
   缩放只作用于振幅, ``rot`` 只旋转**指令相位**, ``grid`` 等向缩放同时作用于
   总相位网格与振幅包络。初始化为单位变换。

与论文的对应关系 (务必先读)
---------------------------
论文的**指令**是一次 Zernike 级数 ``Σ a_k Z_k``, 所以「逐模式族缩放」是
直接作用在系数 ``a_k`` 上的。本仓库的指令是一次**探针相位** (随机高斯瞳面
相位, 硬件路径上还经过带限平滑), 不是 Zernike 级数, 所以这里的实现把指令
相位**投影到 Zernike 基**上再逐族缩放, 投影残差 (非 Zernike 成分) 原样
保留。这是本实现与论文的一处**已知差异**, 记在这里以免被当成 bug。

论文**没有**信任域, 也**没有**逐轮验收 —— 本仓库的「宁可中止也不提交错误
相位」原则与那一条直接冲突, 所以护栏不在本模块, 而在
:mod:`ao_shaping.optimizer.wfless.slm_model_in_loop` (逐轮信任域 + 接受
判据)。本模块只提供**参数化 + clamp**。

术语: 本模块的中文叙述遵守 ``docs/terminology_glossary.md`` (瞳面 / 指令相位 /
对比择优 / 验证集 …)。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar, Sequence

import numpy as np
import numpy.typing as npt

__all__ = [
    "AFFINE_PARAM_NAMES",
    "FAMILY_NAMES",
    "PupilAffine",
    "TravelScale",
    "apply_pupil_affine",
    "apply_travel_scale",
    "family_index_vector",
    "family_of_noll",
    "travel_gain_tensor",
    "travel_gain_vector",
]


#: τ_θ 的 6 个标量, 顺序即论文的 ``[amp_x, amp_y, off_x, off_y, rot, grid]``。
AFFINE_PARAM_NAMES: tuple[str, ...] = (
    "amp_x",
    "amp_y",
    "off_x",
    "off_y",
    "rot",
    "grid",
)

#: γ 的模式族 (论文按模式族分组, 本实现取 7 族)。
FAMILY_NAMES: tuple[str, ...] = (
    "piston",
    "tip_tilt",
    "defocus",
    "astig",
    "coma",
    "trefoil",
    "higher",
)


def family_of_noll(noll: int) -> int:
    """返回一个 Noll 索引所属的模式族序号。

    分族按 Noll 1976 序的经典族划分 (piston / tip-tilt / defocus /
    astigmatism / coma / trefoil / 其余高阶)。piston 与 tip/tilt 对远场
    **强度**不可辨识 (见
    :meth:`~ao_shaping.algorithm.signal_processing.zernike_coefficient_optimizer.\
ZernikeCoefficientOptimizer._apply_mode_mask`), 但仍单列成族, 这样 γ 的
    自由度数目不随 ``frozen_modes`` 改变。

    Args:
        noll: 1 起的 Noll 索引。

    Returns:
        ``0``..``6`` 的族序号。

    Raises:
        ValueError: ``noll`` 小于 1 时。
    """
    index = int(noll)
    if index < 1:
        raise ValueError(f"noll must be >= 1, got {noll!r}")
    if index == 1:
        return 0
    if index <= 3:
        return 1
    if index == 4:
        return 2
    if index <= 6:
        return 3
    if index <= 8:
        return 4
    if index <= 10:
        return 5
    return 6


@dataclass(frozen=True)
class PupilAffine:
    """τ_θ: 瞳面平面内的仿射自标定量 (6 个标量)。

    语义照论文:

    * ``amp_x`` / ``amp_y`` —— 各向异性缩放, **只作用于振幅包络**;
    * ``off_x`` / ``off_y`` —— 平移, **只作用于振幅包络**, 用归一化坐标
      (瞳面半径归一);
    * ``rot`` —— 旋转 (rad), **只旋转指令相位**;
    * ``grid`` —— 等向缩放, **同时作用于总相位网格与振幅包络**。

    单位变换 = ``(1, 1, 0, 0, 0, 1)``。

    Attributes:
        amp_x: x 方向的振幅包络缩放因子。
        amp_y: y 方向的振幅包络缩放因子。
        off_x: 振幅包络中心的 x 偏移 (归一化瞳面坐标)。
        off_y: 振幅包络中心的 y 偏移 (归一化瞳面坐标)。
        rot: 指令相位的旋转角 (rad)。
        grid: 总相位网格与振幅包络的等向缩放因子。
    """

    amp_x: float = 1.0
    amp_y: float = 1.0
    off_x: float = 0.0
    off_y: float = 0.0
    rot: float = 0.0
    grid: float = 1.0

    #: 振幅包络缩放因子的硬边界。
    AMP_MIN: ClassVar[float] = 0.5
    AMP_MAX: ClassVar[float] = 2.0
    #: 振幅包络中心的归一化偏移硬边界。
    OFF_LIMIT: ClassVar[float] = 1.0
    #: 指令相位旋转角的硬边界 (rad)。
    ROT_LIMIT: ClassVar[float] = 0.5
    #: 等向网格缩放因子的硬边界。
    GRID_MIN: ClassVar[float] = 0.5
    GRID_MAX: ClassVar[float] = 2.0

    @property
    def is_identity(self) -> bool:
        """``True`` 当且仅当本组参数是单位变换。"""
        return (
            float(self.amp_x) == 1.0
            and float(self.amp_y) == 1.0
            and float(self.off_x) == 0.0
            and float(self.off_y) == 0.0
            and float(self.rot) == 0.0
            and float(self.grid) == 1.0
        )

    def to_vector(self) -> npt.NDArray[np.float64]:
        """返回按 :data:`AFFINE_PARAM_NAMES` 顺序排列的 6 维向量。"""
        return np.array(
            [
                float(self.amp_x),
                float(self.amp_y),
                float(self.off_x),
                float(self.off_y),
                float(self.rot),
                float(self.grid),
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_vector(cls, vector: Sequence[float]) -> "PupilAffine":
        """从 6 维向量重建 (顺序同 :data:`AFFINE_PARAM_NAMES`)。"""
        values = np.asarray(vector, dtype=np.float64).ravel()
        if values.size != 6:
            raise ValueError(f"PupilAffine needs 6 values, got {values.size}")
        return cls(
            amp_x=float(values[0]),
            amp_y=float(values[1]),
            off_x=float(values[2]),
            off_y=float(values[3]),
            rot=float(values[4]),
            grid=float(values[5]),
        )

    def clamped(self) -> "PupilAffine":
        """返回把每个分量夹到硬边界内的副本。"""
        return PupilAffine(
            amp_x=float(
                min(max(float(self.amp_x), self.AMP_MIN), self.AMP_MAX)
            ),
            amp_y=float(
                min(max(float(self.amp_y), self.AMP_MIN), self.AMP_MAX)
            ),
            off_x=float(
                min(max(float(self.off_x), -self.OFF_LIMIT), self.OFF_LIMIT)
            ),
            off_y=float(
                min(max(float(self.off_y), -self.OFF_LIMIT), self.OFF_LIMIT)
            ),
            rot=float(min(max(float(self.rot), -self.ROT_LIMIT), self.ROT_LIMIT)),
            grid=float(
                min(max(float(self.grid), self.GRID_MIN), self.GRID_MAX)
            ),
        )

    def step_l2(self, other: "PupilAffine") -> float:
        """返回与 ``other`` 之间的逐参数 L2 步长。

        Args:
            other: 参照状态 (通常是本轮开始时的取值)。

        Returns:
            6 维差向量的 L2 范数 (各分量单位不同, 故只作**相对**步长指标)。
        """
        return float(np.linalg.norm(self.to_vector() - other.to_vector()))

    def max_abs_step(self, other: "PupilAffine") -> float:
        """返回与 ``other`` 之间最大的逐参数绝对变化。

        逐轮信任域用**逐参数**上限而不是 L2 范数: 6 个分量的单位互不相同
        (缩放因子 vs 归一化偏移 vs 弧度), 一个 L2 范数会把它们混在一起。
        """
        return float(np.max(np.abs(self.to_vector() - other.to_vector())))

    def to_dict(self) -> dict[str, float]:
        """返回 JSON 可序列化的字典 (键即参数名)。"""
        return {
            "amp_x": float(self.amp_x),
            "amp_y": float(self.amp_y),
            "off_x": float(self.off_x),
            "off_y": float(self.off_y),
            "rot": float(self.rot),
            "grid": float(self.grid),
        }


@dataclass(frozen=True)
class TravelScale:
    """γ: 逐模式族的行程缩放 (缩放因子 = ``1 + γ``)。

    论文的写法是 ``scale = 1 + γ`` 且逐步 clamp ``γ ≥ −1`` (缩放因子非负)。
    本实现额外加了一条**上界** (``GAMMA_MAX``): 论文那条只防负缩放, 防不住
    一个 runaway 的正 γ —— 而 runaway 的 γ 会把指令相位整体放大到毫无意义。
    上界是本仓库的护栏, 不是论文的。

    Attributes:
        gammas: 逐模式族的 γ 值, 长度须等于 :data:`FAMILY_NAMES` 的长度。
    """

    gammas: tuple[float, ...] = (0.0,) * len(FAMILY_NAMES)

    #: γ 的硬下界 (论文): 缩放因子 ``1 + γ`` 不得为负。
    GAMMA_MIN: ClassVar[float] = -1.0
    #: γ 的硬上界 (本仓库护栏, 非论文)。
    GAMMA_MAX: ClassVar[float] = 2.0

    def __post_init__(self) -> None:
        """校验族数与取值。"""
        values = tuple(float(value) for value in self.gammas)
        if len(values) != len(FAMILY_NAMES):
            raise ValueError(
                f"TravelScale needs {len(FAMILY_NAMES)} family values, got "
                f"{len(values)}"
            )
        for value in values:
            if not math.isfinite(value):
                raise ValueError(f"gamma must be finite, got {value!r}")
        object.__setattr__(self, "gammas", values)

    @property
    def scales(self) -> tuple[float, ...]:
        """返回逐族的缩放因子 ``1 + γ``。"""
        return tuple(1.0 + float(value) for value in self.gammas)

    @property
    def is_identity(self) -> bool:
        """``True`` 当且仅当所有 γ 都是零 (即无行程缩放)。"""
        return all(float(value) == 0.0 for value in self.gammas)

    def to_vector(self) -> npt.NDArray[np.float64]:
        """返回 γ 的向量 (顺序同 :data:`FAMILY_NAMES`)。"""
        return np.asarray(self.gammas, dtype=np.float64)

    @classmethod
    def from_vector(cls, vector: Sequence[float]) -> "TravelScale":
        """从向量重建。"""
        return cls(tuple(float(value) for value in vector))

    def clamped(self) -> "TravelScale":
        """返回把每个 γ 夹到硬边界内的副本。"""
        return TravelScale(
            tuple(
                min(max(float(value), self.GAMMA_MIN), self.GAMMA_MAX)
                for value in self.gammas
            )
        )

    def max_abs_step(self, other: "TravelScale") -> float:
        """返回与 ``other`` 之间最大的逐族绝对变化。"""
        return float(np.max(np.abs(self.to_vector() - other.to_vector())))

    def to_dict(self) -> dict[str, float]:
        """返回 JSON 可序列化的字典 (键即族名)。"""
        return {
            name: float(value) for name, value in zip(FAMILY_NAMES, self.gammas)
        }


def travel_gain_vector(
    gammas: Any, n_coefficients: int
) -> npt.NDArray[np.float64]:
    """把逐族 γ 展开成逐模式的缩放因子向量 (numpy 版, 供离线检查用)。

    ⚠️ 这个版本会**断开** torch 的梯度, 所以可微路径必须用
    :func:`travel_gain_tensor`。

    Args:
        gammas: 逐族 γ (长度 = :data:`FAMILY_NAMES` 长度)。
        n_coefficients: Zernike 模式的个数。

    Returns:
        长度 ``n_coefficients`` 的数组, 第 ``j`` 项是 Noll ``j + 1`` 的缩放
        因子 ``1 + γ[family(j + 1)]``。
    """
    values = np.asarray(
        gammas.detach().cpu().numpy() if hasattr(gammas, "detach") else gammas,
        dtype=np.float64,
    ).ravel()
    if values.size != len(FAMILY_NAMES):
        raise ValueError(
            f"expected {len(FAMILY_NAMES)} family gammas, got {values.size}"
        )
    gains = np.empty(int(n_coefficients), dtype=np.float64)
    for index in range(int(n_coefficients)):
        gains[index] = 1.0 + float(values[family_of_noll(index + 1)])
    return gains


def family_index_vector(n_coefficients: int) -> npt.NDArray[np.int64]:
    """返回每个 Zernike 模式所属族序号的数组。

    Args:
        n_coefficients: Zernike 模式的个数。

    Returns:
        长度 ``n_coefficients`` 的 int64 数组, 第 ``j`` 项是 Noll ``j + 1``
        的族序号。
    """
    return np.array(
        [family_of_noll(index + 1) for index in range(int(n_coefficients))],
        dtype=np.int64,
    )


def travel_gain_tensor(gammas: Any, n_coefficients: int) -> Any:
    """把逐族 γ 展开成逐模式的缩放因子张量 (**保留梯度**)。

    Args:
        gammas: 逐族 γ 张量 ``(7,)``。
        n_coefficients: Zernike 模式的个数。

    Returns:
        长度 ``n_coefficients`` 的张量, 与 ``gammas`` 同设备同 dtype。
    """
    torch = _torch()
    index = torch.as_tensor(
        family_index_vector(n_coefficients),
        device=gammas.device,
        dtype=torch.long,
    )
    return 1.0 + gammas.index_select(0, index)


# ---------------------------------------------------------------------------
# torch 侧的作用函数 (torch 延迟导入: utils 是零 torch 依赖的叶子层)
# ---------------------------------------------------------------------------


def _torch():
    """返回 ``torch`` 模块; 缺失时抛出 :class:`ImportError`。"""
    try:
        import torch as _t
    except ImportError:
        raise ImportError(
            "calibration_terms requires PyTorch. "
            "Install with: uv sync --group ml"
        ) from None
    return _t


def _normalized_grid(
    side: int, device: Any, dtype: Any
) -> tuple[Any, Any]:
    """返回归一化瞳面坐标 ``(u, v)``, 取值范围 ``[-1, 1]``。

    归一化按**网格边缘**而不是按瞳面半径: 这样 ``grid_sample`` 的
    ``align_corners=True`` 约定 (坐标 −1 → 第 0 列, +1 → 末列) 与本函数
    的坐标一一对应, 重采样不会引入半像素的系统偏移。

    Args:
        side: 网格边长 (像素)。
        device: torch 设备。
        dtype: 计算精度。

    Returns:
        ``(u, v)`` 两个 ``(side, side)`` 张量; ``u`` 沿列变化, ``v`` 沿行变化。
    """
    torch = _torch()
    axis = torch.arange(side, device=device, dtype=dtype)
    half = max(float(side - 1) / 2.0, 1.0)
    coord = (axis - half) / half
    v, u = torch.meshgrid(coord, coord, indexing="ij")
    return u, v


def _resample(field: Any, u: Any, v: Any) -> Any:
    """按归一化坐标 ``(u, v)`` 双线性重采样一个二维场。

    越界取边界值 (``padding_mode="border"``): 瞳面之外的相位恒为零, 边界
    外推因此不会把零拉进瞳面。

    Args:
        field: 二维场 ``(H, W)``。
        u: 采样点的 x 坐标 (``[-1, 1]``)。
        v: 采样点的 y 坐标 (``[-1, 1]``)。

    Returns:
        重采样后的二维场, 与 ``field`` 同形同 dtype。
    """
    torch = _torch()
    grid = torch.stack((u, v), dim=-1)[None]
    out = torch.nn.functional.grid_sample(
        field[None, None],
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return out[0, 0]


def apply_pupil_affine(
    command: Any,
    aberration: Any,
    amplitude: Any,
    params: Any,
) -> tuple[Any, Any]:
    """把 τ_θ 作用在正向模型上, 返回 (总相位, 振幅包络)。

    作用顺序 (与论文的语义逐项对应):

    1. ``rot`` **只旋转指令相位** —— 待估像差不跟着转;
    2. ``grid`` 等向缩放**总相位网格** (指令 + 像差之和);
    3. ``amp_x`` / ``amp_y`` / ``off_x`` / ``off_y`` 只作用于**振幅包络**,
       ``grid`` 同时作用于振幅包络。

    坐标约定: 归一化瞳面坐标 ``(u, v) ∈ [-1, 1]²``, 瞳面中心在原点。

    * 指令相位: ``Φ_cmd'(u, v) = Φ_cmd(R(−rot)·(u, v))``;
    * 总相位网格: ``Φ''(u, v) = Φ_total(grid·u, grid·v)``;
    * 振幅包络: ``A'(u, v) = A(grid·(u − off_x)/amp_x, grid·(v − off_y)/amp_y)``。

    振幅在返回前被夹到非负 (论文的 ``o ≥ 0`` 约束)。

    ⚠️ ``grid`` 缩放总相位网格时会**连带缩放待估像差**, 于是 ``grid`` 与
    Zernike 系数存在近似简并 (把系数乘 g 与把 grid 乘 g 的效果部分抵消)。
    论文的模型有同样的结构; 本仓库用**逐轮信任域**限制 ``grid`` 的移动速度
    (见 :mod:`ao_shaping.optimizer.wfless.slm_model_in_loop`), 并且默认关闭。

    Args:
        command: 指令相位张量 ``(R, R)`` (raw 未包裹弧度)。
        aberration: 待估像差张量 ``(R, R)`` (弧度)。
        amplitude: 瞳面振幅张量 ``(R, R)``。
        params: τ_θ 的 6 维张量 (顺序同 :data:`AFFINE_PARAM_NAMES`)。

    Returns:
        ``(总相位, 振幅包络)`` 两个张量。
    """
    torch = _torch()
    amp_x = params[0]
    amp_y = params[1]
    off_x = params[2]
    off_y = params[3]
    rot = params[4]
    grid = params[5]

    u, v = _normalized_grid(
        int(command.shape[-1]), command.device, command.dtype
    )

    # 1. 指令相位旋转 (只作用于指令)。
    cos_r = torch.cos(rot)
    sin_r = torch.sin(rot)
    u_rot = u * cos_r + v * sin_r
    v_rot = -u * sin_r + v * cos_r
    command_rot = _resample(command, u_rot, v_rot)

    # 2. 总相位网格的等向缩放。
    patch = command_rot + aberration
    patch = _resample(patch, grid * u, grid * v)

    # 3. 振幅包络: 各向异性缩放 + 平移 + 等向缩放。
    u_amp = grid * (u - off_x) / amp_x
    v_amp = grid * (v - off_y) / amp_y
    amp = _resample(amplitude, u_amp, v_amp)
    return patch, amp.clamp(min=0.0)


def apply_travel_scale(
    command: Any,
    gammas: Any,
    projector: Any,
    basis_flat: Any,
) -> Any:
    """把逐族行程缩放 γ 作用在**指令相位**上。

    指令相位先投影到 Zernike 基 (最小二乘), 再按模式族把投影系数乘
    ``1 + γ``, 投影残差 (非 Zernike 成分) 原样保留。待估像差**不**参与 ——
    与论文一致 (γ 只作用在指令上)。

    Args:
        command: 指令相位张量 ``(R, R)`` (raw 未包裹弧度)。
        gammas: 逐族 γ 张量 ``(7,)``。
        projector: Zernike 基的伪逆 ``(K, N)`` (N = R·R)。
        basis_flat: Zernike 基展平 ``(K, N)``。

    Returns:
        施加行程缩放后的指令相位, 与 ``command`` 同形。
    """
    torch = _torch()
    side = int(command.shape[-1])
    flat = command.reshape(-1)
    coefficients = projector @ flat
    gains = travel_gain_tensor(gammas, int(coefficients.shape[0]))
    delta = (gains - 1.0) * coefficients
    correction = (delta @ basis_flat).reshape(side, side)
    return command + correction
