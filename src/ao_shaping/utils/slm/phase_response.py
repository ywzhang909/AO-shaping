"""SLM 的灰度→相位**实测响应**（幂律模型），用于把请求相位换算成正确的灰度。

为什么不能沿用"2π ↔ 某个灰度"的线性换算
------------------------------------------------
2026-10-23 在焦条件下的实测 + 拟合（`report/slm/diffraction_efficiency/hw_blaze_20261023/report.md`）：

    Δφ(u) = Δφ_max · u^γ ,   u = gray / gray_max

| 量 | 值 |
|---|---|
| 满命令（1023）处的真实深度 | **1.845π**（< 2π） |
| 幂律指数 γ | **0.808**（<1 ⇒ 低灰度段相位长得**快**） |
| 未调制泄漏 ℓ | 0.51% |
| 由 γ 外推的 2π 灰度 | **1130 > 1023 ⇒ 不可达** |

驱动里记的 ``max_gray=993`` 是**标称**值，与真实响应差两层：尺度（993 vs 1130）与
非线性（γ=0.808）。用它做 ``gray = rad/(2π)·993`` 会：

* 满量程处只给出 1.80π（比请求的 2π 少约 10%）；
* 低灰度段给出**多于**请求的相位（γ<1），1 rad 处约偏 **+25%**。

⇒ 小步可微优化（0.03–0.1 rad）正好落在偏得最多的区间，所以必须用实测响应反解。

本模块只做纯数学（numpy + 标准库），不 import 驱动。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

TWO_PI = 2.0 * np.pi


class PhaseResponseError(ValueError):
    """响应参数不合法。"""


@dataclass(frozen=True)
class PowerLawPhaseResponse:
    """``φ(gray) = depth_at_full · (gray/gray_max) ** gamma``（弧度）。

    ``depth_at_full`` 是**满灰度**处的可达相位深度（本器件 1.845π）。相位超过它时物理上
    不可达：``gray_of_phase`` 会钳到 ``gray_max`` 并在返回值里标出是否发生钳位。
    """

    depth_at_full: float
    gamma: float
    gray_max: int = 1023
    leakage: float = 0.0
    source: str = ""

    def __post_init__(self) -> None:
        if not np.isfinite(self.depth_at_full) or self.depth_at_full <= 0:
            raise PhaseResponseError("depth_at_full must be finite and positive")
        if not np.isfinite(self.gamma) or self.gamma <= 0:
            raise PhaseResponseError("gamma must be finite and positive")
        if int(self.gray_max) < 2:
            raise PhaseResponseError("gray_max must be >= 2")

    # -- 正向：灰度 → 相位 ------------------------------------------------
    def phase_of_gray(self, gray: npt.ArrayLike) -> npt.NDArray[np.float64]:
        values = np.clip(np.asarray(gray, dtype=np.float64), 0.0, float(self.gray_max))
        return self.depth_at_full * (values / float(self.gray_max)) ** self.gamma

    # -- 反向：相位 → 灰度 ------------------------------------------------
    def gray_of_phase(self, phase: npt.ArrayLike) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
        """返回 ``(gray, clamped)``；``clamped`` 标出被物理上限截断的像素。"""
        values = np.asarray(phase, dtype=np.float64)
        clamped = values > self.depth_at_full
        clipped = np.clip(values, 0.0, self.depth_at_full)
        return float(self.gray_max) * (clipped / self.depth_at_full) ** (1.0 / self.gamma), clamped

    def gray_for_two_pi(self) -> float:
        """反解"2π 需要多少灰度"；> ``gray_max`` 表示该器件做不到 2π。"""
        return float(self.gray_max) * (TWO_PI / self.depth_at_full) ** (1.0 / self.gamma)

    @property
    def depth_ratio(self) -> float:
        """满命令深度 / 2π（本器件 ≈0.9225）。"""
        return float(self.depth_at_full / TWO_PI)

    def local_gain(self, gray: float) -> float:
        """某灰度处 dφ/dgray（弧度/灰度）；小步可微优化真正感受到的就是它。"""
        g = float(np.clip(gray, 1e-9, float(self.gray_max)))
        return float(self.depth_at_full * self.gamma * (g ** (self.gamma - 1.0))
                     / (float(self.gray_max) ** self.gamma))

    # -- 序列化 ------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return dict(depth_at_full=self.depth_at_full, gamma=self.gamma, gray_max=int(self.gray_max),
                    leakage=self.leakage, source=self.source,
                    depth_ratio=self.depth_ratio, gray_for_two_pi=self.gray_for_two_pi())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PowerLawPhaseResponse":
        return cls(depth_at_full=float(data["depth_at_full"]), gamma=float(data["gamma"]),
                   gray_max=int(data.get("gray_max", 1023)), leakage=float(data.get("leakage", 0.0)),
                   source=str(data.get("source", "")))


#: 本器件在 1064 nm、在焦条件下的实测响应（P=32 px、x 轴、41 点扫描）。
MEASURED_HW_20261023 = PowerLawPhaseResponse(
    depth_at_full=1.845 * np.pi,
    gamma=0.808,
    gray_max=1023,
    leakage=0.0051,
    source="report/slm/diffraction_efficiency/hw_blaze_20261023/report.md (in-focus, 1064 nm)",
)


def load_phase_response(path: str | Path) -> PowerLawPhaseResponse:
    """从 JSON 读响应（``depth_at_full`` / ``gamma`` 必填）。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return PowerLawPhaseResponse.from_dict(data)


def save_phase_response(response: PowerLawPhaseResponse, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(response.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def describe_linear_error(response: PowerLawPhaseResponse, *, gray: int = 158) -> dict[str, float]:
    """对照"线性 2π↔gray_max"换算的误差（便于写报告/日志）。"""
    phase_true = float(response.phase_of_gray(gray))
    phase_linear = TWO_PI * gray / float(response.gray_max)
    return dict(gray=float(gray), phase_true=phase_true, phase_linear=phase_linear,
                relative_error=(phase_true - phase_linear) / max(phase_linear, 1e-12))
