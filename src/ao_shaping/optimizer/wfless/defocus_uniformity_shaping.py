"""离焦可微整形的块级策略：初值、方向检验、A-B-B-A 复测、接受/拒绝。

纯逻辑：相位序列怎么排、实测 CV 怎么判、什么时候停；硬件 I/O 由协议脚本注入，
便于离线测试（方案 §6 的 A–E 阶段）。

判据来源 `report/shape_trials/defocus_differentiable_test_plan_20261023.md`：
§6 C（A/B+/A/B−/A）、§6 D（A-B-B-A、3× 噪声分辨率、连续 2 块即停、cap 减半）、
§4（目标均值不可分辨 ⇒ 该块无效）。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
    FrozenDefocusParameters,
    load_frozen_parameters,
)


@dataclass(frozen=True)
class FrozenCheckpoint:
    """冻结检查点 + 可追溯信息（方案 §3：保存确切 JSON、参数和 SHA256）。"""

    directory: Path
    parameters: FrozenDefocusParameters
    result_sha256: str
    upstream_sha256: str
    selection_rule: str
    parameters_json: dict[str, Any]

    def describe(self) -> dict[str, Any]:
        return dict(directory=str(self.directory), result_sha256=self.result_sha256,
                    upstream_sha256=self.upstream_sha256, selection_rule=self.selection_rule,
                    parameters=self.parameters_json)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_checkpoint(directory: str | Path, *, fft_size: int = 1024, pixel_samples: int = 3,
                           selection_rule: str | None = None) -> FrozenCheckpoint:
    """读检查点并给出可核对溯源（参数 + 两个 JSON 的 SHA256 + 选择规则）。"""
    ckpt = Path(directory)
    if not (ckpt / "result.json").exists():
        raise FileNotFoundError(f"missing result.json under {ckpt}")
    parameters = load_frozen_parameters(ckpt, fft_size=fft_size, pixel_samples=pixel_samples)
    cfg = json.loads((ckpt / "config.json").read_text(encoding="utf-8"))
    sampling_cfg = json.loads((Path(cfg["source"]) / "config.json").read_text(encoding="utf-8"))
    return FrozenCheckpoint(
        directory=ckpt, parameters=parameters,
        result_sha256=_sha256(ckpt / "result.json"),
        upstream_sha256=_sha256(Path(sampling_cfg["fit"])),
        selection_rule=selection_rule or "未声明（按检查点 result.json 的最终参数使用）",
        parameters_json=parameters.to_dict(),
    )


@dataclass
class DefocusShapingConfig:
    """块级协议参数（默认值即方案首轮设定）。"""

    branch: str = "C"
    checkpoint_dir: Path | None = None
    box_px: int = 60
    inner_shrink_px: int = 3
    direction_test_rms: float = 0.03
    block_rms_cap: float = 0.1
    w_phase: float = 0.01
    w_shape: float = 0.0
    lr: float = 0.01
    max_updates_per_block: int = 10
    max_blocks: int = 5
    accept_multiplier: float = 3.0
    stop_after_consecutive_rejects: int = 2
    focus_dir: Path | None = None
    seed_path: Path | None = None

    def __post_init__(self) -> None:
        if self.branch not in {"U", "C"}:
            raise ValueError("branch must be 'U' or 'C'")
        for name in ("direction_test_rms", "block_rms_cap", "lr"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_updates_per_block < 1 or self.max_blocks < 1:
            raise ValueError("block counts must be >= 1")
        if self.accept_multiplier <= 0:
            raise ValueError("accept_multiplier must be positive")
        if self.branch == "C" and self.focus_dir is None:
            raise ValueError("branch 'C' requires focus_dir (焦面补偿是控制输入)")


@dataclass
class Step:
    """一个待上屏的测量点。"""

    label: str
    phase: np.ndarray
    role: str


@dataclass
class CandidateVerdict:
    accepted: bool
    reason: str
    resolution: float
    delta: float


@dataclass
class ShaperState:
    blocks_done: int = 0
    consecutive_rejects: int = 0
    stopped: bool = False
    stop_reason: str = ""
    history: list[dict[str, Any]] = field(default_factory=list)


class DefocusUniformityShaper:
    """块级协议：方向检验 → 逐块 A-B-B-A → 接受/拒绝/停止。"""

    def __init__(self, config: DefocusShapingConfig) -> None:
        self.config = config
        self.state = ShaperState()

    def plan_direction_test(self, seed: np.ndarray, step_phase: np.ndarray) -> list[Step]:
        """A / B+ / A / B− / A（B± = seed ± step）。"""
        if seed.shape != step_phase.shape:
            raise ValueError("seed and step must share shape")
        return [Step("A", seed, "baseline"), Step("B+", seed + step_phase, "candidate"),
                Step("A", seed, "baseline"), Step("B-", seed - step_phase, "candidate"),
                Step("A", seed, "baseline")]

    def plan_block(self, baseline: np.ndarray, candidate: np.ndarray) -> list[Step]:
        """A-B-B-A：候选前后各一次基线，用于扣除漂移。"""
        if baseline.shape != candidate.shape:
            raise ValueError("baseline and candidate must share shape")
        return [Step("A1", baseline, "baseline"), Step("B1", candidate, "candidate"),
                Step("B2", candidate, "candidate"), Step("A2", baseline, "baseline")]

    @staticmethod
    def paired_resolution(baseline: Sequence[float], repeat_scale: float | None = None) -> float:
        """当次配对差的不确定度；``repeat_scale`` 给 3σ 时优先用它。"""
        values = np.asarray(list(baseline), dtype=float)
        if repeat_scale is not None and repeat_scale > 0:
            return float(repeat_scale)
        if values.size < 2:
            return float("nan")
        return float(values.std(ddof=1) / np.sqrt(values.size))

    def evaluate_candidate(self, candidate_cv: float, baseline_cv: Sequence[float],
                           reference_drift_cv: Sequence[float] | None = None) -> CandidateVerdict:
        """实测改善是否超过 ``accept_multiplier × 分辨率``。"""
        baseline = [float(v) for v in baseline_cv]
        if not baseline:
            raise ValueError("baseline_cv must not be empty")
        reference = list(reference_drift_cv) if reference_drift_cv is not None else baseline
        resolution = self.paired_resolution(reference)
        if not np.isfinite(resolution) or resolution <= 0:
            return CandidateVerdict(False, "分辨率不可估（参考重复不足）", float(resolution), float("nan"))
        delta = float(candidate_cv) - float(np.mean(baseline))
        threshold = self.config.accept_multiplier * resolution
        if delta < -threshold:
            return CandidateVerdict(True, f"改善 {abs(delta):.5f} > {threshold:.5f}", resolution, delta)
        return CandidateVerdict(False, f"改善 {abs(delta):.5f} ≤ {threshold:.5f}", resolution, delta)

    def zero_signal_verdict(self, target_mean: float, reference_signal: float, *,
                            multiplier: float = 3.0) -> tuple[bool, str]:
        """目标均值不可分辨时为无效（方案 §4）。"""
        if not np.isfinite(target_mean) or target_mean <= 0:
            return True, "目标区无信号"
        if target_mean <= multiplier * max(reference_signal, 0.0):
            return True, f"目标均值 {target_mean:.2f} ≤ {multiplier:.0f}× 参考信号 {reference_signal:.2f}"
        return False, ""

    def record_block(self, verdict: CandidateVerdict, *, block: int,
                     extra: Mapping[str, Any] | None = None) -> None:
        self.state.blocks_done += 1
        if verdict.accepted:
            self.state.consecutive_rejects = 0
        else:
            self.state.consecutive_rejects += 1
            self.config.block_rms_cap = max(self.config.block_rms_cap / 2.0, 1e-4)
        self.state.history.append(dict(block=block, accepted=verdict.accepted, reason=verdict.reason,
                                       resolution=verdict.resolution, delta=verdict.delta,
                                       rms_cap=self.config.block_rms_cap, **(dict(extra or {}))))
        if self.state.consecutive_rejects >= self.config.stop_after_consecutive_rejects:
            self.state.stopped = True
            self.state.stop_reason = (f"连续 {self.state.consecutive_rejects} 块无可分辨改善；"
                                      f"不增加迭代数（方案 §6）")
        if self.state.blocks_done >= self.config.max_blocks:
            self.state.stopped = True
            self.state.stop_reason = self.state.stop_reason or "达到块数上限"

    @property
    def should_stop(self) -> bool:
        return self.state.stopped

    def summary(self) -> dict[str, Any]:
        return dict(branch=self.config.branch, blocks_done=self.state.blocks_done,
                    stopped=self.state.stopped, stop_reason=self.state.stop_reason,
                    consecutive_rejects=self.state.consecutive_rejects,
                    rms_cap=self.config.block_rms_cap, history=self.state.history)
