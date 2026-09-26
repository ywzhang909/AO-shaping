"""FourierGSNet — deep-unrolled Gerchberg–Saxton beam shaping.

Implements the algorithm of Yan, Holenderski & Meratnia,
"Efficient Gerchberg–Saxton algorithm deep unrolling for phase retrieval with
a complex forward path", Advanced Photonics Nexus 5(2):026005 (2026).

The network predicts an SLM phase mask from ``(source_intensity,
target_intensity)`` in a single forward pass, by unrolling the classical GS
iteration into a stack of differentiable layers, each pairing the physical
amplitude-constraint exchange with a conditioned U-Net phase refinement.
"""

from __future__ import annotations

from ml.gsnet.model import (
    ConditionUNet,
    FFTLayer,
    FourierGSNet,
    angular_difference,
    count_parameters,
)
from ml.gsnet.dataset import (
    GSShapingDataset,
    compute_gs_phase,
    make_source_intensity,
    make_target,
)
from ml.gsnet.losses import ShapingLosses
from ml.gsnet.train import (
    TrainResult,
    circular_mse,
    intensity_mse,
    shaping_loss,
    train_gsnet,
)
from ml.gsnet.evaluate import (
    EvalSummary,
    SampleMetrics,
    compute_sample_metrics,
    divergence_metric,
    evaluate_model,
    phase_mae,
)

__all__ = [
    "ConditionUNet",
    "FFTLayer",
    "FourierGSNet",
    "angular_difference",
    "count_parameters",
    "GSShapingDataset",
    "compute_gs_phase",
    "make_source_intensity",
    "make_target",
    "ShapingLosses",
    "TrainResult",
    "circular_mse",
    "intensity_mse",
    "shaping_loss",
    "train_gsnet",
    "EvalSummary",
    "SampleMetrics",
    "compute_sample_metrics",
    "divergence_metric",
    "evaluate_model",
    "phase_mae",
]