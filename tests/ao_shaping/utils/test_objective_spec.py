"""Contract tests for the canonical ``(objective, target_shape)`` vocabulary.

:class:`~ao_shaping.utils.image.target.objective.ObjectiveSpec` is the single
authority for pairing an objective with its target shape, and
:data:`SHAPING_OBJECTIVE_CHOICES` is the single authority for the objective
vocabulary itself. Every consumer must be *derived* from those two - the CLI
``--objective`` choices, the nested debug sidecar, and the offline analysis
script that reads it. These tests pin that whole contract so a rename, a
re-introduced hand-written duplicate list, or a flat/nested sidecar schema drift
fails loudly instead of silently diverging.

Nothing here opens hardware: the CLI case runs in-process through
``click.testing.CliRunner`` and the sidecar case is built on ``tmp_path``.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from click.testing import CliRunner

from ao_shaping.runners.runner_common import (
    CameraParamsPib,
    ObjectiveParamsPib,
    ObjectiveTarget,
    config_payload,
)
from ao_shaping.runners.slm.slm_shaping_runner import run as slm_pib_run
from ao_shaping.utils.image.target import (
    OBJECTIVE_ALLOWED_SHAPES,
    SHAPING_OBJECTIVE_CHOICES,
    TARGET_SHAPE_CHOICES,
    ObjectiveSpec,
)
from ao_shaping.utils.io.file import _DATA_MODE_OBJECTIVE_KEYS


#: Repo root, so the offline analysis script can be loaded from ``scripts/``.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_ANALYSIS_SCRIPT = _REPO_ROOT / "scripts" / "diff_beam_frame_analysis.py"

#: The objective vocabulary, in the exact declared order. A pure *set* assertion
#: would tolerate a reordering that silently renames recorded columns, so the
#: order is part of the contract too.
_EXPECTED_CHOICES: tuple[str, ...] = (
    "pib",
    "radiu",
    "avg_radiu",
    "rmse",
    "rmse_out",
    "shape",
    "roi_pib",
    "rms_pib",
    # FourierGSNet ``1 - Pearson`` loss, migrated to the hardware path
    # (``utils.image.target.metrics.pearson_shape_metric``). It is lower-is-better
    # and energy-blind, so it is listed in ``GUARDED_OBJECTIVES``.
    "pearson",
)

#: ``(objective, target_shape) -> (resolved name, resolved shape)``.
#:
#: Covers every objective at least once and pins all four resolution rules:
#: shape-aware acceptance, ``pib`` -> ``shape`` promotion (versus the ROI-only
#: family that keeps its identity), the ``rectangle`` default of the ROI family,
#: and the ``None`` shape of the bucket/radius objectives. Both spellings of the
#: case-insensitive inputs are present.
_RESOLVE_TABLE: tuple[tuple[str, str | None, str, str | None], ...] = (
    ("pib", None, "pib", None),
    ("pib", "square", "shape", "square"),  # promotion
    ("PIB", "SQUARE", "shape", "square"),  # case-insensitive
    ("roi_pib", None, "roi_pib", "rectangle"),  # ROI family default
    ("roi_pib", "square", "roi_pib", "square"),  # ROI selector, no promotion
    ("rms_pib", None, "rms_pib", "rectangle"),
    ("rms_pib", "circle", "rms_pib", "circle"),
    ("rmse", None, "rmse", "rectangle"),
    ("rmse", "circle", "rmse", "circle"),
    ("rmse_out", None, "rmse_out", "rectangle"),
    ("rmse_out", "square", "rmse_out", "square"),
    ("shape", None, "shape", "rectangle"),  # defining objective, defaulted
    ("shape", "pentagon", "shape", "pentagon"),
    ("radiu", None, "radiu", None),
    ("avg_radiu", None, "avg_radiu", None),
)

#: ``(objective, target_shape, expected ValueError fragment)``.
#:
#: The third case also pins the *ordering* of the two rejection rules: for an
#: unknown objective supplied together with a shape, the shape/objective
#: compatibility check must fire before the final objective-membership check.
_REJECT_TABLE: tuple[tuple[str, str | None, str], ...] = (
    ("bogus", None, "objective must be one of"),
    ("bogus", "square", "target_shape can only be used with"),
    ("pib", "BOGUS", "target_shape must be one of"),
    ("radiu", "square", "target_shape can only be used with"),
    ("avg_radiu", "circle", "target_shape can only be used with"),
)


def _load_analyze_run() -> Any:
    """Import ``scripts/diff_beam_frame_analysis.py`` by file path.

    The analysis script is a standalone ``scripts/`` entry point rather than an
    importable package member, so it is loaded from its path. Kept in a helper
    (rather than at module scope) so a missing optional plotting dependency
    cannot break collection of the other tests.
    """
    spec = importlib.util.spec_from_file_location(
        "diff_beam_frame_analysis", _ANALYSIS_SCRIPT
    )
    assert spec is not None and spec.loader is not None, _ANALYSIS_SCRIPT
    module: ModuleType = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.analyze_run


def _write_minimal_run(run_dir: Path, config: dict[str, Any]) -> None:
    """Write the smallest run directory ``analyze_run`` can read.

    Args:
        run_dir: Directory to populate (must not exist yet).
        config: Contents of ``config.json``.
    """
    frames_dir = run_dir / "frames"
    frames_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    frame = np.zeros((8, 8), dtype=np.float64)
    frame[2:5, 2:5] = 10.0  # a single blob so the 0.5x-peak region is non-empty
    np.save(frames_dir / "frame_0000.npy", frame)
    (frames_dir / "frame_meta.jsonl").write_text(
        json.dumps({"phase": 0, "spot": [3, 3]}) + "\n", encoding="utf-8"
    )


def test_objective_choices_are_the_nine() -> None:
    assert SHAPING_OBJECTIVE_CHOICES == _EXPECTED_CHOICES
    assert set(SHAPING_OBJECTIVE_CHOICES) == set(_EXPECTED_CHOICES)
    assert len(SHAPING_OBJECTIVE_CHOICES) == 9


@pytest.mark.parametrize("objective,target_shape,name,shape", _RESOLVE_TABLE)
def test_resolve_table(
    objective: str, target_shape: str | None, name: str, shape: str | None
) -> None:
    spec = ObjectiveSpec.resolve(objective, target_shape)

    assert spec.name == name
    assert spec.shape == shape
    # The ROI family always resolves to a concrete, known target shape; the
    # bucket/radius family never does.
    if name in ("shape", "roi_pib", "rms_pib", "rmse", "rmse_out"):
        assert spec.shape in TARGET_SHAPE_CHOICES
    else:
        assert spec.shape is None


@pytest.mark.parametrize("objective,target_shape,message", _REJECT_TABLE)
def test_resolve_rejects_invalid(
    objective: str, target_shape: str | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ObjectiveSpec.resolve(objective, target_shape)


def test_objective_allowed_shapes_matrix() -> None:
    assert set(OBJECTIVE_ALLOWED_SHAPES) == set(SHAPING_OBJECTIVE_CHOICES)
    assert OBJECTIVE_ALLOWED_SHAPES["radiu"] is None
    assert OBJECTIVE_ALLOWED_SHAPES["avg_radiu"] is None

    for objective in SHAPING_OBJECTIVE_CHOICES:
        allowed = OBJECTIVE_ALLOWED_SHAPES[objective]
        for shape in TARGET_SHAPE_CHOICES:
            if allowed is None:
                # No literal shape may be accepted: resolve must reject all.
                with pytest.raises(ValueError):
                    ObjectiveSpec.resolve(objective, shape)
                continue
            assert shape in allowed
            # Accepted, so resolve must not raise - and the surviving objective
            # must still be a canonical one (``pib`` promotes itself to ``shape``).
            resolved = ObjectiveSpec.resolve(objective, shape)
            assert resolved.name in SHAPING_OBJECTIVE_CHOICES


def test_cli_objective_choices_match_canonical() -> None:
    # ``slm-pib`` is a click group; the objective options live on the ``spgd``
    # subcommand, so the group's own help would not contain them. Runs fully
    # in-process - no subprocess, no hardware, no skip.
    result = CliRunner().invoke(slm_pib_run, ["spgd", "--help"])
    assert result.exit_code == 0, result.output

    match = re.search(r"--objective\s+\[([^\]]+)\]", result.output)
    assert match, result.output
    cli_choices = {choice.strip() for choice in match.group(1).split("|")}

    assert cli_choices == set(SHAPING_OBJECTIVE_CHOICES), (
        f"CLI --objective choices {sorted(cli_choices)} != "
        f"SHAPING_OBJECTIVE_CHOICES {sorted(SHAPING_OBJECTIVE_CHOICES)}"
    )


def test_forwarders_are_read_only() -> None:
    params = ObjectiveParamsPib(
        target=ObjectiveTarget(name="roi_pib", target_shape="square")
    )

    # Reads still forward, so existing consumers need no edits.
    assert params.name == "roi_pib"
    assert params.target_shape == "square"

    with pytest.raises(AttributeError):
        params.name = "shape"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        params.target_shape = "circle"  # type: ignore[misc]

    # Flat *construction* stays a hard error so a missed call site surfaces
    # loudly instead of being swallowed.
    with pytest.raises(TypeError):
        ObjectiveParamsPib(name="shape")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        CameraParamsPib(name="shape")  # type: ignore[call-arg]


def test_data_mode_keys_are_subset() -> None:
    # The data-mode recorder is an on-disk contract, deliberately narrower than
    # the user-selectable vocabulary - but never a name the vocabulary lost.
    assert set(_DATA_MODE_OBJECTIVE_KEYS) <= set(SHAPING_OBJECTIVE_CHOICES)


def test_config_payload_roundtrip_old_and_new(tmp_path: Path) -> None:
    analyze_run = _load_analyze_run()

    # New sidecar: the objective group is nested, so ``target_shape`` lives under
    # ``target`` and survives a JSON round trip unchanged.
    payload = config_payload(
        ObjectiveParamsPib(target=ObjectiveTarget(name="shape", target_shape="square"))
    )
    assert payload["target"] == {"name": "shape", "target_shape": "square"}
    assert json.loads(json.dumps(payload))["target"]["target_shape"] == "square"

    # Nested sidecar -> the analysis script reads the nested target_shape.
    new_run = tmp_path / "new_run"
    _write_minimal_run(new_run, {"algorithm": "gs", **payload})
    assert analyze_run(new_run)["config"]["target_shape"] == "square"

    # Legacy (pre-merge) sidecar kept ``target_shape`` flat at the top level and
    # must still be readable.
    legacy_run = tmp_path / "legacy_run"
    _write_minimal_run(
        legacy_run, {"algorithm": "gs", "target_shape": "circle"}
    )
    assert analyze_run(legacy_run)["config"]["target_shape"] == "circle"

    # Nested wins when a transitional sidecar somehow carries both.
    both_run = tmp_path / "both_run"
    _write_minimal_run(
        both_run,
        {
            "algorithm": "gs",
            "target_shape": "circle",
            "target": {"name": "shape", "target_shape": "square"},
        },
    )
    assert analyze_run(both_run)["config"]["target_shape"] == "square"
