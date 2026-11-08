"""Checkpoint-aware wrapper for the issue #237 registration direction gate.

The calculation is performed by executing ``phase_scope_direction_20261107.py``
without modifying or copying its statistics code.  The only runtime hook replaces
the local ``FrozenDefocusParameters`` name while that script's source is executing,
injecting a checkpoint or a command-line registration shift.  This keeps the
frame filtering, predictions, crop, ``direction_agreement``, and median/mean/std
statistics on the archived command-scope baseline path.

Pure offline only: it reads archived ``npz``/JSON files and never opens the SLM,
camera, or laser.  It deliberately does not import ``ao_shaping.drivers``.
"""
from __future__ import annotations

import argparse
import builtins
import contextlib
import json
import sys
import tempfile
import types
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ao_shaping.algorithm.signal_processing.defocus_forward_core import PHASE_SIGN_SCOPES
from ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity import (
    FrozenDefocusParameters,
    load_frozen_parameters,
)

PHASE_SCOPE_SCRIPT = ROOT / "scripts" / "slm" / "phase_scope_direction_20261107.py"
DEFAULT_OUTPUT = ROOT / "data" / "debug" / "slm_transfer" / "registration_eval_20261108" / "direction_gate.json"
DEVICE_STATEMENT = (
    "离线分析，未占用任何设备：未打开 SLM / 相机 / 激光，未 import 任何 "
    "ao_shaping.drivers 模块；全部输入来自磁盘归档 NPZ 帧与冻结拟合/checkpoint，"
    "预测和方向门统计均在 CPU 上确定性执行。"
)


def _load_checkpoint(checkpoint: Path) -> FrozenDefocusParameters:
    """Load a production ``frozen_parameters.json`` file (or its directory)."""
    return load_frozen_parameters(checkpoint.resolve())


def _registration_text(value: tuple[float, float] | None) -> str:
    if value is None:
        return "None"
    return f"({value[0]:.9g}, {value[1]:.9g})"


def _execute_phase_scope_source(argv: list[str], *, checkpoint: FrozenDefocusParameters | None,
                                registration_override: tuple[float, float] | None) -> dict[str, Any]:
    """Execute the old evaluator source unchanged, except its params constructor."""
    if not PHASE_SCOPE_SCRIPT.is_file():
        raise FileNotFoundError(PHASE_SCOPE_SCRIPT)
    source = PHASE_SCOPE_SCRIPT.read_text(encoding="utf-8")
    code = compile(source, str(PHASE_SCOPE_SCRIPT), "exec")
    namespace: dict[str, Any] = {
        "__name__": "__main__",
        "__file__": str(PHASE_SCOPE_SCRIPT),
        "__package__": "",
    }

    def parameter_factory(**kwargs: Any) -> FrozenDefocusParameters:
        params = FrozenDefocusParameters(**kwargs)
        if checkpoint is not None:
            params = replace(
                checkpoint,
                phase_sign=float(kwargs["phase_sign"]),
                phase_sign_scope=str(kwargs["phase_sign_scope"]),
            )
        if registration_override is not None:
            params = replace(params, registration_shift_ccd_px=registration_override)
        elif checkpoint is not None:
            params = replace(params, registration_shift_ccd_px=checkpoint.registration_shift_ccd_px)
        return params

    original_import = builtins.__import__

    def patched_import(name: str, globals: dict[str, Any] | None = None,
                       locals: dict[str, Any] | None = None, fromlist: tuple[str, ...] = (),
                       level: int = 0) -> types.ModuleType:
        module = original_import(name, globals, locals, fromlist, level)
        target = "ao_shaping.algorithm.signal_processing.differentiable_defocus_uniformity"
        if level == 0 and name == target and "FrozenDefocusParameters" in fromlist:
            proxy = types.ModuleType(module.__name__)
            for attribute in fromlist:
                setattr(proxy, attribute, getattr(module, attribute))
            proxy.FrozenDefocusParameters = parameter_factory
            return proxy
        return module

    namespace_builtins = dict(vars(builtins))
    namespace_builtins["__import__"] = patched_import
    namespace["__builtins__"] = namespace_builtins

    saved_argv = sys.argv.copy()
    try:
        sys.argv = argv
        exec(code, namespace)
    finally:
        sys.argv = saved_argv
    return namespace


def _summary(namespace: dict[str, Any]) -> dict[str, Any]:
    """Map the old evaluator's stats onto the requested direction-gate schema."""
    stats = namespace["stats"]
    corrs = [float(value) for value in namespace["corrs"] if np.isfinite(value)]
    effective = namespace["forward"].params.registration_shift_ccd_px
    return {
        "median": None if stats["median"] is None else float(stats["median"]),
        "mean": None if stats["mean"] is None else float(stats["mean"]),
        "std": None if stats["std_ddof1"] is None else float(stats["std_ddof1"]),
        "n_ge_0.5": int(stats["n_ge_0p5"]),
        "n": int(stats["n"]),
        "corrs": corrs,
        "registration_shift_ccd_px": None if effective is None else [float(effective[0]), float(effective[1])],
    }


def _phase_argv(*, run: Path, amp: float, fit: Path, model: str, modes: list[int],
                phase_sign: float, phase_sign_scope: str, origin: str,
                output: Path) -> list[str]:
    return [
        str(PHASE_SCOPE_SCRIPT), "--run", str(run), "--amp", str(amp), "--fit", str(fit),
        "--model", model, "--modes", *[str(mode) for mode in modes], "--phase-sign", str(phase_sign),
        "--phase-sign-scope", phase_sign_scope, "--origin", origin, "--out", str(output),
    ]


def run_one_amp(*, amp: float, run: Path, fit: Path, model: str, modes: list[int],
                phase_sign: float, phase_sign_scope: str, origin: str,
                checkpoint: FrozenDefocusParameters | None = None,
                registration_override: tuple[float, float] | None = None,
                fit_override: Path | None = None) -> dict[str, Any]:
    """Run one original-口径 amp group and return its summary."""
    with tempfile.TemporaryDirectory(prefix="registration_direction_gate_") as directory:
        output = Path(directory) / "phase_scope.json"
        argv = _phase_argv(
            run=run, amp=amp, fit=fit_override or fit, model=model, modes=modes,
            phase_sign=phase_sign, phase_sign_scope=phase_sign_scope, origin=origin,
            output=output,
        )
        namespace = _execute_phase_scope_source(
            argv, checkpoint=checkpoint, registration_override=registration_override,
        )
        return _summary(namespace)


def _amp_key(amp: float) -> str:
    return f"{amp:.2f}"


def _run_arm(amps: list[float], *, run: Path, fit: Path, model: str, modes: list[int],
             phase_sign: float, phase_sign_scope: str, origin: str,
             checkpoint: Path | None, registration_override: tuple[float, float] | None) -> dict[str, Any]:
    loaded = _load_checkpoint(checkpoint) if checkpoint is not None else None
    fit_override: Path | None = None
    if loaded is not None:
        # The archived evaluator builds its camera crop around ``MODEL['origin']``.
        # Supply the checkpoint origin through an equivalent, temporary fit view so
        # the unchanged source follows the same geometry path.
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
            fit_override = Path(handle.name)
            json.dump({
                "ramp_matrix": [list(row) for row in loaded.ramp_matrix],
                "models": {model: {
                    "defocus_diopter": loaded.defocus_diopter,
                    "origin": list(loaded.origin_xy),
                    "amplitude_center": list(loaded.amplitude_center_xy),
                    "amplitude_width": list(loaded.amplitude_width_xy),
                    "residual_noll_radians": {str(k): v for k, v in loaded.residual_noll_radians.items()},
                    "gain": loaded.gain,
                    "background": loaded.background,
                }},
            }, handle)
    groups: dict[str, dict[str, Any]] = {}
    try:
        for amp in amps:
            groups[_amp_key(amp)] = run_one_amp(
                amp=amp, run=run, fit=fit, model=model, modes=modes, phase_sign=phase_sign,
                phase_sign_scope=phase_sign_scope, origin=origin,
                checkpoint=loaded, registration_override=registration_override,
                fit_override=fit_override,
            )
    finally:
        if fit_override is not None:
            fit_override.unlink(missing_ok=True)
    effective = next(iter(groups.values()))["registration_shift_ccd_px"] if groups else None
    return {
        "regime": "checkpoint" if loaded is not None else "fit",
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "registration_shift_ccd_px": effective,
        "groups": groups,
        "device_occupied": False,
        "device_statement": DEVICE_STATEMENT,
    }


def _difference(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    keys = ("median", "mean", "std", "n_ge_0.5")
    result: dict[str, Any] = {}
    for amp in current["groups"]:
        lhs = current["groups"][amp]
        rhs = baseline["groups"].get(amp)
        if rhs is None:
            result[amp] = {"missing_baseline": True}
            continue
        result[amp] = {
            key: None if lhs[key] is None or rhs[key] is None else float(lhs[key] - rhs[key])
            for key in keys
        }
        result[amp]["corrs_same_n"] = len(lhs["corrs"]) == len(rhs["corrs"])
    return result


def _passes_groups(groups: dict[str, dict[str, Any]]) -> dict[str, bool]:
    return {amp: bool(group["median"] is not None and group["median"] >= 0.50) for amp, group in groups.items()}


def _median_regressed(baseline: dict[str, Any], current: dict[str, Any]) -> bool:
    for amp, group in current["groups"].items():
        reference = baseline["groups"].get(amp)
        if reference and group["median"] is not None and reference["median"] is not None:
            if float(group["median"]) < float(reference["median"]) - 0.05:
                return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--amp", type=float, nargs="+", required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--registration-shift", type=float, nargs=2, metavar=("DY", "DX"), default=None)
    parser.add_argument("--origin", choices=("model", "centroid"), default="model")
    parser.add_argument("--model", default="M1_defocus")
    parser.add_argument("--modes", type=int, nargs="+", default=list(range(5, 12)))
    parser.add_argument("--phase-sign", type=float, default=-1.0, choices=(-1.0, 1.0))
    parser.add_argument("--phase-sign-scope", default="command", choices=PHASE_SIGN_SCOPES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    fit = args.fit.resolve()
    checkpoint = args.checkpoint.resolve() if args.checkpoint is not None else None
    override = tuple(args.registration_shift) if args.registration_shift is not None else None
    kwargs = dict(
        run=args.run.resolve(), fit=fit, model=args.model, modes=args.modes,
        phase_sign=args.phase_sign, phase_sign_scope=args.phase_sign_scope,
        origin=args.origin,
    )

    off = _run_arm(args.amp, checkpoint=checkpoint, registration_override=None, **kwargs)
    result: dict[str, Any] = {
        "schema": "registration_direction_gate/1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "command": "scripts/slm/registration_direction_gate_20261108.py",
        "device_occupied": False,
        "device_statement": DEVICE_STATEMENT,
        "evaluator_reused": str(PHASE_SCOPE_SCRIPT.relative_to(ROOT)),
        "reuse_method": "executes the archived evaluator source unchanged; only the FrozenDefocusParameters construction is hooked",
        "config": {
            "run": str(args.run), "fit": str(args.fit), "model": args.model, "modes": args.modes,
            "phase_sign": args.phase_sign, "phase_sign_scope": args.phase_sign_scope,
            "origin": args.origin, "amps": args.amp,
            "requested_checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "requested_registration_shift": list(args.registration_shift) if args.registration_shift else None,
        },
        "arms": {"registration_off": off},
        "gate": {
            "threshold_median": 0.50,
            "passes_registration_off": _passes_groups(off["groups"]),
        },
    }

    # A single arm remains the documented one-run result.  Providing the shift runs
    # A/B in one deterministic command so the final artifact contains deltas.
    if override is not None:
        on = _run_arm(args.amp, checkpoint=checkpoint, registration_override=override, **kwargs)
        negative_override = None
        if _median_regressed(off, on) and override is not None:
            negative_override = (-override[0], -override[1])
            negative = _run_arm(args.amp, checkpoint=checkpoint, registration_override=negative_override, **kwargs)
        else:
            negative = None
        result["arms"] |= {"registration_on": on}
        if negative is not None:
            result["arms"]["registration_on_negated"] = negative
        result["registration_diff"] = _difference(on, off)
        result["gate"] |= {
            "passes_registration_on": _passes_groups(on["groups"]),
            "median_regressed_more_than_0.05": _median_regressed(off, on),
        }
        if negative is not None:
            result["gate"]["negated_registration_ran"] = True
            result["gate"]["passes_registration_on_negated"] = _passes_groups(negative["groups"])
        # The top-level schema is the requested current (on) arm, while arms keeps A/B.
        result["regime"] = on["regime"]
        result["checkpoint"] = on["checkpoint"]
        result["registration_shift_ccd_px"] = on["registration_shift_ccd_px"]
        result["groups"] = on["groups"]
    else:
        result["regime"] = off["regime"]
        result["checkpoint"] = off["checkpoint"]
        result["registration_shift_ccd_px"] = off["registration_shift_ccd_px"]
        result["groups"] = off["groups"]

    imported_drivers = sorted(
        name for name in sys.modules
        if name == "ao_shaping.drivers" or name.startswith("ao_shaping.drivers.")
    )
    if imported_drivers:
        raise RuntimeError(f"offline direction gate unexpectedly imported drivers: {imported_drivers}")

    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"registration off: {_registration_text(off['registration_shift_ccd_px'])}")
    if override is not None:
        print(f"registration on: {_registration_text(result['registration_shift_ccd_px'])}")
    print(f"wrote: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
