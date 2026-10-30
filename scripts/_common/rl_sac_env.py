"""Eval-env factory shared by the two ``scripts/rl`` SAC run-analysis scripts.

``run_long_sac_experiments.py`` (``_build_env``) and
``sweep_mamba_turbulence_report.py`` (``_build_env_from_config``) carried the
same body under two names; the single copy keeps the former's name. The two
``ao_shaping.optimizer.rl.*`` imports are deferred into the function: the
``rl`` dependency group is optional, and a module-level import here would make
``scripts._common.rl_sac_env`` un-importable on a bare install (the same trap
documented at the import site of ``scripts/oopao/generate_oopao_impact_report.py``).
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _build_env(log_dir: Path):
    from ao_shaping.optimizer.rl.sac_train import build_env as build_static_env
    from ao_shaping.optimizer.rl.sac_turbulence_train import build_env as build_turb_env

    config = json.loads((log_dir / "config.json").read_text(encoding="utf-8"))
    common = {
        "max_steps": config["max_steps"],
        "n_grid": config["n_grid"],
        "n_actuators": config["n_actuators"],
        "n_subapertures": config["n_subapertures"],
        "history_len": config["history_len"],
        "goal_gain": config["goal_gain"],
        "hold_target_steps": config["hold_target_steps"],
        "action_scale": config["action_scale"],
        "time_penalty": config["time_penalty"],
        "action_penalty": config["action_penalty"],
        "saturation_penalty": config["saturation_penalty"],
    }
    if "n_zernike_modes" in config:
        env = build_static_env(
            **common,
            n_zernike_modes=config["n_zernike_modes"],
            zernike_coeff_std=config["zernike_coeff_std"],
            zernike_coeff_clip=config["zernike_coeff_clip"],
        )
        model_path = ROOT / "models" / log_dir.name / "sac_static_final.zip"
        return env, model_path, "static"
    env = build_turb_env(
        **common,
        cn2=config["cn2"],
        screen_step_px=config["screen_step_px"],
        screen_margin_steps=config["screen_margin_steps"],
    )
    model_path = ROOT / "models" / log_dir.name / "sac_turbulence_final.zip"
    return env, model_path, "turbulence"
