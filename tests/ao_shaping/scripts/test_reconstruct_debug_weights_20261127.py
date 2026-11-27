# -*- coding: utf-8 -*-
"""离线门禁: 从 iterations.pkl 重建 debug_weights（§5 第 2 项数据缺口）。

覆盖:
- 展平键集与 ``fit._flatten_parameters`` 一致;
- 三件套 + 重建溯源落盘，npz 可读回且首迭代值与源一致;
- 只复核模式不落盘;
- 拒绝覆盖原 ``debug_weights/``。

不打开设备、不重新拟合、不写仓库内文件。
"""

from __future__ import annotations

import importlib
import json
import pickle
from pathlib import Path

import numpy as np

SCRIPT = importlib.import_module("scripts.slm.reconstruct_debug_weights_20261127")


def _parameters(iteration: int) -> dict[str, object]:
    return {
        "defocus_diopter": 0.5 + iteration,
        "gain": 1.0 + iteration / 10,
        "background": 10.0,
        "sigma_ccd_px": 1.2,
        "origin": [100.0 + iteration, 200.0],
        "amplitude_center": [0.7, 0.8],
        "amplitude_width": [2.0, 3.0],
        "residual_noll_radians": {"5": 0.1 * iteration, "6": -0.2},
    }


def _session(tmp_path: Path) -> Path:
    session = tmp_path / "session"
    fit_dir = session / "fit"
    fit_dir.mkdir(parents=True)
    entries = [
        {
            "iteration": index,
            "loss": 1.0 / (index + 1),
            "parameters": _parameters(index),
        }
        for index in range(3)
    ]
    (fit_dir / "iterations.pkl").write_bytes(pickle.dumps(entries))
    return session


def test_load_trajectory_matches_fit_flatten(tmp_path: Path) -> None:
    session = _session(tmp_path)
    trajectory, losses = SCRIPT.load_trajectory(session / "fit" / "iterations.pkl")
    assert len(trajectory) == len(losses) == 3
    assert trajectory[0]["defocus_diopter"] == 0.5
    assert trajectory[1]["origin_0"] == 101.0
    assert trajectory[2]["residual_noll_5"] == 0.2
    assert "bias_counts" not in trajectory[0]


def test_write_reconstruction_and_verify(tmp_path: Path) -> None:
    session = _session(tmp_path)
    out = tmp_path / "recon"
    assert SCRIPT.main(["--session", str(session), "--out", str(out), "--write"]) == 0
    for name in (
        "weights_trajectory.npz",
        "weights_summary.json",
        "weights_history.jsonl",
        "reconstruction.json",
        "verify.json",
    ):
        assert (out / name).is_file(), name

    with np.load(out / "weights_trajectory.npz") as data:
        assert list(data["iterations"]) == [0, 1, 2]
        assert data["defocus_diopter"][0] == 0.5
        assert data["origin_0"][2] == 102.0

    doc = json.loads((out / "reconstruction.json").read_text(encoding="utf-8"))
    assert doc["hardware_used"] is False
    assert doc["refit_performed"] is False
    assert doc["iterations"] == 3

    lines = (out / "weights_history.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert "delta" not in json.loads(lines[0])
    assert "delta" in json.loads(lines[1])


def test_review_mode_writes_nothing(tmp_path: Path) -> None:
    session = _session(tmp_path)
    out = tmp_path / "not_written"
    assert SCRIPT.main(["--session", str(session), "--out", str(out)]) == 0
    assert not out.exists()


def test_refuses_to_overwrite_original_debug_weights(tmp_path: Path) -> None:
    session = _session(tmp_path)
    assert (
        SCRIPT.main(
            [
                "--session",
                str(session),
                "--out",
                str(session / "fit" / "debug_weights"),
                "--write",
            ]
        )
        == 3
    )
