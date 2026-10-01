"""Static-vs-dynamic turbulence comparison in the ``slm_pib_sim`` report.

The report is a pure offline reader, so these tests build synthetic run
directories (fake PKL + sidecar + disturbance companion) rather than depending
on a real hardware or simulation run. They pin the two things the comparison
stands on: that each run's tag comes from the companion manifest (so two runs
cannot collide on figure filenames), and that the report's claims are derived
from the recorded data rather than asserted.
"""

from __future__ import annotations

import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

_MODULE_NAME = "generate_slm_pib_sim_report_under_test"
_report = None


def _load_report():
    """Import the report generator by path (it lives outside the package)."""
    global _report
    if _report is None:
        spec = importlib.util.spec_from_file_location(
            _MODULE_NAME, ROOT / "scripts" / "generate_slm_pib_sim_report.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[_MODULE_NAME] = module
        spec.loader.exec_module(module)
        _report = module
    return _report


def _write_run(
    root: Path,
    stamp: str,
    *,
    tag: str | None,
    mode: str = "static",
    sigma_total: float = 0.6691,
    rms_values: list[float] | None = None,
    n_screens: int = 1,
    epochs: int = 4,
) -> Path:
    """Create a synthetic ``slm_pib_shape_<stamp>/<stamp>/`` artifact dir."""
    run_dir = root / f"slm_pib_shape_{stamp}" / stamp
    run_dir.mkdir(parents=True)

    history = {
        epoch: {
            "J": -2.0 + 0.05 * epoch,
            "_p%": 0.20 + 0.01 * epoch,
            "_c": list(np.linspace(0.0, 0.10, 15)),
            "_img": np.full((16, 16), 1.0 + epoch, dtype=float),
        }
        for epoch in range(epochs)
    }
    with (run_dir / "run.pkl").open("wb") as handle:
        pickle.dump(history, handle)
    (run_dir / "run.json").write_text(
        json.dumps({"objective": "shape", "cam_type": "sim"}), encoding="utf-8"
    )

    if tag is not None:
        rms = rms_values if rms_values is not None else [0.6691] * epochs
        manifest = {
            "enabled": mode != "none",
            "mode": mode,
            "config": {
                "mode": mode,
                "cn2": 2e-13,
                "distance_m": 500.0,
                "l_max": 30.0,
                "l_min": 2e-3,
                "wavelength_m": 1064e-9,
                "pixel_pitch_m": 8e-6,
                "thermal_halo_pv_waves": 0.30,
                "thermal_halo_radius_px": 600.0,
                "halo_noll": [[4, -1.0], [11, -1.0]],
                "halo_n_max": 4,
                "seed": 20261001,
            },
            "measured": {
                "sigma_turb_rad": 0.3741,
                "sigma_halo_rad": 0.4049,
                "sigma_total_rad": sigma_total,
                "streaks_used": float(n_screens),
                "calls": float(len(rms)),
                "beam_w0": 400.0,
                "panel_pixels": 2304000.0,
            },
            "run": {
                "tag": tag,
                "epochs": epochs,
                "objective": "shape",
                "target_shape": "square",
                "cam_type": "sim",
                "seed": 42,
                "screens_archived": n_screens,
                "streaks_total": n_screens,
            },
        }
        (run_dir / "disturbance.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        rng = np.random.default_rng(0)
        np.savez_compressed(
            run_dir / "disturbance.npz",
            screens=(rng.random((n_screens, 6, 8)) * 2 - 1).astype(np.float32),
            streak_indices=np.arange(n_screens, dtype=np.int32),
            call_rms=np.asarray(rms, dtype=np.float64),
            call_streak_index=np.arange(len(rms), dtype=np.int32) % max(n_screens, 1),
            archive_factor=np.int32(8),
        )
    return run_dir


def _render(tmp_path: Path, argv: list[str], monkeypatch) -> Path:
    mod = _load_report()
    out_dir = tmp_path / "docs_out"
    monkeypatch.setattr(
        sys, "argv", ["generate_slm_pib_sim_report.py", "--out", str(out_dir), *argv]
    )
    mod.cli()
    return out_dir / "report.md"


def _links(markdown: str) -> list[str]:
    import re

    return re.findall(r"!\[[^\]]*\]\(([^)]+)\)", markdown)


def _resolve(out_dir: Path, link: str) -> Path:
    path = Path(link)
    return path if path.is_absolute() else out_dir / link


# --- trace classification (the claim behind the proof figure) ----------------


def test_classify_trace_flat_for_constant_and_single_sample() -> None:
    mod = _load_report()
    assert mod.classify_disturbance_trace([0.5, 0.5, 0.5, 0.5]) == "flat"
    assert mod.classify_disturbance_trace([0.5]) == "flat"
    assert mod.classify_disturbance_trace([]) == "flat"


def test_classify_trace_varying_when_values_differ() -> None:
    mod = _load_report()
    assert mod.classify_disturbance_trace([0.5, 0.6, 0.45]) == "varying"


def test_classify_trace_uses_a_tight_tolerance() -> None:
    mod = _load_report()
    assert mod.classify_disturbance_trace([0.5, 0.5 + 1e-13]) == "flat"
    assert mod.classify_disturbance_trace([0.5, 0.5 + 1e-6]) == "varying"
    assert mod.classify_disturbance_trace([0.5, 0.51]) == "varying"


# --- tagging -----------------------------------------------------------------


def test_tag_is_derived_from_the_manifest_not_run_index(tmp_path: Path, monkeypatch) -> None:
    run_dir = _write_run(tmp_path, "20260101_000001", tag="static")
    run = _load_report().load_run(run_dir)
    assert _load_report().companion_tag(run) == "static"
    assert _load_report().unique_tags([run]) == ["static"]


def test_tags_fall_back_to_run_index_without_a_companion(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, "20260101_000002", tag=None)
    run = _load_report().load_run(run_dir)
    assert _load_report().companion_tag(run) is None
    assert _load_report().unique_tags([run]) == ["run0"]


def test_tags_are_made_unique_when_two_runs_share_a_mode(tmp_path: Path) -> None:
    first = _load_report().load_run(_write_run(tmp_path, "20260101_000003", tag="static"))
    second = _load_report().load_run(_write_run(tmp_path, "20260101_000004", tag="static"))
    tags = _load_report().unique_tags([first, second])
    assert tags[0] == "static"
    assert tags[1] != tags[0]
    assert len(set(tags)) == 2


def test_disturbance_json_does_not_shadow_the_runner_sidecar(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, "20260101_000005", tag="static")
    run = _load_report().load_run(run_dir)
    # `disturbance.json` is also a *.json here; the sidecar must still win.
    assert run["payload"].get("objective") == "shape"


# --- rendering ---------------------------------------------------------------


def test_two_runs_produce_distinct_figure_filenames(tmp_path: Path, monkeypatch) -> None:
    """Scenario S8: tags must come from the manifest, so filenames cannot collide."""
    _write_run(tmp_path, "20260101_000010", tag="static")
    _write_run(
        tmp_path,
        "20260101_000011",
        tag="dynamic",
        mode="dynamic",
        rms_values=[0.6, 0.61, 0.59, 0.62],
        n_screens=4,
    )
    report_md = _render(tmp_path, ["--debug-root", str(tmp_path), "--max-runs", "2"], monkeypatch)
    text = report_md.read_text()

    assert "运行 `static`" in text
    assert "运行 `dynamic`" in text
    assert "__" not in text  # sanity: no mangled tag

    fig_dir = report_md.parent / "figures"
    names = sorted(p.name for p in fig_dir.glob("*.png"))
    assert any(n.startswith("static_") for n in names)
    assert any(n.startswith("dynamic_") for n in names)
    assert len(names) == len(set(names))


def test_all_markdown_figure_links_resolve(tmp_path: Path, monkeypatch) -> None:
    """Scenario S1: every rendered link must point at a file that exists."""
    _write_run(tmp_path, "20260101_000020", tag="static")
    _write_run(
        tmp_path,
        "20260101_000021",
        tag="dynamic",
        mode="dynamic",
        rms_values=[0.6, 0.61, 0.59, 0.62],
        n_screens=4,
    )
    report_md = _render(tmp_path, ["--debug-root", str(tmp_path), "--max-runs", "2"], monkeypatch)
    out_dir = report_md.parent
    links = _links(report_md.read_text())
    assert links, "report should embed figures"
    missing = [link for link in links if not _resolve(out_dir, link).exists()]
    assert not missing, f"broken figure links: {missing}"


def test_static_and_dynamic_traces_are_classified_differently(
    tmp_path: Path, monkeypatch
) -> None:
    _write_run(tmp_path, "20260101_000030", tag="static", rms_values=[0.6691] * 4)
    _write_run(
        tmp_path,
        "20260101_000031",
        tag="dynamic",
        mode="dynamic",
        rms_values=[0.60, 0.63, 0.58, 0.66],
        n_screens=4,
    )
    text = _render(
        tmp_path, ["--debug-root", str(tmp_path), "--max-runs", "2"], monkeypatch
    ).read_text()
    assert "| flat |" in text
    assert "| varying |" in text


def test_manifest_is_rendered_with_measured_values(tmp_path: Path, monkeypatch) -> None:
    _write_run(tmp_path, "20260101_000040", tag="static", sigma_total=0.6691)
    text = _render(tmp_path, ["--debug-root", str(tmp_path), "--max-runs", "1"], monkeypatch).read_text()
    assert "0.6691" in text
    assert "实测" in text


def test_caveats_are_present(tmp_path: Path, monkeypatch) -> None:
    _write_run(tmp_path, "20260101_000050", tag="static")
    _write_run(
        tmp_path, "20260101_000051", tag="dynamic", mode="dynamic", n_screens=4,
        rms_values=[0.6, 0.61, 0.59, 0.62],
    )
    text = _render(
        tmp_path, ["--debug-root", str(tmp_path), "--max-runs", "2"], monkeypatch
    ).read_text()
    assert "没有真实的公里级大气路径" in text
    assert "不是风模型" in text
    assert "noise_gate_window" in text
    assert "Zernike 无法合成真正的方形远场" in text
    assert "稳态低阶" in text, "the thermal halo's steady-state approximation must be stated"


def test_missing_companion_degrades_gracefully(tmp_path: Path, monkeypatch) -> None:
    _write_run(tmp_path, "20260101_000060", tag=None)
    text = _render(tmp_path, ["--debug-root", str(tmp_path), "--max-runs", "1"], monkeypatch).read_text()
    assert "未记录" in text


def test_malformed_companion_json_does_not_raise(tmp_path: Path, monkeypatch) -> None:
    run_dir = _write_run(tmp_path, "20260101_000070", tag="static")
    (run_dir / "disturbance.json").write_text("{ not valid json", encoding="utf-8")
    text = _render(tmp_path, ["--debug-root", str(tmp_path), "--max-runs", "1"], monkeypatch).read_text()
    assert "未记录" in text


def test_none_mode_run_still_renders(tmp_path: Path, monkeypatch) -> None:
    _write_run(tmp_path, "20260101_000080", tag="none", mode="none", n_screens=0, rms_values=[])
    text = _render(tmp_path, ["--debug-root", str(tmp_path), "--max-runs", "1"], monkeypatch).read_text()
    assert "运行 `none`" in text
    assert "体制 `none`" in text
    assert "| 0 | 0 |" in text  # zero streaks, zero evaluations


def test_single_run_with_zero_screens_does_not_raise(tmp_path: Path, monkeypatch) -> None:
    _write_run(tmp_path, "20260101_000090", tag="static", n_screens=0, rms_values=[])
    report_md = _render(tmp_path, ["--debug-root", str(tmp_path), "--max-runs", "1"], monkeypatch)
    assert report_md.exists()
