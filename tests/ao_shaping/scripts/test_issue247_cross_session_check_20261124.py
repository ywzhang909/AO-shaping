"""跨会话独立复现检查的纯离线验证；只用合成 JSON，不连接硬件、不读真实帧。"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from pathlib import Path
from typing import Any

import pytest

MODULE_NAME = Path(__file__).stem.removeprefix("test_")
SCRIPT = importlib.import_module(f"scripts.slm.{MODULE_NAME}")

if getattr(SCRIPT, "DEPS_MISSING", None):
    pytest.skip(
        f"依赖模块尚未入库 (仅在硬件机工作区): {SCRIPT.DEPS_MISSING}",
        allow_module_level=True,
    )

BASE_IDS = tuple(SCRIPT.EVALUABLE_BASE_IDS)
PINNED_IDS = tuple(SCRIPT.PINNED_MAIN_BASE_IDS)
NON_PINNED_IDS = tuple(SCRIPT.EXCLUDED_P192_BASE_IDS)
P192_IDS = tuple(SCRIPT.SUPPLEMENTAL_BASE_IDS)
SESSION_IDS = tuple(SCRIPT.SESSION_IDS)


# ---------------------------------------------------------------------------
# 合成载荷
# ---------------------------------------------------------------------------
def _main_item(base_id: str, angles: list[float | None]) -> dict[str, Any]:
    rows = [
        {
            "round": f"r{index + 1}",
            "status": "assessed" if angle is not None else "not_assessable",
            "gate_axis_angle_deg": angle,
        }
        for index, angle in enumerate(angles)
    ]
    return {"base_id": base_id, "per_round": rows}


def _items_by_session(
    angles_by_session: dict[str, list[float | None]], base_id: str = "demo"
) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        session_id: {base_id: _main_item(base_id, angles)}
        for session_id, angles in angles_by_session.items()
    }


def _audit_item(
    base_id: str,
    angles: list[float],
    *,
    strong: list[bool] | None = None,
    zero: tuple[float, float] = (100.0, 100.0),
    radius_px: float = 38.0,
) -> dict[str, Any]:
    flags = strong if strong is not None else [True] * len(angles)
    frames = []
    for index, angle in enumerate(angles):
        radians = math.radians(angle)
        frames.append(
            {
                "frame": f"frame_{index}.npy",
                "row": zero[0] + radius_px * math.sin(radians),
                "col": zero[1] + radius_px * math.cos(radians),
                "radius_px": radius_px,
                "peak_minus_background": 420.0,
                "within_core_exclude": True,
                "strong_candidate": flags[index],
            }
        )
    return {
        "pattern_id": base_id,
        "zero_order": {"row": zero[0], "col": zero[1]},
        "n_frames": len(frames),
        "frames": frames,
    }


def _audit_items_by_session(
    angles_by_session: dict[str, list[float]],
    base_id: str = "demo_p192",
    **kwargs: Any,
) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        session_id: {base_id: _audit_item(base_id, angles, **kwargs)}
        for session_id, angles in angles_by_session.items()
    }


# ---------------------------------------------------------------------------
# 主判据
# ---------------------------------------------------------------------------
def test_main_criterion_reproduced_when_all_pairwise_diffs_within_gate() -> None:
    result = SCRIPT.main_pattern_verdict(
        _items_by_session(
            {
                "s1": [1.0, 1.0, 1.0],
                "s2": [1.8, 1.8, 1.8],
                "s3": [2.2, 2.2, 2.2],
            }
        ),
        "demo",
    )

    assert result["verdict"] == "reproduced"
    assert result["max_pairwise_diff_deg"] == pytest.approx(1.2)
    assert set(result["pairwise_diff_deg"]) == {"s1-s2", "s1-s3", "s2-s3"}
    assert all(
        session["n_assessable_rounds"] == 3
        for session in result["sessions"].values()
    )


def test_main_criterion_not_reproduced_when_one_session_breaches_gate() -> None:
    result = SCRIPT.main_pattern_verdict(
        _items_by_session(
            {
                "s1": [1.0, 1.0, 1.0],
                "s2": [1.8, 1.8, 1.8],
                "s3": [4.0, 4.0, 4.0],
            }
        ),
        "demo",
    )

    assert result["verdict"] == "not_reproduced"
    assert result["max_pairwise_diff_deg"] == pytest.approx(3.0)
    assert result["reason"] is not None and "2.0" in result["reason"]


def test_main_criterion_folds_wrap_at_180_without_fake_359_diff() -> None:
    result = SCRIPT.main_pattern_verdict(
        _items_by_session(
            {
                "s1": [179.5, 179.5],
                "s2": [-179.5, -179.5],
                "s3": [0.5, 0.5],
            }
        ),
        "demo",
    )

    assert result["verdict"] == "reproduced"
    assert result["max_pairwise_diff_deg"] == pytest.approx(1.0)
    assert all(
        diff <= 90.0 for diff in result["pairwise_diff_deg"].values()
    )


def test_main_criterion_session_with_one_round_is_not_assessable() -> None:
    result = SCRIPT.main_pattern_verdict(
        _items_by_session(
            {
                "s1": [10.0, 10.0, 10.0],
                "s2": [10.0, None, None],
                "s3": [10.0, 10.0, 10.0],
            }
        ),
        "demo",
    )

    assert result["verdict"] == "not_assessable"
    assert result["max_pairwise_diff_deg"] is None
    assert result["pairwise_diff_deg"] == {}
    assert result["sessions"]["s2"]["axial_mean_deg"] is None
    assert result["sessions"]["s2"]["n_assessable_rounds"] == 1
    assert result["reason"] is not None and "s2" in result["reason"]
    # 另两个会话一致也不能替代缺失的 s2。
    assert result["verdict"] != "reproduced"


def test_main_criterion_never_counts_p192_as_assessable_when_missing() -> None:
    items = _items_by_session({"s1": [None] * 3, "s2": [None] * 3, "s3": [None] * 3})
    result = SCRIPT.main_pattern_verdict(items, "demo")

    assert result["verdict"] == "not_assessable"
    assert result["sessions"]["s1"]["gate_angles_deg"] == []


def _reports(
    angles_by_session: dict[str, list[float | None]],
    overrides: dict[str, list[float | None]] | None = None,
    per_session_overrides: dict[str, dict[str, list[float | None]]] | None = None,
) -> list[dict[str, Any]]:
    """构造三会话主判据输入。

    ``overrides`` 按 base_id 覆盖所有会话；``per_session_overrides`` 按
    ``session_id -> base_id`` 只覆盖指定会话（用于构造单会话超门）。
    """
    reports = []
    for session_id in SESSION_IDS:
        session_angles = angles_by_session[session_id]
        only_this_session = (
            None if per_session_overrides is None
            else per_session_overrides.get(session_id)
        )
        main_items = {}
        for base_id in BASE_IDS:
            if only_this_session is not None and base_id in only_this_session:
                angles = only_this_session[base_id]
            elif overrides is not None and base_id in overrides:
                angles = overrides[base_id]
            else:
                angles = list(session_angles)
            main_items[base_id] = _main_item(base_id, angles)
        reports.append({"session_id": session_id, "main_items": main_items})
    return reports


def test_pinned_ids_are_proper_subset_and_exclusions_match_registry() -> None:
    pinned = set(PINNED_IDS)
    assert len(pinned) == len(PINNED_IDS) == 15
    assert PINNED_IDS[:10] == (
        "grating_x_p128",
        "tilt_x_plus",
        "tilt_x_minus",
        "patch_x_plus_p128",
        "patch_x_minus_p128",
        "grating_y_p128",
        "tilt_y_plus",
        "tilt_y_minus",
        "patch_y_plus_p128",
        "patch_y_minus_p128",
    )
    assert pinned < set(BASE_IDS)
    assert set(BASE_IDS) - pinned == set(NON_PINNED_IDS)
    assert set(NON_PINNED_IDS) == {
        "patch_x_minus_p192",
        "tilt_y_plus_p192",
        "tilt_y_minus_p192",
        "patch_y_plus_p192",
        "patch_y_minus_p192",
    }
    # 预注册排除项必须全部来自 p192，且全部包含在补审范围里。
    assert all(SCRIPT.core_audit.P192_TOKEN in base for base in NON_PINNED_IDS)
    assert set(NON_PINNED_IDS) <= set(P192_IDS)
    # 结构校验本身必须能通过（漂移时抛 RejectedError）。
    SCRIPT._assert_pinned_scope_matches_registry()


def test_main_criterion_derives_missing_p192_subset_without_hardcoding() -> None:
    # 预注册 §3.1 的 5 个被排除 p192 缺测：pinned_15 仍全 reproduced，主判据应通过。
    missing = list(NON_PINNED_IDS)
    reports = _reports({"s1": [10.0] * 3, "s2": [10.0] * 3, "s3": [10.0] * 3})
    for report in reports:
        for base_id in missing:
            report["main_items"][base_id] = _main_item(base_id, [None] * 3)

    result = SCRIPT.main_criterion(reports)

    assert result["p192_not_assessable"] == missing
    assert result["counts"]["not_assessable"] == len(missing)
    assert result["counts"]["reproduced"] == len(PINNED_IDS)
    assert result["pinned_counts"]["reproduced"] == len(PINNED_IDS)
    assert result["pinned_counts"]["not_assessable"] == 0
    # 5 个非 pinned 缺测不计入主门，因此主判据可以通过。
    assert result["overall_pass"] is True
    assert result["verdict"] == "reproduced"
    assert result["contradiction_outside_gate"] is False


def test_main_criterion_uses_dynamic_p192_subset_not_frozen_list() -> None:
    # “5 个 p192 缺测”是结果而非冻结清单：让 pinned 的某个 p192 缺测，
    # 派生列表必须精确等于实际缺测项（而不是固定 5 个）。
    missing = ["grating_x_p192"]
    reports = _reports({"s1": [10.0] * 3, "s2": [10.0] * 3, "s3": [10.0] * 3})
    for report in reports:
        report["main_items"][missing[0]] = _main_item(missing[0], [None] * 3)

    result = SCRIPT.main_criterion(reports)

    assert result["p192_not_assessable"] == missing
    assert result["counts"]["not_assessable"] == 1
    # pinned 缺测 → 主门不可评估，不能用另外 14 个替代。
    assert result["overall_pass"] is False
    assert result["verdict"] == "not_assessable"
    assert result["contradiction_outside_gate"] is False


def test_main_criterion_pinned_missing_session_is_not_assessable() -> None:
    # pinned 中的一个图样缺 s3：另两个会话一致也不得判 reproduced。
    missing_session = "s3"
    reports = _reports({"s1": [10.0] * 3, "s2": [10.0] * 3, "s3": [10.0] * 3})
    for report in reports:
        if report["session_id"] == missing_session:
            report["main_items"][PINNED_IDS[1]] = _main_item(
                PINNED_IDS[1], [None] * 3
            )

    result = SCRIPT.main_criterion(reports)

    assert result["overall_pass"] is False
    assert result["verdict"] == "not_assessable"
    assert result["pinned_counts"]["not_assessable"] == 1
    assert result["p192_not_assessable"] == []


def test_main_criterion_pinned_angle_breach_is_not_reproduced() -> None:
    breach = {"s3": {PINNED_IDS[0]: [30.0] * 3}}
    reports = _reports(
        {"s1": [10.0] * 3, "s2": [10.0] * 3, "s3": [10.0] * 3},
        per_session_overrides=breach,
    )

    result = SCRIPT.main_criterion(reports)

    assert result["overall_pass"] is False
    assert result["verdict"] == "not_reproduced"
    assert result["pinned_counts"]["not_reproduced"] == 1
    assert result["contradiction_outside_gate"] is False


def test_main_criterion_non_pinned_contradiction_does_not_upgrade_failure() -> None:
    # 非 pinned p192 本轮反而可评估且方向超门：记为“与 11-24 相反的新证据”，
    # 但不改写 pinned_15 主判据结论。
    breach = {"s3": {NON_PINNED_IDS[0]: [40.0] * 3}}
    reports = _reports(
        {"s1": [10.0] * 3, "s2": [10.0] * 3, "s3": [10.0] * 3},
        per_session_overrides=breach,
    )

    result = SCRIPT.main_criterion(reports)

    assert result["overall_pass"] is True
    assert result["verdict"] == "reproduced"
    assert result["contradiction_outside_gate"] is True
    assert result["contradiction_outside_gate_ids"] == [NON_PINNED_IDS[0]]
    assert result["counts"]["not_reproduced"] == 1
    assert result["pinned_counts"]["not_reproduced"] == 0
    assert result["p192_not_assessable"] == []
    entry = next(
        item
        for item in result["non_pinned_patterns"]
        if item["base_id"] == NON_PINNED_IDS[0]
    )
    assert entry["status"] == "not_reproduced"
    assert "不计入 pinned_15" in entry["gate_scope_excludes"]


def test_main_criterion_non_pinned_patterns_list_all_five() -> None:
    reports = _reports({"s1": [10.0] * 3, "s2": [10.0] * 3, "s3": [10.0] * 3})
    for report in reports:
        for base_id in NON_PINNED_IDS:
            report["main_items"][base_id] = _main_item(base_id, [None] * 3)

    result = SCRIPT.main_criterion(reports)

    assert [item["base_id"] for item in result["non_pinned_patterns"]] == list(
        NON_PINNED_IDS
    )
    assert all(
        item["status"] == "not_assessable"
        for item in result["non_pinned_patterns"]
    )
    assert result["contradiction_outside_gate"] is False
    gate = result["gate"]
    assert gate["scope"] == "pinned_15"
    assert gate["n_patterns"] == 15
    assert gate["n_required"] == 15
    assert gate["excluded_p192"] == list(NON_PINNED_IDS)


# ---------------------------------------------------------------------------
# p192 附加审计
# ---------------------------------------------------------------------------
def test_supplemental_three_of_four_strong_frames_is_assessable() -> None:
    items = _audit_items_by_session(
        {
            "s1": [10.0, 10.4, 9.6, 10.0],
            "s2": [10.2, 9.8, 10.1, 10.0],
            "s3": [10.1, 10.0, 9.9, 10.0],
        },
        strong=[True, True, True, False],
    )
    result = SCRIPT.supplemental_pattern_verdict(items, "demo_p192")

    assert result["verdict"] == "reproduced"
    assert all(
        session["n_strong_frames_within_core"] == 3
        for session in result["sessions"].values()
    )
    assert result["max_pairwise_diff_deg"] <= 2.0


def test_supplemental_two_of_four_strong_frames_is_not_assessable() -> None:
    items = _audit_items_by_session(
        {"s1": [10.0] * 4, "s2": [10.0] * 4, "s3": [10.0] * 4},
        strong=[True, True, False, False],
    )
    result = SCRIPT.supplemental_pattern_verdict(items, "demo_p192")

    assert result["verdict"] == "not_assessable"
    assert all(
        session["n_strong_frames_within_core"] == 2
        for session in result["sessions"].values()
    )
    assert result["sessions"]["s1"]["reason"] is not None


def test_supplemental_three_assessable_sessions_over_gate_is_not_reproduced() -> None:
    items = _audit_items_by_session(
        {"s1": [10.0, 10.0, 10.0], "s2": [10.0, 10.0, 10.0], "s3": [40.0] * 3}
    )
    result = SCRIPT.supplemental_pattern_verdict(items, "demo_p192")

    assert result["verdict"] == "not_reproduced"
    assert result["max_pairwise_diff_deg"] == pytest.approx(30.0)


def test_supplemental_verdict_never_overwrites_main_p192_verdict() -> None:
    main_result = SCRIPT.main_pattern_verdict(
        _items_by_session(
            {"s1": [None] * 3, "s2": [None] * 3, "s3": [None] * 3},
            base_id="demo_p192",
        ),
        "demo_p192",
    )
    supplemental_result = SCRIPT.supplemental_pattern_verdict(
        _audit_items_by_session(
            {"s1": [10.0] * 3, "s2": [10.0] * 3, "s3": [10.0] * 3}
        ),
        "demo_p192",
    )

    assert main_result["verdict"] == "not_assessable"
    assert supplemental_result["verdict"] == "reproduced"
    assert main_result["verdict"] != supplemental_result["verdict"]


# ---------------------------------------------------------------------------
# 清单与 CLI
# ---------------------------------------------------------------------------
def _plan() -> dict[str, Any]:
    return {
        "schema": "issue247-reproduction-plan/v1",
        "plan_name": "reproduction",
        "pattern_ids": ["demo"],
        "frames_per_pattern": 4,
        "n_frames_expected": 4,
    }


def _digest() -> str:
    return SCRIPT.capture_manifest.plan_digest(_plan())


def _write_root(
    root: Path, *, mutate: Any = None, session_ids: tuple[str, ...] = SESSION_IDS
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    plan = _plan()
    digest = _digest()
    sessions: list[dict[str, Any]] = []
    for session_id in session_ids:
        out = root / session_id / "out"
        probe = root / session_id / "probe"
        out.mkdir(parents=True)
        probe.mkdir(parents=True)
        (out / "plan.json").write_text(
            json.dumps(plan, ensure_ascii=False), encoding="utf-8"
        )
        sessions.append(
            {
                "session_id": session_id,
                "status": "ok",
                "exit_code": 0,
                "artifact_problems": [],
                "plan_sha256": digest,
                "out": str(out),
                "probe_dir": str(probe),
            }
        )
    manifest: dict[str, Any] = {
        "schema": SCRIPT.MANIFEST_SCHEMA,
        "dry_run": False,
        "stopped_early": False,
        "completed_sessions": 3,
        "capture_settings": {"exposure_ms": 0.04, "settle_s": 0.8},
        "plan_sha256": digest,
        "sessions": sessions,
    }
    if mutate is not None:
        mutate(manifest, root)
    (root / SCRIPT.MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return root


def _missing_session(manifest: dict[str, Any], root: Path) -> None:
    del root
    manifest["sessions"].pop()


def _duplicate_out(manifest: dict[str, Any], root: Path) -> None:
    del root
    manifest["sessions"][1]["out"] = manifest["sessions"][0]["out"]


def _stopped_early(manifest: dict[str, Any], root: Path) -> None:
    del root
    manifest["stopped_early"] = True


def _dry_run(manifest: dict[str, Any], root: Path) -> None:
    del root
    manifest["dry_run"] = True


def _plan_sha_mismatch(manifest: dict[str, Any], root: Path) -> None:
    del root
    manifest["sessions"][1]["plan_sha256"] = "0" * 64


def _exposure_mismatch(manifest: dict[str, Any], root: Path) -> None:
    del root
    manifest["capture_settings"]["exposure_ms"] = 0.05


@pytest.mark.parametrize(
    "mutate",
    [
        _missing_session,
        _duplicate_out,
        _stopped_early,
        _dry_run,
        _plan_sha_mismatch,
        _exposure_mismatch,
    ],
    ids=[
        "missing_session",
        "duplicate_out",
        "stopped_early",
        "dry_run",
        "plan_sha_mismatch",
        "exposure_mismatch",
    ],
)
def test_invalid_manifest_exits_3_and_writes_nothing(
    tmp_path: Path, mutate: Any
) -> None:
    root = _write_root(tmp_path / "root", mutate=mutate)
    out_dir = tmp_path / "out"

    code = SCRIPT.main(["--root", str(root), "--out-dir", str(out_dir)])

    assert code == 3
    assert not out_dir.exists()


def test_failed_session_check_exits_3_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _write_root(tmp_path / "root")

    def failing_check(
        capture_dir: Path, baseline_dir: Path | None, probe_dir: Path
    ) -> dict[str, Any]:
        del baseline_dir, probe_dir
        raise SCRIPT.repeatability_check.CheckError(f"{capture_dir}: 帧数不足")

    def unused_audit(capture_dir: Path) -> dict[str, Any]:
        raise AssertionError(f"检查已失败时不应进入补审：{capture_dir}")

    monkeypatch.setattr(SCRIPT, "_RUN_CHECK", failing_check)
    monkeypatch.setattr(SCRIPT, "_RUN_AUDIT", unused_audit)
    out_dir = tmp_path / "out"

    code = SCRIPT.main(["--root", str(root), "--out-dir", str(out_dir)])

    # 不得静默跳过失败会话，也不得写任何产物。
    assert code == 3
    assert not out_dir.exists()


def test_session_plan_digest_mismatch_exits_3_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _write_root(tmp_path / "root")
    _install_fakes(
        monkeypatch,
        angles_by_session={"s1": 10.0, "s2": 10.0, "s3": 10.0},
        p192_assessable=True,
    )
    plan_path = root / "s2" / "out" / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["plan_name"] = "tampered"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    out_dir = tmp_path / "out"

    code = SCRIPT.main(["--root", str(root), "--out-dir", str(out_dir)])

    assert code == 3
    assert not out_dir.exists()


def _fake_check_payload(
    session_angles: float,
    p192_assessable: bool,
    overrides: dict[str, list[float | None]] | None = None,
    per_session_overrides: dict[str, list[float | None]] | None = None,
) -> dict[str, Any]:
    """合成单个会话的 ``check_capture`` 结果。

    ``overrides`` 按 base_id 覆盖所有会话；``per_session_overrides`` 只覆盖当前
    会话，用于构造“仅某会话角度超门”的场景。
    """
    patterns = []
    for base_id in BASE_IDS:
        angles: list[float | None]
        if per_session_overrides is not None and base_id in per_session_overrides:
            angles = per_session_overrides[base_id]
        elif overrides is not None and base_id in overrides:
            angles = overrides[base_id]
        elif base_id in NON_PINNED_IDS and not p192_assessable:
            angles = [None, None, None]
        else:
            angles = [session_angles] * 3
        patterns.append(_main_item(base_id, angles))
    return {
        "repeatability": {
            "patterns": patterns,
            "counts": {"reproduced": 0, "not_reproduced": 0, "not_assessable": 0},
            "n_evaluable_patterns": len(BASE_IDS),
            "overall_pass": True,
        }
    }


def _fake_audit_payload(
    angle: float = 10.0, strong: list[bool] | None = None
) -> dict[str, Any]:
    return {
        "supplemental_audit": {
            "patterns": [
                _audit_item(base_id, [angle] * 4, strong=strong)
                for base_id in P192_IDS
            ]
        }
    }


def _snapshot(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    angles_by_session: dict[str, float],
    p192_assessable: bool,
    overrides: dict[str, list[float | None]] | None = None,
    audit_angles_by_session: dict[str, float] | None = None,
    audit_strong: list[bool] | None = None,
    breach: dict[str, dict[str, list[float | None]]] | None = None,
) -> None:
    def fake_check(
        capture_dir: Path, baseline_dir: Path | None, probe_dir: Path
    ) -> dict[str, Any]:
        del baseline_dir, probe_dir
        session_id = Path(capture_dir).parent.name
        per_session = None if breach is None else breach.get(session_id)
        return _fake_check_payload(
            angles_by_session[session_id], p192_assessable, overrides, per_session
        )

    def fake_audit(capture_dir: Path) -> dict[str, Any]:
        session_id = Path(capture_dir).parent.name
        angle = (
            10.0
            if audit_angles_by_session is None
            else audit_angles_by_session[session_id]
        )
        return _fake_audit_payload(angle, audit_strong)

    monkeypatch.setattr(SCRIPT, "_RUN_CHECK", fake_check)
    monkeypatch.setattr(SCRIPT, "_RUN_AUDIT", fake_audit)


def test_fake_run_check_writes_only_out_dir_and_leaves_inputs_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _write_root(tmp_path / "root")
    before = _snapshot(root)
    _install_fakes(
        monkeypatch,
        angles_by_session={"s1": 10.0, "s2": 10.0, "s3": 10.0},
        p192_assessable=True,
    )
    out_dir = tmp_path / "out"

    code = SCRIPT.main(["--root", str(root), "--out-dir", str(out_dir)])

    assert code == 0
    assert _snapshot(root) == before
    assert sorted(path.name for path in out_dir.iterdir()) == sorted(
        [SCRIPT.DEFAULT_JSON_NAME, SCRIPT.DEFAULT_MD_NAME]
    )
    payload = json.loads(
        (out_dir / SCRIPT.DEFAULT_JSON_NAME).read_text(encoding="utf-8")
    )
    assert payload["schema"] == SCRIPT.CHECK_SCHEMA
    assert payload["manifest"]["sha256"] == hashlib.sha256(
        (root / SCRIPT.MANIFEST_NAME).read_bytes()
    ).hexdigest()
    assert payload["main"]["counts"] == {
        "not_assessable": 0,
        "not_reproduced": 0,
        "reproduced": len(BASE_IDS),
    }
    assert payload["supplemental"]["verdict"] == "reproduced"
    assert payload["frozen_main_verdict_scope"]
    assert payload["supplemental_audit_scope"]
    assert payload["conclusion_boundaries"]


def test_pinned_pass_with_excluded_p192_missing_exits_0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 预注册 §3.1 的核心场景：pinned_15 全 reproduced，5 个被排除 p192 缺测，
    # 补审 reproduced → 主判据 pass、退出码 0，缺测不得伪装成失败。
    root = _write_root(tmp_path / "root")
    _install_fakes(
        monkeypatch,
        angles_by_session={"s1": 10.0, "s2": 10.0, "s3": 10.0},
        p192_assessable=False,
    )

    code = SCRIPT.main(
        ["--root", str(root), "--out-dir", str(tmp_path / "out")]
    )

    assert code == 0
    payload = json.loads(
        (tmp_path / "out" / SCRIPT.DEFAULT_JSON_NAME).read_text(encoding="utf-8")
    )
    main_block = payload["main"]
    main_by_id = {item["base_id"]: item for item in main_block["patterns"]}
    for base_id in NON_PINNED_IDS:
        assert main_by_id[base_id]["verdict"] == "not_assessable"
    for base_id in PINNED_IDS:
        assert main_by_id[base_id]["verdict"] == "reproduced"
    assert main_block["overall_pass"] is True
    assert main_block["verdict"] == "reproduced"
    assert main_block["pinned_counts"]["reproduced"] == 15
    assert main_block["p192_not_assessable"] == list(NON_PINNED_IDS)
    assert len(main_block["non_pinned_patterns"]) == 5
    assert main_block["contradiction_outside_gate"] is False
    assert main_block["gate"]["scope"] == "pinned_15"
    assert payload["supplemental"]["verdict"] == "reproduced"


def test_pinned_angle_breach_exits_1_with_not_reproduced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 15 个 pinned 里恰好 1 个在 s3 方向超门：主判据 not_reproduced，补审仍 reproduced，
    # 总退出码 1（主判据负责这一票）。
    root = _write_root(tmp_path / "root")
    _install_fakes(
        monkeypatch,
        angles_by_session={"s1": 10.0, "s2": 10.0, "s3": 10.0},
        p192_assessable=False,
        breach={"s3": {PINNED_IDS[0]: [40.0] * 3}},
    )

    code = SCRIPT.main(
        ["--root", str(root), "--out-dir", str(tmp_path / "out")]
    )

    assert code == 1
    payload = json.loads(
        (tmp_path / "out" / SCRIPT.DEFAULT_JSON_NAME).read_text(encoding="utf-8")
    )
    assert payload["main"]["overall_pass"] is False
    assert payload["main"]["verdict"] == "not_reproduced"
    assert payload["main"]["pinned_counts"]["not_reproduced"] == 1
    assert payload["supplemental"]["verdict"] == "reproduced"


def test_main_pass_but_supplemental_failure_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 主判据全过但补审未复现：退出码 1，且两个 verdict 仍分列、不合并。
    root = _write_root(tmp_path / "root")
    _install_fakes(
        monkeypatch,
        angles_by_session={"s1": 10.0, "s2": 10.0, "s3": 10.0},
        p192_assessable=False,
        audit_angles_by_session={"s1": 10.0, "s2": 10.0, "s3": 40.0},
    )

    code = SCRIPT.main(
        ["--root", str(root), "--out-dir", str(tmp_path / "out")]
    )

    assert code == 1
    payload = json.loads(
        (tmp_path / "out" / SCRIPT.DEFAULT_JSON_NAME).read_text(encoding="utf-8")
    )
    assert payload["main"]["overall_pass"] is True
    assert payload["main"]["verdict"] == "reproduced"
    assert payload["supplemental"]["verdict"] == "not_reproduced"


def test_pinned_missing_session_exits_1_with_not_assessable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _write_root(tmp_path / "root")
    _install_fakes(
        monkeypatch,
        angles_by_session={"s1": 10.0, "s2": 10.0, "s3": 10.0},
        p192_assessable=False,
        overrides={PINNED_IDS[2]: [None, None, None]},
    )

    code = SCRIPT.main(
        ["--root", str(root), "--out-dir", str(tmp_path / "out")]
    )

    assert code == 1
    payload = json.loads(
        (tmp_path / "out" / SCRIPT.DEFAULT_JSON_NAME).read_text(encoding="utf-8")
    )
    main_block = payload["main"]
    assert main_block["overall_pass"] is False
    assert main_block["verdict"] == "not_assessable"
    assert main_block["pinned_counts"]["not_assessable"] == 1


def test_non_pinned_contradiction_flags_but_keeps_exit_0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 非 pinned p192 本轮可评估但方向超门：记为“与 11-24 相反的新证据”，
    # 不得升级为主判据失败，退出码仍由 pinned_15 + 补审决定。
    root = _write_root(tmp_path / "root")
    _install_fakes(
        monkeypatch,
        angles_by_session={"s1": 10.0, "s2": 10.0, "s3": 10.0},
        p192_assessable=True,
        breach={"s3": {NON_PINNED_IDS[0]: [40.0] * 3}},
    )

    code = SCRIPT.main(
        ["--root", str(root), "--out-dir", str(tmp_path / "out")]
    )

    assert code == 0
    payload = json.loads(
        (tmp_path / "out" / SCRIPT.DEFAULT_JSON_NAME).read_text(encoding="utf-8")
    )
    main_block = payload["main"]
    assert main_block["overall_pass"] is True
    assert main_block["verdict"] == "reproduced"
    assert main_block["contradiction_outside_gate"] is True
    assert main_block["contradiction_outside_gate_ids"] == [NON_PINNED_IDS[0]]
    assert main_block["pinned_counts"]["not_reproduced"] == 0
    markdown = (
        tmp_path / "out" / SCRIPT.DEFAULT_MD_NAME
    ).read_text(encoding="utf-8")
    assert "与 11-24 相反的新证据" in markdown
    assert "pinned_15" in markdown


def test_existing_output_is_refused_with_exit_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _write_root(tmp_path / "root")
    _install_fakes(
        monkeypatch,
        angles_by_session={"s1": 10.0, "s2": 10.0, "s3": 10.0},
        p192_assessable=True,
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / SCRIPT.DEFAULT_JSON_NAME).write_text("{}", encoding="utf-8")

    code = SCRIPT.main(["--root", str(root), "--out-dir", str(out_dir)])

    assert code == 4
    assert (out_dir / SCRIPT.DEFAULT_JSON_NAME).read_text(encoding="utf-8") == "{}"


def test_out_dir_inside_capture_root_is_refused_with_exit_4(
    tmp_path: Path,
) -> None:
    root = _write_root(tmp_path / "root")

    code = SCRIPT.main(
        ["--root", str(root), "--out-dir", str(root / "fresh_out")]
    )

    assert code == 4
    assert not (root / "fresh_out").exists()


def test_out_dir_that_contains_inputs_is_refused_with_exit_4(
    tmp_path: Path,
) -> None:
    root = _write_root(tmp_path / "root")

    code = SCRIPT.main(["--root", str(root), "--out-dir", str(tmp_path)])

    assert code == 4
    assert not (tmp_path / SCRIPT.DEFAULT_JSON_NAME).exists()
    assert not (tmp_path / SCRIPT.DEFAULT_MD_NAME).exists()


def test_verdict_values_are_limited_to_the_frozen_domain() -> None:
    result = SCRIPT.main_pattern_verdict(
        _items_by_session(
            {"s1": [0.0, 0.0], "s2": [0.0, 0.0], "s3": [5.0, 5.0]}
        ),
        "demo",
    )
    supplemental = SCRIPT.supplemental_pattern_verdict(
        _audit_items_by_session(
            {"s1": [0.0] * 3, "s2": [0.0] * 3, "s3": [0.0] * 3}
        ),
        "demo_p192",
    )

    assert result["verdict"] in SCRIPT.VERDICTS
    assert supplemental["verdict"] in SCRIPT.VERDICTS
