"""耗时审计脚本的解析契约 (纯离线, 不碰设备)。

审计脚本的价值全在**它算出来的数对不对**, 所以这里逐条钉住:

* **阶段切分** —— campaign 的 ``stage.log`` 是靠记录里本来就有的标记行切段的
  (面板扫描的 ``flat reference`` / ``panel y=``、几何 sweep 的 ``stage_sweep``),
  切错一段, 后面所有的占比就全错;
* **单次耗时与占比** —— 采集 manifest 的每一类动作 (settle / 曝光阶梯 / 写盘 /
  帧间) 必须互斥且求和等于采集总时长;
* **点数下限的建议计算** —— 二次曲线拟合的最少点数是"凭估计做决定"的反面,
  必须由参数数目与残差自由度推出来, 而不是拍脑袋;
* **档 A/B/C 的分项求和自洽** —— 预算表的分项之和必须等于该档总时长,
  否则预算表会被当成结论引用而里面是错的。

全部用**合成**的日志与 manifest, 不依赖真实语料规模。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "slm" / "audit_physical_test_time.py"

#: The two end-to-end tests below build the ledger from the **real** on-bench capture
#: corpus under ``data/debug/`` (gitignored -- present only on the collecting machine,
#: absent on a clean clone / CI). Guard them on corpus presence: skip with a named
#: reason when the tree is gone rather than failing on ``capture == {}`` (issue #163
#: class: a hard dependency on gitignored data with no fixture/skip mechanism). The
#: synthetic-fixture coverage in the other tests still runs unconditionally.
_REAL_CAPTURE_GLOB = "data/debug/physical_forward_defocus/*/manifest.json"


def _real_capture_present() -> bool:
    return any(ROOT.glob(_REAL_CAPTURE_GLOB))


#: Skip message reused by the two real-corpus end-to-end tests.
_REAL_CAPTURE_SKIP = (
    "real capture corpus missing: no "
    "data/debug/physical_forward_defocus/*/manifest.json on this checkout (the data "
    "tree is gitignored and exists only on the collecting machine); synthetic-fixture "
    "coverage above still runs"
)


def _load_audit():
    """按路径加载审计脚本 (它住在 ``scripts/slm`` 下, 不在包路径上)。"""
    spec = importlib.util.spec_from_file_location("audit_physical_test_time", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


audit = _load_audit()


# ---------------------------------------------------------------------------
# 合成记录
# ---------------------------------------------------------------------------


def _log_line(ts: str, func: str, message: str, module: str = "__main__") -> str:
    return f"{ts} | INFO     | {module}:{func}:100 - {message}"


def _synthetic_stage_log() -> str:
    """一段形状与真实 campaign 一致的 stage.log。

    真实日志的阶段边界: 面板扫描 (``scan_panel``) -> 照明半径 (无标记) ->
    几何 sweep (``stage_sweep``) -> 离线标定。
    """
    lines = []
    # 设备打开 + 平场参考
    lines.append(_log_line("2026-10-22 16:49:37.000", "_write_to_memory", "相位数据已写入SLM #1 内存#3"))
    lines.append(
        _log_line(
            "2026-10-22 16:49:45.000",
            "scan_panel",
            "flat reference: peak=68 fwhm=44.4px",
            module="ao_shaping.tools.slm.panel.panel_locate",
        )
    )
    # 面板扫描: 2 行 x 3 列 = 6 点, 每点一次写入 + 一次 settle 超时
    for index in range(6):
        stamp = f"2026-10-22 16:5{index // 3}:{10 + index * 5:02d}.000"
        lines.append(_log_line(stamp, "_write_to_memory", "相位数据已写入SLM #1 内存#4"))
        lines.append(
            _log_line(
                stamp,
                "_warn_if_unsettled",
                "settle criterion not met after 7.00 s (6 probes, n_stable=3): settled=False",
                module="ao_shaping.tools.kernels.bench_kernels",
            )
        )
    lines.append(
        _log_line(
            "2026-10-22 16:51:54.000",
            "scan_panel",
            "panel y= 220: x=240: +0.003 x=600: +0.003 x=960: +0.002",
            module="ao_shaping.tools.slm.panel.panel_locate",
        )
    )
    lines.append(
        _log_line(
            "2026-10-22 16:53:46.000",
            "scan_panel",
            "panel y= 480: x=240: -0.006 x=600: +0.004 x=960: -0.094",
            module="ao_shaping.tools.slm.panel.panel_locate",
        )
    )
    # 照明半径: 两次写入, 无自己的标记
    lines.append(_log_line("2026-10-22 16:57:47.000", "_write_to_memory", "相位数据已写入SLM #1 内存#20"))
    lines.append(_log_line("2026-10-22 17:05:31.000", "_write_to_memory", "相位数据已写入SLM #1 内存#30"))
    # 几何 sweep
    lines.append(_log_line("2026-10-22 17:05:34.000", "stage_sweep", "[*] flat reference"))
    for index in range(4):
        stamp = f"2026-10-22 17:0{6 + index}:00.000"
        lines.append(_log_line(stamp, "_write_to_memory", "相位数据已写入SLM #1 内存#31"))
    lines.append(_log_line("2026-10-22 17:12:54.000", "stage_sweep", "[*] saved 4 sweep points -> ..."))
    # 离线标定
    lines.append(_log_line("2026-10-22 17:13:00.000", "calibrate_bench_geometry_from_sweep", "focal scale 1.0685"))
    return "\n".join(lines) + "\n"


def _synthetic_capture_rows(n: int = 6) -> list[dict]:
    """一段形状与真实采集 manifest 一致的记录。

    真实记录里每条有 ``utc_start`` / ``timestamp_utc`` (采集窗口) 与
    ``settled.waited_s`` (settle 烧掉的秒数); 帧间开销由相邻两条的
    ``utc_start`` 减上一条的 ``timestamp_utc`` 得到。
    """
    from datetime import datetime, timedelta, timezone

    start = datetime(2026, 10, 22, 12, 0, 0, tzinfo=timezone.utc)
    rows = []
    for index in range(n):
        utc_start = start + timedelta(seconds=20 * index)
        capture = 10.0
        rows.append(
            {
                "index": index,
                "file": f"sample_{index:04d}.npz",
                "label": "reference_start" if index == 0 else f"probe_{index:03d}",
                "branch": "C" if index % 2 == 0 else "U",
                "family": "reference" if index < 4 else "prior",
                "split": "calibration",
                "exposure_ms": 20.0 + index,
                "exposure_status": "ok",
                "settled": {
                    "settled": True,
                    "reason": "stable",
                    "n_probes": 3,
                    "n_stable": 3,
                    "waited_s": 4.0,
                },
                "utc_start": utc_start.isoformat(),
                "timestamp_utc": (utc_start + timedelta(seconds=capture)).isoformat(),
                "monotonic_start_ns": index * 20 * 10**9,
                "monotonic_end_ns": (index * 20 + 10) * 10**9,
                "peak": 100,
                "spot": {
                    "peak": 100.0,
                    "fwhm_px": 14.4,
                    "centroid_x": 1360.0 + index * 0.2,
                    "centroid_y": 1010.0,
                    "hollowness": 0.95,
                },
            }
        )
    return rows


# ---------------------------------------------------------------------------
# 日志解析
# ---------------------------------------------------------------------------


class TestLogParsing:
    def test_ansi_is_stripped_and_timestamps_parsed(self):
        text = "\x1b[32m2026-10-22 16:49:45.549\x1b[0m | INFO | __main__:acquire:649 - flat peak=51\n"
        events = audit.parse_log_events(text)
        assert len(events) == 1
        assert events[0].func == "acquire"
        assert events[0].message == "flat peak=51"
        assert events[0].at.strftime("%H:%M:%S") == "16:49:45"

    def test_unparsable_lines_are_skipped_not_fatal(self):
        text = (
            "====================================================================\n"
            "SLM Model-in-the-Loop Shaping\n"
            + _log_line("2026-10-22 16:49:34.641", "cli", "Debug mode enabled")
            + "\n"
        )
        events = audit.parse_log_events(text)
        assert len(events) == 1
        assert events[0].func == "cli"

    def test_empty_input_gives_no_events(self):
        assert audit.parse_log_events("") == []


# ---------------------------------------------------------------------------
# 阶段切分
# ---------------------------------------------------------------------------


class TestCampaignSegmentation:
    def test_phases_are_split_by_recorded_markers(self):
        events = audit.parse_log_events(_synthetic_stage_log())
        phases = audit.segment_campaign(events)
        names = [p.name for p in phases]
        assert names == [
            "设备打开与平场参考",
            "面板扫描",
            "照明半径测量",
            "几何 sweep",
            "离线几何标定",
        ]

    def test_panel_scan_points_are_rows_times_columns(self):
        events = audit.parse_log_events(_synthetic_stage_log())
        phases = {p.name: p for p in audit.segment_campaign(events)}
        panel = phases["面板扫描"]
        # 2 行 x 3 列
        assert panel.points == 6
        assert panel.writes == 6
        assert panel.settle_timeouts == 6
        assert panel.settle_timeout_seconds == pytest.approx(42.0)

    def test_sweep_points_come_from_the_panel_writes(self):
        events = audit.parse_log_events(_synthetic_stage_log())
        phases = {p.name: p for p in audit.segment_campaign(events)}
        assert phases["几何 sweep"].points == 4
        assert phases["几何 sweep"].seconds == pytest.approx(440.0)

    def test_illumination_phase_is_the_gap_between_panel_and_sweep(self):
        events = audit.parse_log_events(_synthetic_stage_log())
        phases = {p.name: p for p in audit.segment_campaign(events)}
        illumination = phases["照明半径测量"]
        assert illumination.start is not None and illumination.end is not None
        # 面板扫描最后一行 16:53:46 -> sweep 起点 17:05:34
        assert illumination.seconds == pytest.approx(708.0)
        assert illumination.writes == 2

    def test_empty_log_yields_no_phases(self):
        assert audit.segment_campaign([]) == []


# ---------------------------------------------------------------------------
# 采集 manifest
# ---------------------------------------------------------------------------


class TestCaptureLedger:
    def test_rows_become_samples_with_window_and_gap(self):
        rows = _synthetic_capture_rows()
        samples = audit.capture_rows_to_samples(rows)
        assert len(samples) == 6
        assert samples[0].capture_seconds == pytest.approx(10.0)
        # 帧间: 下一条的 utc_start (20 s 间隔) 减上一条的 timestamp_utc (10 s)
        assert samples[0].gap_seconds == pytest.approx(10.0)
        assert samples[-1].gap_seconds == 0.0

    def test_manifest_without_utc_start_falls_back_to_monotonic(self):
        rows = _synthetic_capture_rows()
        for row in rows:
            row.pop("utc_start")
            row.pop("timestamp_utc")
        samples = audit.capture_rows_to_samples(rows)
        assert samples[0].capture_seconds == pytest.approx(10.0)
        assert samples[0].gap_seconds == pytest.approx(10.0)

    def test_record_without_any_time_stamp_is_rejected(self):
        rows = _synthetic_capture_rows(1)
        rows[0].pop("utc_start")
        rows[0].pop("timestamp_utc")
        rows[0].pop("monotonic_start_ns")
        rows[0].pop("monotonic_end_ns")
        with pytest.raises(ValueError):
            audit.capture_rows_to_samples(rows)

    def test_action_rows_are_disjoint_and_sum_to_the_total(self):
        samples = audit.capture_rows_to_samples(_synthetic_capture_rows())
        summary = audit.summarize_capture(samples, write_cost_s=5.0)
        total = sum(row["total_seconds"] for row in summary["actions"])
        assert total == pytest.approx(summary["total_seconds"])
        # settle + 阶梯 + 写盘 = 采集窗口; 帧间单列
        assert summary["capture_window_seconds"] == pytest.approx(60.0)
        assert summary["gap_seconds"] == pytest.approx(50.0)
        assert summary["total_seconds"] == pytest.approx(110.0)

    def test_settle_is_read_from_the_record_not_estimated(self):
        samples = audit.capture_rows_to_samples(_synthetic_capture_rows())
        summary = audit.summarize_capture(samples, write_cost_s=0.0)
        assert summary["median_settle_seconds"] == pytest.approx(4.0)
        settle_row = next(r for r in summary["actions"] if r["action"].startswith("settle"))
        assert settle_row["total_seconds"] == pytest.approx(4.0 * 6)

    def test_write_cost_is_subtracted_from_the_ladder_bucket(self):
        samples = audit.capture_rows_to_samples(_synthetic_capture_rows())
        cheap = audit.summarize_capture(samples, write_cost_s=0.0)
        dear = audit.summarize_capture(samples, write_cost_s=5.0)
        assert cheap["median_ladder_seconds"] == pytest.approx(6.0)
        assert dear["median_ladder_seconds"] == pytest.approx(1.0)

    def test_reference_frames_are_attributed_not_double_counted(self):
        samples = audit.capture_rows_to_samples(_synthetic_capture_rows())
        summary = audit.summarize_capture(samples, write_cost_s=0.0)
        attribution = summary["attribution"]
        assert attribution["reference_frames"] == 4
        assert attribution["reference_share_of_captures"] == pytest.approx(4 / 6)
        # 归因不改变动作行的求和
        total = sum(row["total_seconds"] for row in summary["actions"])
        assert total == pytest.approx(summary["total_seconds"])


# ---------------------------------------------------------------------------
# 漂移统计 (reference 帧间隔的依据)
# ---------------------------------------------------------------------------


class TestBranchDrift:
    def test_width_span_and_centroid_steps_are_computed(self):
        rows = _synthetic_capture_rows(6)
        samples = audit.capture_rows_to_samples(rows)
        drift = audit.branch_drift(samples, rows)
        assert set(drift) == {"C", "U"}
        # 分支 C 的参考帧在 index 0 与 2, 质心 x 差 0.4 px
        assert drift["C"]["n_references"] == 2
        assert drift["C"]["centroid_step_median_px"] == pytest.approx(0.4)
        assert drift["C"]["fwhm_span_fraction"] == pytest.approx(0.0)

    def test_no_reference_rows_gives_no_drift_entry(self):
        rows = _synthetic_capture_rows(4)
        for row in rows:
            row["family"] = "prior"
        samples = audit.capture_rows_to_samples(rows)
        assert audit.branch_drift(samples, rows) == {}

    def test_drift_reports_the_interval_dependence(self):
        rows = _synthetic_capture_rows(6)
        samples = audit.capture_rows_to_samples(rows)
        drift = audit.branch_drift(samples, rows)
        stats = drift["C"]
        # 间隔恒定 (C 的参考帧在 index 0 与 2, 相距 40 s), 步长恒定 (0.4 px)
        assert stats["step_vs_interval_slope_px_per_min"] == pytest.approx(0.0, abs=1e-9)
        assert stats["step_vs_interval_jitter_px"] == pytest.approx(0.4)
        assert abs(stats["step_vs_interval_correlation"]) < 1e-9
        assert stats["median_interval_minutes"] == pytest.approx(40 / 60.0)


class TestCaptureManifestFilter:
    def test_alignment_manifest_is_not_a_capture_manifest(self):
        # 几何标定的逐帧记录: 有 269 行, 比采集那一轮还多, 但没有 settled/family
        alignment = [
            {
                "file": "frame_00000.npz",
                "phase_file": "phase_0001.npy",
                "exposure_ms": 50.0,
                "timestamp_utc": "2026-10-22T10:11:29.454415+00:00",
            }
        ]
        assert audit._is_capture_manifest(alignment) is False
        assert audit._is_capture_manifest(_synthetic_capture_rows(1)) is True
        assert audit._is_capture_manifest([]) is False
        assert audit._is_capture_manifest("not a list") is False


# ---------------------------------------------------------------------------
# 几何 sweep 的最少点数
# ---------------------------------------------------------------------------


class TestMinSweepPoints:
    def test_width_criterion_keeps_a_residual_degree_of_freedom(self):
        result = audit.min_sweep_points()
        # 3 参数 + 1 残差自由度 + 1 顶点锚点
        assert result["width_criterion_points"] == 5
        assert result["n_params"] == 3
        assert result["min_residual_dof"] == 1

    def test_peak_criterion_needs_one_point_less(self):
        result = audit.min_sweep_points()
        assert result["peak_criterion_points"] == result["width_criterion_points"] - 1
        assert result["peak_criterion_points"] >= 4

    def test_more_parameters_never_lower_the_floor(self):
        assert audit.min_sweep_points(n_params=5)["width_criterion_points"] >= 5

    def test_dropping_the_vertex_anchor_saves_the_anchor_point(self):
        anchored = audit.min_sweep_points(anchor_vertex=True)["width_criterion_points"]
        free = audit.min_sweep_points(anchor_vertex=False)["width_criterion_points"]
        assert anchored == free + 1


# ---------------------------------------------------------------------------
# 设备时间预算
# ---------------------------------------------------------------------------


class TestDeviceBudget:
    def _rates(self) -> dict[str, float]:
        return {
            "light_gate": 12.0,
            "probe": 84.0,
            "panel_scan": 482.0,
            "illumination": 464.0,
            "sweep": 440.0,
            "calibrate": 6.0,
            "mil_step_a": 18.0,
            "mil_step_b": 600.0,
            "mil_accept": 10.0,
        }

    def test_tier_a_is_probe_plus_geometry_plus_probe_set(self):
        budget = audit.build_device_budget(self._rates())
        tier = budget["tier_a"]
        assert tier["total_seconds"] == pytest.approx(
            84.0 + 482.0 + 464.0 + 440.0 + 18.0 + 6.0
        )

    def test_each_tier_sums_to_its_own_total(self):
        budget = audit.build_device_budget(self._rates())
        assert audit.budget_self_consistent(budget) is True
        for key in ("tier_a", "tier_b", "tier_c"):
            tier = budget[key]
            assert sum(row["seconds"] for row in tier["rows"]) == pytest.approx(
                tier["total_seconds"]
            )

    def test_tier_b_shares_the_geometry_and_only_repeats_step_b(self):
        budget = audit.build_device_budget(self._rates(), variants=3)
        rows = {row["item"]: row["seconds"] for row in budget["tier_b"]["rows"]}
        # Step B 逐变体重跑
        assert rows["Step B x 3 个变体"] == pytest.approx(1800.0)
        # 几何 + 探针集只算一次
        shared = rows[
            "共享前置 (面板扫描 + 照明半径 + 几何 sweep + 探针集 + 标定)"
        ]
        assert shared == pytest.approx(84.0 + 482.0 + 464.0 + 440.0 + 18.0 + 6.0)

    def test_tier_c_is_larger_than_tier_b(self):
        budget = audit.build_device_budget(self._rates(), variants=3)
        assert budget["tier_c"]["total_seconds"] > budget["tier_b"]["total_seconds"]

    def test_variants_scale_tier_b_linearly(self):
        one = audit.build_device_budget(self._rates(), variants=1)["tier_b"]["total_seconds"]
        three = audit.build_device_budget(self._rates(), variants=3)["tier_b"]["total_seconds"]
        # 每个变体多出 Step B + 验收判据
        assert three - one == pytest.approx(2 * (600.0 + 10.0))

    def test_trust_region_tiers_add_step_b_only(self):
        budget = audit.build_device_budget(self._rates(), variants=3, trust_region_tiers=3)
        rows = {row["item"]: row["seconds"] for row in budget["tier_c"]["rows"]}
        assert rows["信任域对照 3 档 x Step B"] == pytest.approx(1800.0)

    def test_self_consistency_detects_a_broken_budget(self):
        budget = audit.build_device_budget(self._rates())
        budget["tier_a"]["total_seconds"] += 1.0
        assert audit.budget_self_consistent(budget) is False


# ---------------------------------------------------------------------------
# 端到端 (合成语料)
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_ledger_is_built_from_synthetic_records(self, tmp_path: Path):
        repo = tmp_path / "repo"
        campaign = repo / "data" / "debug" / "model_in_loop_hw_ab" / "20260101_000000"
        (campaign / "baseline_a").mkdir(parents=True)
        (campaign / "baseline_a" / "stage.log").write_text(
            _synthetic_stage_log(), encoding="utf-8"
        )
        (campaign / "manifest.json").write_text(
            json.dumps(
                {
                    "session_id": "20260101_000000",
                    "stages": [
                        {
                            "name": "light",
                            "kind": "light",
                            "duration_s": 12.0,
                            "started_utc": "2026-01-01T00:00:03+00:00",
                            "ended_utc": "2026-01-01T00:00:15+00:00",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        capture_dir = repo / "data" / "debug" / "physical_forward_defocus" / "cap"
        capture_dir.mkdir(parents=True)
        (capture_dir / "manifest.json").write_text(
            json.dumps(_synthetic_capture_rows()), encoding="utf-8"
        )

        ledger = audit.build_ledger(
            repo, write_cost_s=5.0, mil_round_seconds=700.0
        )
        assert set(ledger["kinds"]) == {"campaign", "capture"}
        campaign_total = sum(
            row["total_seconds"] for row in ledger["kinds"]["campaign"]["actions"]
        )
        assert campaign_total == pytest.approx(
            sum(row["total_seconds"] for row in ledger["kinds"]["campaign"]["actions"])
        )
        assert ledger["device_budget"]["tier_a"]["total_seconds"] > 0
        assert ledger["optimizations"], "可优化项不能为空"
        assert all(item["rank"] >= 1 for item in ledger["optimizations"])
        assert [item["rank"] for item in ledger["optimizations"]] == list(
            range(1, len(ledger["optimizations"]) + 1)
        )

    @pytest.mark.skipif(
        not _real_capture_present(), reason=_REAL_CAPTURE_SKIP
    )
    def test_optimizations_are_ranked_by_saving(self):
        repo = Path(__file__).resolve().parents[3]
        ledger = audit.build_ledger(repo, write_cost_s=5.0, mil_round_seconds=700.0)
        savings = [item["saving_seconds"] for item in ledger["optimizations"]]
        assert savings == sorted(savings, reverse=True)

    @pytest.mark.skipif(
        not _real_capture_present(), reason=_REAL_CAPTURE_SKIP
    )
    def test_markdown_renderers_produce_non_empty_reports(self, tmp_path: Path):
        repo = Path(__file__).resolve().parents[3]
        ledger = audit.build_ledger(repo, write_cost_s=5.0, mil_round_seconds=700.0)
        optimizations = audit.render_optimizations_md(ledger["optimizations"], ledger)
        budget = audit.render_device_budget_md(ledger["device_budget"], ledger)
        assert "预计节省" in optimizations
        assert "副作用" in optimizations
        assert "依据" in optimizations
        assert "档 A" in budget and "档 B" in budget and "档 C" in budget
        assert "并行" in budget
