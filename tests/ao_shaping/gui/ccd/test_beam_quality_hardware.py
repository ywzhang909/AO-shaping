"""隐藏页停渲染（①）的在线有光硬件验收测试。

用 ``--hardware`` 显式开启；默认收集后跳过，避免离线 CI 碰到真实设备::

    pytest tests/ao_shaping/gui/ccd/test_beam_quality_hardware.py --hardware -v

设备占用与释放（本用例唯一一次同时占用两个设备）：

* 占用：``SLM #1`` + ``Daheng CCD cam_id=0``（页 1 真实迁移学习闭环会话）；
  页 0 用 ``SimBeamSource`` 制造 UI 分析负载，**不占相机**，避免与页 1 抢 CCD。
* 释放：``finally`` 中先请求停止（若仍在运行），再 ``shell.close()``；记录
  关闭耗时、``beam._source`` 与 ``page._running`` 的最终状态；两者都必须在
  证据 JSON 中可见，否则视为未正常释放。
* 证据：默认写到 ``data/debug/beam_quality/hidden_gate_hardware.json``，可用
  环境变量 ``BEAM_HARDWARE_EVIDENCE`` 改路径；采样时长可用
  ``BEAM_HARDWARE_SAMPLE_S`` 覆盖（默认 10 s）。

注意：页 0 打开 sim 源会经 ``get_opened_camera()`` 把页 1 相机误种成 sim，
所以启动真实闭环前必须显式 ``page.set_camera("daheng")``。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

import numpy as np

from ao_shaping.utils.io.file import ROOT_DIR

EVIDENCE_PATH = Path(
    os.environ.get(
        "BEAM_HARDWARE_EVIDENCE",
        str(ROOT_DIR / "data/debug/beam_quality/hidden_gate_hardware.json"),
    )
)
SAMPLE_SECONDS = float(os.environ.get("BEAM_HARDWARE_SAMPLE_S", "10.0"))
RUN_TIMEOUT_S = 900.0


def _pump(app: Any, seconds: float) -> None:
    """手动驱动 Qt 事件循环（没有真正的 exec_()，与一次性脚本口径一致）。"""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.005)


@pytest.mark.hardware
@pytest.mark.slow
@pytest.mark.timeout(1200)
def test_hidden_page_stops_rendering_during_slm_closed_loop() -> None:
    """页 0 可见 / 隐藏两段对比 + 页 1 真实 SLM #1 + CCD 闭环全程保持降频。"""
    pytest.importorskip("PySide6.QtWidgets")
    from ao_shaping.drivers.ccd.daheng.discovery import enumerate_devices
    from ao_shaping.gui.ccd.beam_quality.app import SOURCE_KINDS, BeamQualityApp
    from ao_shaping.gui.ccd.beam_quality.shell import DesktopShellApp
    from PySide6.QtWidgets import QApplication

    discovery = enumerate_devices()
    if not discovery.sdk_ready or not discovery.online:
        pytest.skip(
            f"大恒 CCD 不可用: sdk_ready={discovery.sdk_ready}, "
            f"online={discovery.online}"
        )

    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    app.setApplicationName("ao-shaping-console")

    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    result: dict = {
        "test": "hidden_page_render_gate",
        "optimization": "① hide gate + ② analyze_frame CPU 降本",
        "started_at": started_at,
        "sample_seconds": SAMPLE_SECONDS,
        "qt_qpa_platform": os.environ.get("QT_QPA_PLATFORM", "<unset>"),
        "preflight": {
            "daheng_sdk_ready": discovery.sdk_ready,
            "daheng_online": discovery.online,
        },
        "devices_occupied": {
            "at": started_at,
            "slm": "SLM #1 (页 1 闭环默认设备)",
            "ccd": "Daheng cam_id=0 (页 1 闭环)",
            "page0_source": "sim (不占相机)",
        },
    }
    visible: dict = {}
    hidden: dict = {}
    during_run_early: dict = {}
    run: dict = {}
    cleanup: dict = {}
    analyze_times: list[float] = []
    original_analyze = None

    shell: Any = None
    beam: Any = None
    page: Any = None

    def sample(label: str, seconds: float) -> dict:
        assert beam is not None
        wall0 = time.monotonic()
        cpu0 = time.process_time()
        n0 = beam._processed
        analyze0 = len(analyze_times)
        _pump(app, seconds)
        wall = time.monotonic() - wall0
        cpu = time.process_time() - cpu0
        segment = analyze_times[analyze0:]
        return {
            "label": label,
            "wall_s": round(wall, 3),
            "cpu_s": round(cpu, 3),
            "cpu_per_wall": round(cpu / wall, 3),
            "processed_delta": beam._processed - n0,
            "processed_per_wall": round((beam._processed - n0) / wall, 3),
            "render_active": beam._render_active,
            "poll_interval_ms": beam._poll_timer.interval(),
            "trend_len": len(beam._buffer),
            "analyze_calls": len(segment),
            "analyze_ms_median": (
                round(float(np.median(segment)), 3) if segment else None
            ),
        }

    print(
        f"[硬件占用] {started_at} SLM #1 + Daheng CCD cam_id=0："
        "页 1 迁移学习闭环；页 0 仿真源制造 UI 负载",
        flush=True,
    )

    try:
        shell = DesktopShellApp()
        shell.show()
        _pump(app, 0.5)
        beam = shell._beam_app

        # ② 的耗时证据: 只包一层实例方法, 不改算法口径; 结束时还原。
        original_analyze = beam._analyze

        def timed_analyze(frame):
            t0 = time.perf_counter()
            try:
                return original_analyze(frame)
            finally:
                analyze_times.append((time.perf_counter() - t0) * 1000.0)

        beam._analyze = timed_analyze

        kinds = [kind for _label, kind in SOURCE_KINDS]
        beam._source_combo.setCurrentIndex(kinds.index("sim"))
        beam.open_source()
        deadline = time.monotonic() + 15.0
        while beam._source is None and time.monotonic() < deadline:
            _pump(app, 0.1)
        if beam._source is None:
            raise RuntimeError("仿真帧源未能打开")
        deadline = time.monotonic() + 10.0
        while beam._processed == 0 and time.monotonic() < deadline:
            _pump(app, 0.1)
        result["source_opened"] = {
            "kind": "sim",
            "at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "processed": beam._processed,
        }

        visible = sample("visible", SAMPLE_SECONDS)

        shell.select_page(1)
        deadline = time.monotonic() + 180.0
        while (
            shell._transfer_page is None or beam._render_active
        ) and time.monotonic() < deadline:
            _pump(app, 0.2)
        if shell._transfer_page is None:
            raise RuntimeError("迁移学习闭环整形页预热超时")
        if beam._render_active:
            raise RuntimeError("页 0 隐藏后可见性门控未生效")
        hidden = sample("hidden", SAMPLE_SECONDS)

        page = shell._transfer_page
        # 页 0 的 sim 源会被 get_opened_camera() 误种到页 1；显式改回真机后端。
        page.set_camera("daheng")
        _pump(app, 0.3)
        result["before_run"] = {
            "at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "problems": page.current_problems(),
            "run_enabled": page.run_button().isEnabled(),
            "values": str(page.form_values()),
            "beam_processed": beam._processed,
        }
        if not page.run_button().isEnabled():
            pytest.skip(f"运行按钮不可用（设备/参数前置条件不满足）: {page.current_problems()}")

        page.run_button().click()
        _pump(app, 0.2)
        if not page._running:
            raise RuntimeError(f"点击开始运行后未进入运行态: {page._status.text()}")

        _pump(app, 1.0)
        during_run_early = {
            "t_s": 1.0,
            "beam_poll_interval_ms": beam._poll_timer.interval(),
            "beam_render_active": beam._render_active,
        }

        stages: list[dict] = []
        last_stage = ""
        t_run = time.monotonic()
        deadline = t_run + RUN_TIMEOUT_S
        next_checkpoint = t_run + 60.0
        while page._running and time.monotonic() < deadline:
            _pump(app, 1.0)
            stage = page._current.text()
            if stage != last_stage:
                last_stage = stage
                stages.append(
                    {"t_s": round(time.monotonic() - t_run, 1), "stage": stage}
                )
            if time.monotonic() >= next_checkpoint:
                next_checkpoint += 60.0
                result["during_run_checkpoint"] = {
                    "t_s": round(time.monotonic() - t_run, 1),
                    "beam_processed": beam._processed,
                    "beam_poll_interval_ms": beam._poll_timer.interval(),
                    "beam_render_active": beam._render_active,
                }
        session_dir = Path(page._session_dir) if page._session_dir else None
        run = {
            "started_at": result["before_run"]["at"],
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "wall_s": round(time.monotonic() - t_run, 1),
            "timed_out": bool(page._running),
            "status": page._status.text(),
            "session_dir": str(session_dir) if session_dir else "",
            "stages": stages,
            "journal_exists": bool(
                session_dir and (session_dir / "journal.json").is_file()
            ),
            "optical_probe_exists": bool(
                session_dir and (session_dir / "optical_probe.json").is_file()
            ),
        }
        result["run"] = run
    finally:
        if beam is not None and original_analyze is not None:
            beam._analyze = original_analyze
        # 失败路径同样归还设备：先停闭环，再关外壳。
        if page is not None and getattr(page, "_running", False):
            page.stop_button().click()
            stop_deadline = time.monotonic() + 30.0
            while page._running and time.monotonic() < stop_deadline:
                _pump(app, 0.2)
        close_started = time.monotonic()
        if shell is not None:
            shell.close()
            _pump(app, 0.5)
        released_at = time.strftime("%Y-%m-%d %H:%M:%S")
        cleanup = {
            "at": released_at,
            "close_wall_s": round(time.monotonic() - close_started, 3),
            "beam_source": str(beam._source) if beam is not None else "<no beam app>",
            "transfer_running": (
                bool(page._running) if page is not None else "<no transfer page>"
            ),
        }
        result["devices_released"] = cleanup
        # 设备前置不满足而 skip 时不要覆盖上一次成功运行的证据。
        if shell is not None:
            EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
            EVIDENCE_PATH.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(
                f"[硬件释放] {released_at} SLM #1 + Daheng CCD 已关闭；"
                f"close={cleanup['close_wall_s']:.3f}s, "
                f"source={cleanup['beam_source']}, "
                f"running={cleanup['transfer_running']}；证据 {EVIDENCE_PATH}",
                flush=True,
            )

    assert visible["render_active"] is True
    assert visible["poll_interval_ms"] == BeamQualityApp.POLL_INTERVAL_MS
    assert visible["processed_delta"] > 0
    assert visible["analyze_calls"] > 0
    assert visible["analyze_ms_median"] is not None
    assert hidden["render_active"] is False
    assert hidden["poll_interval_ms"] == BeamQualityApp.HIDDEN_POLL_INTERVAL_MS
    assert hidden["processed_delta"] > 0
    assert hidden["analyze_calls"] < visible["analyze_calls"]
    assert hidden["trend_len"] > visible["trend_len"]
    assert hidden["cpu_per_wall"] < visible["cpu_per_wall"]
    assert hidden["processed_per_wall"] < visible["processed_per_wall"]
    assert during_run_early["beam_render_active"] is False
    assert during_run_early["beam_poll_interval_ms"] == BeamQualityApp.HIDDEN_POLL_INTERVAL_MS
    checkpoint = result.get("during_run_checkpoint")
    if checkpoint is not None:
        assert checkpoint["beam_render_active"] is False
        assert checkpoint["beam_poll_interval_ms"] == BeamQualityApp.HIDDEN_POLL_INTERVAL_MS
    assert run["timed_out"] is False
    assert "会话完成" in run["status"]
    assert run["journal_exists"] is True
    assert run["optical_probe_exists"] is True
    assert cleanup["beam_source"] == "None"
    assert cleanup["transfer_running"] is False
