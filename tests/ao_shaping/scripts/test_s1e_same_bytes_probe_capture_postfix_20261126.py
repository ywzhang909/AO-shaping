# -*- coding: utf-8 -*-
"""离线门禁: 修复后 S1e 重采脚本 (§12.16.5 post-fix round).

断言:
- 计划模式只读、顺序/数组/前驱仍与冻结 §12.16.2 一致 (12 pattern, 24 frame);
- ``physical_sha256`` 现记录守卫生效后的实际送屏字节 (H = 逻辑 C 序 7276d9da),
  ``pre_guard_buffer_sha256`` 保留修复前的 F 序裸 buffer (b98be9a6) 供审计;
- B / flat 的 C-contiguous 输入前后字节一致, 判据阈值不变。

不打开设备、不取锁、不建输出目录。冻结脚本 / 冻结测试的 blob 不在本文件范围内。
"""

from __future__ import annotations

import argparse
import importlib
from typing import Any

import numpy as np
import pytest

SCRIPT = importlib.import_module(
    "scripts.slm.s1e_same_bytes_probe_capture_postfix_20261126"
)

# 独立转写自 §12.16.2, 不从 SCRIPT 导入.
FROZEN_ORDER: tuple[str, ...] = (
    "flat",
    "s1e_a1",
    "s1e_a2",
    "s1e_b1",
    "s1e_a3",
    "s1e_b2",
    "s1e_a4",
    "flat_mid",
    "s1e_a5",
    "s1e_b3",
    "s1e_a6",
    "flat_post",
)
FROZEN_ARRAY_BY_ID = {
    "flat": "flat",
    "s1e_a1": "H",
    "s1e_a2": "H",
    "s1e_b1": "B",
    "s1e_a3": "H",
    "s1e_b2": "B",
    "s1e_a4": "H",
    "flat_mid": "flat",
    "s1e_a5": "H",
    "s1e_b3": "B",
    "s1e_a6": "H",
    "flat_post": "flat",
}
H_LOGICAL = "7276d9da3b2a5ea536d56ea944adcfdcb38c98ad1dd01c1433d1b2219ac8218e"
H_PRE_GUARD = "b98be9a6696cd94dbb3fa966e22a7f86e52f52b16ae252d21f7aa07af4079c3a"
B_BYTES = "c8d937cedb294339fa385f9ffa65cebd0ff3471ff62009cef42d7637bf590a73"
FLAT_BYTES = "cb228af4065a8be95006792a8199172366d0b63a4d39ce365f9591246861db16"
FROZEN_CRITERIA: dict[str, Any] = {
    "intra_upload_max_dist_px": 1.5,
    "D1": {"min_effective_uploads": 5, "max_pairwise_dist_px": 1.5},
    "D2": {
        "axis_ratio_min": 1.5,
        "col_dist_px": [90.0, 102.0],
        "row_dist_px": [52.0, 64.0],
    },
    "D4": {"weak_peak_fraction": 0.10},
    "D6": {"zero_exception_groups_required": True},
}


def _args(**overrides: Any) -> argparse.Namespace:
    values: dict[str, Any] = {
        "out": None,
        "execute": False,
        "wait_until_idle": 1800.0,
        "exposure_ms": 0.087,
        "settle_s": 0.8,
        "frames_per_upload": 2,
        "slm_number": 1,
        "cam_id": 0,
        "wavelength_nm": 1064.0,
        "saturation_fraction": 0.9,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_frozen_order_arrays_and_criteria_unchanged() -> None:
    assert SCRIPT.FROZEN_ORDER == FROZEN_ORDER
    assert SCRIPT.ARRAY_BY_ID == FROZEN_ARRAY_BY_ID
    assert len(set(FROZEN_ORDER)) == 12
    assert [i for i, pid in enumerate(FROZEN_ORDER) if SCRIPT.ARRAY_BY_ID[pid] == "H"] == [
        1, 2, 4, 6, 8, 10,
    ]
    assert SCRIPT.FROZEN_CRITERIA == FROZEN_CRITERIA


def test_build_plan_is_readonly_and_keeps_frozen_shape() -> None:
    plan = SCRIPT.build_plan(_args())
    assert list(plan["capture_order"]) == list(FROZEN_ORDER)
    assert len(plan["patterns"]) == 12
    assert len(plan["frames"]) == 24
    assert plan["display_topology"]["status"] == "not_probed"
    assert plan["focus_compensation"]["focus_d_m1"] == 0.0
    assert "post-fix" in plan["design_source"]


def test_postfix_expected_digests_record_sent_and_pre_guard_bytes() -> None:
    h = SCRIPT.expected_digests("H")
    assert h["gray_sha256"] == H_LOGICAL
    assert h["physical_sha256"] == H_LOGICAL  # 守卫生效后实际送屏 == 逻辑 C 序
    assert h["pre_guard_buffer_sha256"] == H_PRE_GUARD  # 修复前 F 序裸 buffer
    assert h["physical_sha256"] != h["pre_guard_buffer_sha256"]

    b = SCRIPT.expected_digests("B")
    assert b["physical_sha256"] == B_BYTES
    assert b["pre_guard_buffer_sha256"] == B_BYTES  # C-contiguous: 前后一致

    flat = SCRIPT.expected_digests("flat")
    assert flat["physical_sha256"] == FLAT_BYTES
    assert flat["pre_guard_buffer_sha256"] == FLAT_BYTES


def test_layout_record_reports_both_byte_streams_for_frozen_arrays() -> None:
    h = SCRIPT.build_gray("s1e_a1")
    assert h.flags.f_contiguous and not h.flags.c_contiguous  # 冻结布局未变
    layout = SCRIPT.layout_record(h)
    assert layout["physical_sha256"] == H_LOGICAL
    assert layout["pre_guard_buffer_sha256"] == H_PRE_GUARD
    assert layout["f_contiguous"] and not layout["c_contiguous"]

    b = SCRIPT.build_gray("s1e_b1")
    assert b.flags.c_contiguous
    layout_b = SCRIPT.layout_record(b)
    assert layout_b["physical_sha256"] == B_BYTES
    assert layout_b["pre_guard_buffer_sha256"] == B_BYTES


def test_check_frozen_arrays_accepts_postfix_plan() -> None:
    plan = SCRIPT.build_plan(_args())
    SCRIPT._check_frozen_arrays(plan["patterns"])  # 不应抛异常


def test_check_frozen_arrays_rejects_pre_guard_buffer_drift(
    monkeypatch: Any,
) -> None:
    records = SCRIPT.build_pattern_records()
    # 篡改 H 的 pre_guard 记录, 应被门禁捕获.
    for record in records:
        if record["array"] == "H":
            record["gray_layout"]["pre_guard_buffer_sha256"] = "0" * 64
            break
    with pytest.raises(RuntimeError, match="pre-guard buffer drifted"):
        SCRIPT._check_frozen_arrays(records)
