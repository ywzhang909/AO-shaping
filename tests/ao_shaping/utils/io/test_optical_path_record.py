"""The standard optical-path record keeps every schema key, missing or not.

``libs/skills/ao-optical-path-calibration/SKILL.md`` §4 fixes the fields of
``optical_path_record.json`` because that is the only way to compare two
experiments without mistaking optical-path or device drift for an algorithm
effect. The builder is pure, so the contracts below can be pinned without a
camera or an SLM:

* every key of the v1 schema is present in a complete record;
* a field the caller could not supply is ``null`` *and* explained in
  ``missing`` -- never dropped (a dropped key reads as "not applicable" while a
  null reads as "unknown", and the comparison rule depends on the difference);
* the schema string is exactly ``ao-optical-path-record/v1``.
"""

from __future__ import annotations

from typing import Any

from ao_shaping.utils.io.optical_path_record import (
    SCHEMA,
    build_optical_path_record,
)

# The v1 schema as spelled out in SKILL.md §4, written out here on purpose: the
# test pins the document, not the module's own constants.
SCHEMA_KEYS: dict[str, Any] = {
    "schema": None,
    "run_id": None,
    "date": None,
    "operator_note": None,
    "devices": {
        "slm": (
            "model",
            "drive_board_id",
            "option_board_id",
            "slm_number",
            "wavelength_nm",
            "two_pi_grayscale",
            "panel_res",
            "readsu_status_at_open",
            "open_mode",
            "power_cycled_this_session",
        ),
        "camera": (
            "model",
            "serial",
            "bit_depth",
            "full_scale",
            "exposure_range_ms",
            "exposure_used_ms",
            "gain",
            "averaging_frames",
            "discard_frames_after_exposure_change",
            "pixel_size_um",
        ),
        "wfs": None,
        "dm": None,
    },
    "optical_path": (
        "config_id",
        "aperture",
        "fresnel_compensation_m_inv",
        "geometry_calib_file",
        "geometry_calib_mtime",
        "probe_status",
        "probe_report",
    ),
    "exposure_check": (
        "peak_counts_at_used_exposure",
        "saturation_threshold_counts",
        "saturated",
        "settle_timeout",
    ),
    "occupancy": ("acquired_at", "released_at", "closed_normally"),
    "git_commit": None,
}


def _assert_schema_keys(
    record: dict[str, Any],
    spec: dict[str, Any] = SCHEMA_KEYS,
    prefix: str = "",
) -> None:
    """每条 schema 键都在 (递归; 值是什么不在此处断言)。"""
    for key, expected in spec.items():
        path = f"{prefix}{key}"
        assert key in record, f"缺键: {path}"
        if expected is None:
            continue
        assert isinstance(record[key], dict), f"{path} 应当是段 (dict)"
        if isinstance(expected, dict):
            _assert_schema_keys(record[key], spec=expected, prefix=f"{path}.")
        else:
            for field in expected:
                assert field in record[key], f"缺键: {path}.{field}"


def _complete_inputs() -> dict[str, Any]:
    return {
        "run_id": "20261116_093000_probe",
        "date": "2026-11-16",
        "operator_note": "bench as left by the previous session",
        "devices": {
            "slm": {
                "model": "Santec SLM-200",
                "drive_board_id": "22030108",
                "option_board_id": "23020027",
                "slm_number": 1,
                "wavelength_nm": 1064,
                "two_pi_grayscale": 993,
                "panel_res": [1920, 1200],
                "readsu_status_at_open": 0,
                "open_mode": "memory",
                "power_cycled_this_session": False,
            },
            "camera": {
                "model": "Daheng MER2-507-23GM",
                "serial": "FJB24112232",
                "bit_depth": 12,
                "full_scale": 4095,
                "exposure_range_ms": [0.05, 50.0],
                "exposure_used_ms": 50.0,
                "gain": 0.0,
                "averaging_frames": 1,
                "discard_frames_after_exposure_change": 3,
                "pixel_size_um": 2.2,
            },
            "wfs": "not used",
            "dm": "not used",
        },
        "optical_path": {
            "config_id": "2f_1064_aperture_closed",
            "aperture": "closed",
            "fresnel_compensation_m_inv": 0.626,
            "geometry_calib_file": "data/slm_geometry_calib.npz",
            "geometry_calib_mtime": "2026-11-16T08:12:00+00:00",
            "probe_status": "ok",
            "probe_report": "data/debug/optical_path_probe/20261116_093000Z/optical_path_probe.json",
        },
        "exposure_check": {
            "peak_counts_at_used_exposure": 173.0,
            "saturation_threshold_counts": 3890.25,
            "saturated": False,
            "settle_timeout": False,
        },
        "occupancy": {
            "acquired_at": "2026-11-16T09:29:58+00:00",
            "released_at": "2026-11-16T09:36:41+00:00",
            "closed_normally": True,
        },
        "git_commit": "1a2b3c4",
    }


def test_a_complete_record_carries_every_schema_key() -> None:
    record = build_optical_path_record(**_complete_inputs())

    _assert_schema_keys(record)
    assert record["devices"]["slm"]["panel_res"] == [1920, 1200]
    assert record["devices"]["camera"]["exposure_used_ms"] == 50.0
    assert record["optical_path"]["probe_status"] == "ok"
    assert record["occupancy"]["closed_normally"] is True
    assert record["missing"] == {}, "完整输入的记录不该有 missing 条目"


def test_a_missing_input_is_null_with_a_reason_and_still_present() -> None:
    # 只有 run_id: 其余每一段都缺, 每个键仍须在, 值须为 null 且带原因。
    record = build_optical_path_record(run_id="20261116_093000_probe")

    _assert_schema_keys(record)
    assert record["run_id"] == "20261116_093000_probe"
    assert record["optical_path"]["aperture"] is None
    assert record["optical_path"]["probe_status"] is None
    assert record["exposure_check"]["peak_counts_at_used_exposure"] is None
    assert record["occupancy"]["released_at"] is None
    assert record["devices"]["slm"]["drive_board_id"] is None

    for path in (
        "date",
        "git_commit",
        "optical_path.aperture",
        "exposure_check.settle_timeout",
        "occupancy.released_at",
        "devices.slm.drive_board_id",
        "devices.camera.pixel_size_um",
    ):
        assert path in record["missing"], f"{path} 是 null, 必须带原因"
        assert record["missing"][path].strip(), f"{path} 的原因不能是空串"

    # 段存在、只缺一个字段: 缺的那个 null + 原因, 其余照常保留。
    partial = build_optical_path_record(
        devices={"slm": {"slm_number": 1}, "wfs": "not used", "dm": "not used"}
    )
    assert partial["devices"]["slm"]["slm_number"] == 1
    assert partial["devices"]["slm"]["two_pi_grayscale"] is None
    assert partial["missing"]["devices.slm.two_pi_grayscale"]
    assert partial["missing"]["devices.camera.serial"]

    # 调用方给的原因优先于缺省文案。
    explained = build_optical_path_record(
        optical_path={"aperture": None},
        reasons={"optical_path.aperture": "探针不控制光阑"},
    )
    assert explained["missing"]["optical_path.aperture"] == "探针不控制光阑"


def test_the_schema_string_is_the_v1_record() -> None:
    assert SCHEMA == "ao-optical-path-record/v1"
    assert build_optical_path_record()["schema"] == "ao-optical-path-record/v1"
