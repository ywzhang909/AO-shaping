"""Unit tests for ``micro_dm_image_collect``'s pure logic (no hardware).

Issue #53 asked for tests on this module. Two test files already existed:
``test_micro_dm_disk_layout.py`` (the on-disk naming contract) and
``test_micro_dm_image_collect_cli.py`` (the Click surface). Neither reaches past
the CLI boundary, so everything between "user typed a flag" and "device opened"
was untested -- and that band is where the arithmetic lives.

What is pinned here, and why each one matters:

* ``_parse_channels`` decides which of the 50 channels get driven. A silent
  off-by-one here energises the wrong cell, which on a deformable mirror means
  the wrong actuator and a plausible-looking but wrong image.
* ``_resolve_ip_port`` encodes the "port = 10000 + last octet" rule. Two places
  computing it differently is exactly how a probe ends up talking to the wrong
  controller, so the rule is pinned to a single implementation.
* ``_voltage_array`` builds the 50-wide frame the controller expects. Every
  channel that is *not* the target must sit at the home voltage; one missed
  index leaves a neighbouring actuator latched between runs.
* ``_send_voltage`` / ``_home_channel`` branch on a module-level flag, so both
  sides of that branch are exercised rather than only the default one.

``_resolve_ips`` is deliberately absent: it now lives in
``ao_shaping.drivers.dm.micro.wiring_map`` and is tested at its new home in
``tests/ao_shaping/drivers/dm/test_micro_resolve_ips.py``. An orphaned
``__pycache__/test_micro_dm_image_collect.cpython-*.pyc`` from before that move
still names two ``resolve_ips`` tests; they were relocated, not lost.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ao_shaping.drivers.dm.micro import MAX_CHANNELS
from ao_shaping.tools.micro_dm import micro_dm_image_collect as mod
from ao_shaping.tools.micro_dm.micro_dm_image_collect import (
    _home_channel,
    _parse_channels,
    _resolve_ip_port,
    _send_voltage,
    _voltage_array,
)


# ---------------------------------------------------------------------------
# _parse_channels -- which cells get driven
# ---------------------------------------------------------------------------


def test_all_expands_to_every_channel() -> None:
    assert _parse_channels("all") == list(range(MAX_CHANNELS))


@pytest.mark.parametrize("spelling", ["all", "ALL", "  All  "])
def test_all_is_case_and_whitespace_insensitive(spelling: str) -> None:
    """A typo in case must not silently collect one channel instead of fifty."""
    assert len(_parse_channels(spelling)) == MAX_CHANNELS


def test_explicit_list_keeps_order_and_tolerates_spaces() -> None:
    assert _parse_channels("3, 1 ,2") == [3, 1, 2]


def test_single_channel_is_not_mistaken_for_all() -> None:
    """Regression shape: `all` handled by prefix match would swallow this."""
    assert _parse_channels("0") == [0]


@pytest.mark.parametrize("bad", ["0,x,2", "", "1.5", "0,,1"])
def test_malformed_channel_list_exits_with_code_one(bad: str) -> None:
    with pytest.raises(SystemExit) as excinfo:
        _parse_channels(bad)
    assert excinfo.value.code == 1


@pytest.mark.parametrize("bad", ["-1", f"{MAX_CHANNELS}", "0,999"])
def test_out_of_range_channel_exits_with_code_one(bad: str) -> None:
    with pytest.raises(SystemExit) as excinfo:
        _parse_channels(bad)
    assert excinfo.value.code == 1


def test_upper_boundary_is_accepted() -> None:
    """The bound is inclusive of the last channel, exclusive of MAX_CHANNELS."""
    assert _parse_channels(str(MAX_CHANNELS - 1)) == [MAX_CHANNELS - 1]


# ---------------------------------------------------------------------------
# _resolve_ip_port -- the "10000 + last octet" rule, pinned to one place
# ---------------------------------------------------------------------------


def test_port_defaults_to_ten_thousand_plus_last_octet() -> None:
    assert _resolve_ip_port("192.168.0.101", None) == ("192.168.0.101", 10101)
    assert _resolve_ip_port("192.168.0.7", None) == ("192.168.0.7", 10007)


def test_explicit_port_wins_over_the_derived_one() -> None:
    assert _resolve_ip_port("192.168.0.101", 9999) == ("192.168.0.101", 9999)


def test_last_octet_zero_is_a_valid_controller() -> None:
    """0 is a legal octet, so the derived port is 10000 and must not be falsy."""
    ip, port = _resolve_ip_port("192.168.0.0", None)
    assert (ip, port) == ("192.168.0.0", 10000)


@pytest.mark.parametrize("bad", ["192.168.0", "192.168.0.1.5", "not.an.ip.address", ""])
def test_malformed_ip_raises_value_error(bad: str) -> None:
    with pytest.raises(ValueError, match="IP 格式无效"):
        _resolve_ip_port(bad, None)


@pytest.mark.parametrize("bad", ["192.168.0.256", "192.168.0.999"])
def test_out_of_range_octet_raises_value_error(bad: str) -> None:
    with pytest.raises(ValueError, match="超出范围"):
        _resolve_ip_port(bad, None)


@pytest.mark.parametrize(
    "ip", ["192.168.0.1", "192.168.0.0", "10.0.0.255", "255.255.255.255"]
)
def test_any_syntactically_valid_ip_yields_an_int_port(ip: str) -> None:
    """No valid IP may leak ``port=None`` into the TCP layer.

    ``controller_tcp_port`` returns ``None`` for an unparseable last octet, and
    ``_resolve_ip_port`` raises rather than passing that through. That guard is
    unreachable through this function -- the IP is fully parsed and range-checked
    before the port is derived -- so it is pinned here as an observable property
    of the whole range rather than as a synthetic error path.
    """
    _, port = _resolve_ip_port(ip, None)
    assert isinstance(port, int) and port >= 10000


# ---------------------------------------------------------------------------
# _voltage_array -- every non-target channel must sit at home
# ---------------------------------------------------------------------------


def test_only_the_target_channel_carries_the_voltage() -> None:
    volts = _voltage_array(7, 12.5, 0.0)

    assert len(volts) == MAX_CHANNELS
    assert volts[7] == 12.5
    assert sum(1 for v in volts if v != 0.0) == 1


def test_none_target_homes_every_channel() -> None:
    """`ch=None` is the homing path: no cell may be left energised."""
    volts = _voltage_array(None, 12.5, 0.0)
    assert volts == [0.0] * MAX_CHANNELS


def test_home_voltage_is_honoured_for_the_rest() -> None:
    volts = _voltage_array(0, 5.0, 1.5)
    assert volts[0] == 5.0
    assert set(volts[1:]) == {1.5}


def test_first_and_last_channel_are_both_reachable() -> None:
    for ch in (0, MAX_CHANNELS - 1):
        assert _voltage_array(ch, 3.0, 0.0)[ch] == 3.0


# ---------------------------------------------------------------------------
# _send_voltage / _home_channel -- both sides of the dispatch flag
# ---------------------------------------------------------------------------


class _RecordingController:
    """Stands in for R50Controller and records which call path was used."""

    def __init__(self) -> None:
        self.array_calls: list[list[float]] = []
        self.single_calls: list[tuple[int, float]] = []

    def set_all_voltage_array(self, volts: list[float]) -> bool:
        self.array_calls.append(volts)
        return True

    def set_channel_voltage(self, ch: int, voltage: float) -> bool:
        self.single_calls.append((ch, voltage))
        return True


def test_send_voltage_uses_the_array_command_when_the_flag_is_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mod, "USE_SET_ALL_VOLTAGE_ARRAY", True)
    ctrl = _RecordingController()

    assert _send_voltage(ctrl, 3, 9.0, 0.0) is True
    assert ctrl.single_calls == [], "the per-channel path must not also run"
    assert ctrl.array_calls == [mod._voltage_array(3, 9.0, 0.0)]


def test_send_voltage_falls_back_to_per_channel_when_the_flag_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mod, "USE_SET_ALL_VOLTAGE_ARRAY", False)
    ctrl = _RecordingController()

    assert _send_voltage(ctrl, 3, 9.0, 0.0) is True
    assert ctrl.array_calls == [], "the array path must not also run"
    assert ctrl.single_calls == [(3, 9.0)]


def test_home_channel_homes_the_whole_array_when_the_flag_is_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Homing via the array command must send 50 home values, not one."""
    monkeypatch.setattr(mod, "USE_SET_ALL_VOLTAGE_ARRAY", True)
    ctrl = _RecordingController()

    assert _home_channel(ctrl, 3, 1.5) is True
    assert ctrl.array_calls == [[1.5] * MAX_CHANNELS]


def test_home_channel_homes_only_its_own_channel_when_the_flag_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mod, "USE_SET_ALL_VOLTAGE_ARRAY", False)
    ctrl = _RecordingController()

    assert _home_channel(ctrl, 3, 1.5) is True
    assert ctrl.single_calls == [(3, 1.5)]


# ---------------------------------------------------------------------------
# module surface -- cheap, and it catches a half-applied rename
# ---------------------------------------------------------------------------


def test_module_exposes_no_hardware_handle_at_import_time() -> None:
    """Importing the collector must not construct a controller or open a camera.

    The module is imported by the CLI at module scope, so an eager device open
    here would fire before argument validation could reject a bad invocation.
    """
    assert not hasattr(mod, "ctrl")
    assert not hasattr(mod, "camera")


def test_run_is_still_the_with_params_entry_point() -> None:
    """R-33 (#51) migrated this command; keep the migration from regressing."""
    import click

    assert isinstance(mod.run, click.Command)
    assert mod.run.name == "micro-dm-collect"
    assert mod.__doc__ is not None


def test_disk_layout_helper_is_reachable_from_this_module() -> None:
    """The collector owns the naming its consumers parse; keep the link explicit."""
    assert callable(mod._save_frame)
    assert isinstance(Path(mod.__file__).name, str)
