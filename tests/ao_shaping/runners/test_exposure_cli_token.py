"""The three-state exposure vocabulary: ``auto`` / ``default`` / a number.

Why these exist
---------------
``--exposure-time-ms`` used to be declared seven times in ``runner_common.py``
plus once in ``tools/slm/panel/panel_locate.py``, and ``0`` meant two different
things across those sites (the runner configs read it as "no fixed value",
``panel_locate`` read it as "probe the ceiling"). A user who wrote the token
meant one thing got the other. :mod:`ao_shaping.utils.cli.exposure` is the one
resolver that now decides, and this file is its contract:

* ``auto`` means **exactly** what ``0`` meant at that site -- the compatibility
  anchor, deliberately not unified across sites;
* ``default`` is a *separate* answer ("change nothing") and must never be
  collapsed onto ``auto``'s sentinel;
* a bare ``0`` is an error, and the error names ``auto``.

Offline: no device is opened, so this belongs in the default ``pytest`` run
(the online counterpart is ``tests/ao_shaping/drivers/ccd/test_exposure_states.py``,
which the ``addopts`` ``--ignore`` keeps out of a bare run).
"""

from __future__ import annotations

import click
import pytest

from ao_shaping.utils.cli.exposure import (
    AUTO_EXPOSURE,
    AUTO_EXPOSURE_TOKEN,
    DEFAULT_EXPOSURE,
    DEFAULT_EXPOSURE_TOKEN,
    ExposureChoice,
    exposure_ms_legacy,
    resolve_exposure,
)

HINT = "--exposure-time-ms"


class TestNumeric:
    def test_decimal_string(self) -> None:
        assert resolve_exposure("0.1", param_hint=HINT) == 0.1

    @pytest.mark.parametrize(
        ("raw", "expected"), [("1", 1.0), ("0.02", 0.02), ("  2.5  ", 2.5), (3, 3.0), (0.5, 0.5)]
    )
    def test_accepted_spellings(self, raw: object, expected: float) -> None:
        assert resolve_exposure(raw, param_hint=HINT) == expected

    def test_numbers_round_trip_through_the_legacy_collapse(self) -> None:
        assert exposure_ms_legacy(resolve_exposure("0.1", param_hint=HINT)) == 0.1


class TestTokens:
    @pytest.mark.parametrize("raw", ["auto", "AUTO", "Auto", " auto ", "\tauto\n"])
    def test_auto_is_recognised_case_and_space_insensitively(self, raw: str) -> None:
        assert resolve_exposure(raw, param_hint=HINT) is AUTO_EXPOSURE

    @pytest.mark.parametrize("raw", ["default", "DEFAULT", " default "])
    def test_default_is_recognised(self, raw: str) -> None:
        assert resolve_exposure(raw, param_hint=HINT) is DEFAULT_EXPOSURE

    def test_the_two_tokens_are_distinguishable_values(self) -> None:
        """The whole point of the enum: ``default`` must not *be* ``auto``.

        Collapsing them into one sentinel is exactly the bug this file exists to
        prevent -- it is what let one ``0`` carry two meanings.
        """
        assert AUTO_EXPOSURE is not DEFAULT_EXPOSURE
        assert len({AUTO_EXPOSURE, DEFAULT_EXPOSURE}) == 2

    def test_both_tokens_collapse_onto_the_legacy_float_rung(self) -> None:
        """What keeps ``auto`` bit-compatible with the ``0`` it replaces.

        ``exposure_ms_legacy`` is the *only* place the vocabulary meets the
        float APIs; a caller that wants to honour ``default`` separately must
        branch before calling it (see ``panel_locate.scan_panel``).
        """
        assert exposure_ms_legacy(AUTO_EXPOSURE) == 0.0
        assert exposure_ms_legacy(DEFAULT_EXPOSURE) == 0.0

    def test_none_reads_as_auto(self) -> None:
        """An unset programmatic value must not be an error."""
        assert resolve_exposure(None, param_hint=HINT) is AUTO_EXPOSURE

    def test_an_enum_member_round_trips(self) -> None:
        assert resolve_exposure(DEFAULT_EXPOSURE, param_hint=HINT) is DEFAULT_EXPOSURE


class TestRejected:
    @pytest.mark.parametrize(
        "raw",
        [
            "0",
            "0.0",
            "0.00",
            " 0 ",
            0,
            0.0,
            "-0",
            -0.0,
        ],
    )
    def test_bare_zero_is_rejected_in_every_spelling(self, raw: object) -> None:
        """``0`` is not a synonym for either token -- it is an error."""
        with pytest.raises(click.BadParameter, match="auto"):
            resolve_exposure(raw, param_hint=HINT)

    @pytest.mark.parametrize("raw", ["-1", "-0.01", -1, -0.01, "-inf", "nan", "inf"])
    def test_negative_and_non_finite_are_rejected(self, raw: object) -> None:
        with pytest.raises(click.BadParameter, match="auto"):
            resolve_exposure(raw, param_hint=HINT)

    @pytest.mark.parametrize("raw", ["", "  ", "abc", "0.1ms", "1,5", [], {"ms": 1}])
    def test_unparseable_text_is_rejected(self, raw: object) -> None:
        with pytest.raises(click.BadParameter, match="auto"):
            resolve_exposure(raw, param_hint=HINT)

    @pytest.mark.parametrize("raw", [True, False])
    def test_bools_are_not_exposures(self, raw: bool) -> None:
        """``bool`` is an ``int``; without the guard ``True`` is a 1 ms exposure."""
        with pytest.raises(click.BadParameter, match="auto"):
            resolve_exposure(raw, param_hint=HINT)

    def test_the_message_names_both_tokens_and_attributes_the_flag(self) -> None:
        with pytest.raises(click.BadParameter) as excinfo:
            resolve_exposure("0", param_hint=HINT)
        message = str(excinfo.value)
        assert AUTO_EXPOSURE_TOKEN in message
        assert DEFAULT_EXPOSURE_TOKEN in message
        assert excinfo.value.param_hint == HINT

    def test_one_message_for_every_unusable_value(self) -> None:
        """One rule, one message -- the anti-drift property of a single resolver."""
        messages = set()
        for raw in ("0", -1, "", "abc", float("nan")):
            with pytest.raises(click.BadParameter) as excinfo:
                resolve_exposure(raw, param_hint=HINT)
            # Strip the echoed input so only the fixed part is compared.
            messages.add(str(excinfo.value).split(" is not a usable exposure:")[1])
        assert len(messages) == 1


class TestTokenSpellings:
    def test_the_tokens_are_what_the_help_text_advertises(self) -> None:
        assert AUTO_EXPOSURE_TOKEN == ExposureChoice.AUTO.value == "auto"
        assert DEFAULT_EXPOSURE_TOKEN == ExposureChoice.DEFAULT.value == "default"