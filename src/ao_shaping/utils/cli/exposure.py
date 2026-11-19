"""``--exposure-time-ms`` / ``--exposure-ms``: one flag, three states, one resolver.

Why this module exists
----------------------
The exposure flag is declared **seven** times in
:mod:`ao_shaping.runners.runner_common` plus once in
:mod:`ao_shaping.tools.slm.panel.panel_locate`, and ``0`` meant **two genuinely
different things** depending on which declaration you landed on:

* the runner configs read a non-positive value as "no fixed value" -- the camera
  is opened with it and the optimizer auto-exposes; on Daheng the literal ``0``
  that reaches the driver is *clamped to the device minimum* (~0.02 ms);
* ``panel_locate`` reads ``0`` as a *request*: probe the device ceiling
  (:func:`resolve_daheng_exposure_limit` / :func:`probe_exposure_ceiling_ms`)
  and work at a fraction of it.

Same spelling, different semantics, and the ambiguity had hardware consequences:
``runners/slm/model_in_loop_session.py`` forwards its ``--exposure-time-ms``
straight into ``PanelLocateParams.exposure_ms``, so the runner's "clamp to the
device minimum" and the probe's "probe the ceiling" were the same run. On
2026-10-19 a 0.1 ms request there put the preflight flat reference on the noise
floor and tripped the flat-drift gate.

The three states
----------------
=====================  =========================================================
``auto``               "Let the session/device determine it." **This means
                       exactly what ``0`` meant at that call site today** --
                       the compatibility anchor. It is deliberately *not*
                       unified across sites, because the two existing meanings
                       are different physical things.
``default``           "Change nothing; keep whatever the device is set to." No
                       probe, no reset, no clamp to the device minimum.
a number              Use that many milliseconds, unchanged.
=====================  =========================================================

``0`` is rejected in every spelling. It was the token whose meaning already
differed between call sites, so accepting it would keep the ambiguity that
caused the failure.

The two non-numeric states are returned as two *distinct* values
(:class:`ExposureChoice` members), never as one shared ``None``/``0``: a caller
that can honour "keep the device's current setting" has to be able to tell it
apart from "probe the ceiling".

Honest limitation (report this, do not paper over it)
-----------------------------------------------------
:func:`exposure_ms_legacy` collapses both sentinels onto ``0.0``, because every
float API downstream (``optimize_pib``, ``SlmGsRefineConfig``,
``create_camera(exposure_time_ms=...)``) has a single "no fixed value" rung. So
``default`` is only *behaviourally* distinct today at
:func:`ao_shaping.tools.slm.panel.panel_locate.scan_panel`, which is the one
site that actually had separate probe and keep-current code paths. Giving the
other sites a real "don't touch" path needs a third state threaded through the
optimizer configs, which is not an additive change. Separately, on Daheng the
driver writes (and clamps) the exposure inside ``open()``, so even a caller that
wanted to leave it alone cannot: that is a driver-level constraint, not a CLI
one.

Leaf module
-----------
Same rule as :mod:`ao_shaping.utils.cli.params`: stdlib + ``click`` only, never
``ao_shaping``. ``runners/`` and ``tools/`` both consume this, and
``runners/slm/zernike_matrix_runner.py`` imports ``ao_shaping.tools.slm`` at
module level, so anything else here would reopen that cycle.
"""

from __future__ import annotations

import math
from enum import Enum

import click

__all__ = [
    "AUTO_EXPOSURE",
    "AUTO_EXPOSURE_TOKEN",
    "DEFAULT_EXPOSURE",
    "DEFAULT_EXPOSURE_TOKEN",
    "ExposureChoice",
    "exposure_ms_legacy",
    "resolve_exposure",
]

#: The literal CLI token meaning "let the session/device determine the exposure".
AUTO_EXPOSURE_TOKEN = "auto"
#: The literal CLI token meaning "change nothing; keep the device's setting".
DEFAULT_EXPOSURE_TOKEN = "default"


class ExposureChoice(Enum):
    """The two non-numeric answers, as two *distinct* values.

    An enum rather than two bare sentinels so a caller cannot construct a third
    identity, and so ``is``/``isinstance`` checks read as intent.
    """

    AUTO = AUTO_EXPOSURE_TOKEN
    DEFAULT = DEFAULT_EXPOSURE_TOKEN


AUTO_EXPOSURE = ExposureChoice.AUTO
DEFAULT_EXPOSURE = ExposureChoice.DEFAULT

_TOKENS: dict[str, ExposureChoice] = {
    AUTO_EXPOSURE_TOKEN: ExposureChoice.AUTO,
    DEFAULT_EXPOSURE_TOKEN: ExposureChoice.DEFAULT,
}


def _reject(raw: object, param_hint: str) -> click.BadParameter:
    """The single error every unusable value gets, always naming ``auto``."""
    return click.BadParameter(
        f"{raw!r} is not a usable exposure: pass a positive number of "
        f"milliseconds, {AUTO_EXPOSURE_TOKEN!r} (let the session/device choose) "
        f"or {DEFAULT_EXPOSURE_TOKEN!r} (change nothing). "
        f"0 is not a synonym for {AUTO_EXPOSURE_TOKEN!r}",
        param_hint=param_hint,
    )


def resolve_exposure(raw: object, *, param_hint: str) -> float | ExposureChoice:
    """Resolve a raw ``--exposure-time-ms`` / ``--exposure-ms`` value.

    Args:
        raw: The value straight off the dataclass field -- ``str`` from click,
            but possibly a plain number when a caller built the parameter object
            itself (GUI adapters, tests). ``None`` is read as ``auto`` so an
            already-resolved value round-trips.
        param_hint: The flag to attribute the error to, so click renders it as
            ``Invalid value for '--exposure-time-ms'``.

    Returns:
        A strictly positive exposure in ms, :data:`AUTO_EXPOSURE`, or
        :data:`DEFAULT_EXPOSURE` -- three distinguishable answers.

    Raises:
        click.BadParameter: For ``0``, a negative number, a non-finite number,
            empty text or unparseable text. The message always names ``auto``.
    """
    if raw is None:
        return AUTO_EXPOSURE
    if isinstance(raw, ExposureChoice):
        return raw
    if isinstance(raw, str):
        token = raw.strip()
        choice = _TOKENS.get(token.lower())
        if choice is not None:
            return choice
        try:
            # ``float("")`` raises ValueError, so the empty string lands on the
            # rejection below rather than being read as "unset".
            value = float(token)
        except ValueError:
            raise _reject(raw, param_hint) from None
    elif isinstance(raw, bool):
        # ``bool`` is an ``int`` subclass: without this, ``True`` would silently
        # become a 1 ms exposure.
        raise _reject(raw, param_hint)
    elif isinstance(raw, (int, float)):
        value = float(raw)
    else:
        raise _reject(raw, param_hint)

    if not math.isfinite(value) or value <= 0.0:
        raise _reject(raw, param_hint)
    return value


def exposure_ms_legacy(value: float | ExposureChoice) -> float:
    """Collapse a resolved exposure onto the legacy float rung.

    ``auto`` and ``default`` both become ``0.0``, because that is what every
    float API downstream has always taken for "no fixed value here" -- which is
    what keeps ``auto`` bit-compatible with the ``0`` it replaces.

    A caller that can act on ``default`` separately (today: only
    ``panel_locate.scan_panel``) must branch on
    ``value is DEFAULT_EXPOSURE`` **before** calling this, or the distinction
    is lost. See the module docstring for why the other sites cannot.
    """
    return 0.0 if isinstance(value, ExposureChoice) else float(value)