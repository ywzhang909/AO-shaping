"""Characterization tests for the ``tools/slm`` CLI ↔ dataclass binding contract.

Why this file exists
--------------------
``runners/runner_common.with_params`` builds its dataclass by *popping known
field names* out of the ``kwargs`` the click parser produced
(``runner_common.py:305-318``)::

    for _fld, param, _field_type, _delayed in options:
        if param in kwargs:
            obj_kwargs[param] = kwargs.pop(param)
    ...
    kwargs[kw_name] = arg_class(**obj_kwargs)

If a field is **missing** from ``kwargs``, ``arg_class(**obj_kwargs)`` quietly
falls back to that field's dataclass default.  ``_collect_click_annotations``
does raise on a duplicate parameter name (R3 in ``tools/slm/TODO.md``), but
only *within a single class tree*: ``seen`` is created fresh per call, so two
stacked ``@with_params`` decorators never see each other.  The failure mode is
therefore a flag that is documented in ``--help``, accepted by the parser, and
then **silently ignored**.

The four tests below pin the ``tools/slm`` CLI ↔ dataclass binding contract:

1. :func:`test_no_field_equals_its_default` — the R3 guard.  Every one of the
   17 click commands in ``tools/slm`` is invoked with a fixed argv of
   **non-default** values; no delivered field may still equal its default.
   No hardware is opened: every device-construction symbol is monkeypatched to
   raise a sentinel, and the sentinel derives from :class:`BaseException` so a
   broad ``except Exception`` in a command body cannot swallow it.
2. :func:`test_sweep_cli_namespace_is_pinned` — freezes the sweep probe's
   21-option click namespace (taken while it was still ``argparse``) so its
   click migration can be proven equivalent.
3. :func:`test_santec_and_camera_call_sites_are_stable` — freezes the number
   of ``Santec(`` / camera-open construction sites in the package, making
   ``TODO.md`` §六 ("do **not** unify device sessions into a context manager")
   machine-enforceable: settle / slot / dark-frame steps are position
   sensitive, and moving them yields plausible-looking *wrong* data (a
   measured 3.3x slope error).
4. :func:`test_params_module_does_not_import_runners` — keeps the shared
   ``tools/slm/params.py`` groups free of any ``ao_shaping.runners`` edge.
   ``runners/slm/zernike_matrix_runner.py`` imports ``tools.slm.*`` at module
   level, so such an edge closes a real import cycle; ``with_params`` plumbing is
   pure metadata and must not depend on hardware orchestration.
"""

from __future__ import annotations

import ast
import importlib
import re
import tokenize
from dataclasses import MISSING, dataclass, fields, is_dataclass, make_dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from click import Command
from click.testing import CliRunner
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[4]
TOOLS_SLM_DIR = REPO_ROOT / "src" / "ao_shaping" / "tools" / "slm"


# ---------------------------------------------------------------------------
# Hardware blocking
# ---------------------------------------------------------------------------


class _HardwareBlocked(BaseException):
    """Sentinel raised by every patched device-construction symbol.

    Deliberately a :class:`BaseException`: several ``tools/slm`` command bodies
    wrap their device opens in ``except Exception`` (e.g. ``phase_capture.py``
    catches and continues without the SLM), which would let a normal
    ``Exception`` subclass escape the interception and let the command run on.
    ``CliRunner.invoke`` only catches ``SystemExit`` and ``Exception``, so this
    propagates out of the runner untouched.
    """


def _blocked(*_args: Any, **_kwargs: Any) -> Any:
    raise _HardwareBlocked("tools/slm device construction blocked by test")


#: ``(module path, attribute)`` pairs patched at their *definition* site, which
#: is what function-local (deferred) ``from ... import Santec`` statements read.
_DEVICE_SOURCES: tuple[tuple[str, str], ...] = (
    ("ao_shaping.drivers.slm.santec", "Santec"),
    ("ao_shaping.drivers.slm", "Santec"),
    ("ao_shaping.drivers.ccd.miicam.driver", "MIICamera"),
    ("ao_shaping.drivers.ccd.daheng.driver", "DahengCamera"),
    ("ao_shaping.drivers.wfs.thorlab_wfs", "ThorlabWFS"),
    ("ao_shaping.drivers.ccd.common", "create_camera"),
    ("ao_shaping.utils.hardware_utils", "open_camera"),
)

#: Symbol names rebound in each tool module's own namespace.
_DEVICE_NAMES: frozenset[str] = frozenset(
    {
        "Santec",
        "MIICamera",
        "DahengCamera",
        "ThorlabWFS",
        "create_camera",
        "open_camera",
    }
)


def _block_device_opens(monkeypatch: pytest.MonkeyPatch, module: ModuleType) -> None:
    """Rebind every device-construction symbol reachable from ``module``."""
    for path, attr in _DEVICE_SOURCES:
        try:
            source = importlib.import_module(path)
        except ImportError:  # pragma: no cover - optional SDK-dependent module
            continue
        if hasattr(source, attr):
            monkeypatch.setattr(source, attr, _blocked)
    for name in _DEVICE_NAMES:
        if hasattr(module, name):
            monkeypatch.setattr(module, name, _blocked)


# ---------------------------------------------------------------------------
# Test 1 — the R3 guard
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    """One click command plus a fixed argv of non-default values."""

    module: str
    attr: str
    argv: tuple[str, ...]
    #: ``"device"`` -> abort at the first device open (default).
    #: ``"delegate:<name>"`` -> stub a module-level delegate instead (used when
    #: a command has an offline branch that would otherwise divert).
    #: ``"offline"`` -> the command's first branch *returns* (e.g. ``--no-hw``),
    #: leaving no device open to intercept; require exit 0 instead.
    stop: str = "device"
    #: Fields that provably cannot hold a non-default value on the hardware
    #: path; each is structurally re-verified by the test.
    exempt: frozenset[str] = frozenset()


# NOTE: every argv entry below differs from the module's declared default.
# Boolean flags are passed with their non-default polarity (``--no-save`` for a
# ``default=False`` flag, ``--skip-first``/``--no-skip-first`` for a paired
# flag); ``click.Choice``/``click.IntRange``/``click.FloatRange`` bounds are
# respected.  ``{out}``/``{out2}``/``{out3}``/``{resume}`` are substituted with
# paths under pytest's ``tmp_path`` so no command writes into the repo.
CASES: tuple[Case, ...] = (
    Case(
        "calibration",
        "main_shift_calib",
        (
            "--slm-number", "3",
            "--slm-wavelength", "1064",
            "--wfs-exposure-ms", "5.5",
            "--zernike-radius", "450.0",
            "--defocus-a", "15.0",
            "--shift-limit", "400",
            "--coarse-step", "80",
            "--coarse-half", "250",
            "--fine-half", "30",
            "--iterations", "3",
            "--n-avg-scan", "4",
            "--n-avg-verify", "6",
            "--improve-ratio", "0.6",
            "--radius-scan",
            "--no-save",
            "-o", "{out}",
        ),
    ),
    Case(
        "calibration",
        "main",
        (
            "--out-calib", "{out}",
            "--calib", "{out2}",
            "--skip-align",
            "--skip-beam",
            "--verify-only",
            "--align-margin", "5.5",
            "--align-min-window", "96",
            "--exposure-ms", "2.4",
            "--settle-s", "0.5",
        ),
    ),
    Case(
        "gray_response",
        "run",
        (
            "-N", "3",
            "--exposure-ms", "1.5",
            "-o", "{out}",
            "--slm-number", "2",
            "--miicam-id", "1",
            "--wavelength", "532",
            "--wait-time-s", "0.7",
            "--n-sample", "2",
            "--no-skip-first",
            "--discard-count", "2",
            "--bit-depth", "16",
        ),
    ),
    Case(
        "phase_capture",
        "run",
        (
            "--mode", "zernike",
            "-n", "3",
            "-o", "{out}",
            "--slm-number", "2",
            "--wavelength", "532",
            "--memory-slot", "5",
            "--cn2", "2.5e-14",
            "-L", "1500.0",
            "--n-max", "8",
            "--max-coeff", "2.0",
            "--zernike-radius", "400.0",
            "--daheng-id", "1",
            "--daheng-exposure", "25",
            "--miicam-id", "1",
            "--miicam-exposure", "2.0",
            "--miicam-bit-depth", "12",
            "--n-sample", "2",
            "--interval", "0.75",
            "--seed", "1234",
            "-f", "{resume}",
        ),
        # ``skip_first`` defaults True and has no ``--no-skip-first`` counterpart.
        # ``no_daheng``/``no_miicam``/``no_slm`` each *gate* the device open they
        # would otherwise abort at, so passing them would make the hardware
        # sentinel unreachable; all four are re-verified structurally.
        exempt=frozenset({"skip_first", "no_daheng", "no_miicam", "no_slm"}),
    ),
    Case(
        "slm_beam_extent",
        "main",
        (
            "--slm-number", "3",
            "--slm-wavelength", "532",
            "--cam-type", "miicam",
            "--cam-id", "1",
            "--exposure-ms", "4.5",
            "--axis", "y",
            "--steps", "8",
            "--frames", "3",
            "--repeats", "3",
            "--core-radius", "30.0",
            "--seed", "99",
        ),
    ),
    Case(
        "slm_diagnose",
        "main",
        (
            "--slm-number", "2",
            "--slm-wavelength", "532",
            "--cam-id", "1",
            "--camera-type", "daheng",
            "--period-ref", "48",
            "--period-test", "24",
            "--exposure-ms", "1.5",
            "--settle-s", "0.5",
            "--step", "freeze",
            "-o", "{out}",
        ),
    ),
    Case(
        "slm_exposure_check",
        "main",
        (
            "--cam-type", "miicam",
            "--cam-id", "1",
            "--exposure-ms", "5.5",
            "--n-grabs", "5",
            "--grab-delay-s", "0.2",
        ),
    ),
    Case(
        "slm_lut_runner",
        "run",
        (
            "--method", "offset",
            "--period-ref", "48",
            "--period-test", "24",
            "--gray-step", "8",
            "--exposure-ms", "0.05",
            "--n-frames", "4",
            "--camera-type", "daheng",
            "--cam-id", "1",
            "--settle-time", "0.5",
            "--slm-number", "2",
            "--slm-wavelength", "532",
            "--spot-window", "31",
            "--bright-floor", "0.05",
            "--saturation-stop", "0.8",
            "-o", "{out}",
            "--display",
        ),
    ),
    Case(
        "slm_panel_locate",
        "main",
        (
            "--slm-number", "3",
            "--slm-wavelength", "532",
            "--cam-type", "miicam",
            "--cam-id", "1",
            "--exposure-ms", "4.5",
            "--patch-radius", "300",
            "--grid-xs", "100,500,900,1300,1700",
            "--grid-ys", "200,460,700,960",
            "--frames", "3",
            "--seed", "13",
        ),
    ),
    Case(
        "slm_phase_resolution",
        "main",
        (
            "--slm-number", "3",
            "--slm-wavelength", "532",
            "--cam-type", "miicam",
            "--cam-id", "1",
            "--exposure-ms", "4.5",
            "--zernike-radius", "320",
            "--pupil-center", "900,640",
            "--orders", "3,6,10",
            "--scale", "1.1",
            "--frames", "3",
            "--seed", "17",
        ),
    ),
    Case(
        # ``--render-only`` is the only way to flip ``render_only``, and it
        # diverts to the offline branch *before* any device open, so the
        # hardware sentinel would never fire.  Stub the offline delegate
        # instead: the parsed values are still delivered to the command body.
        "slm_phase_response",
        "main",
        (
            "--probe", "defocus",
            "--slm-number", "3",
            "--slm-wavelength", "532",
            "--cam-id", "1",
            "--exposure-ms", "0.05",
            "--n-sample", "4",
            "--slot-min", "3",
            "--slot-max", "120",
            "--settle-s", "0.6",
            "-o", "{out}",
            "--render-only",
        ),
        stop="delegate:_render_only",
    ),
    Case(
        "slm_tilt_probe",
        "main",
        (
            "--slm-number", "3",
            "--slm-wavelength", "532",
            "--cam-type", "miicam",
            "--cam-id", "1",
            "--exposure-ms", "4.5",
            "--periods", "360,180,90",
            "--axis", "y",
            "--frames", "3",
            "--repeat", "3",
        ),
    ),
    Case(
        "slm_wfs_probe",
        "main",
        (
            "--slm-number", "3",
            "--wavelength", "1064",
            "--mla-index", "768",
            "--exposure-ms", "2.5",
            "--no-save",
            "-o", "{out}",
        ),
    ),
    Case(
        "slm_wfs_reference",
        "main",
        (
            "--slm-number", "3",
            "--slm-wavelength", "1064",
            "--wfs-exposure-ms", "5.5",
            "--wfs-order", "8",
            "--zernike-radius", "450.0",
            "--tilt-amps", "0.2,0.4,0.8,1.6",
            "--n-avg", "3",
            "--settle-extra-s", "0.3",
            "--flat-rms-threshold", "0.08",
            "-o", "{out}",
        ),
    ),
    Case(
        "slm_zernike_correction",
        "main",
        (
            "--stage", "matrix",
            "--slm-number", "3",
            "--slm-wavelength", "1064",
            "--wfs-exposure-ms", "5.5",
            "--wfs-order", "8",
            "--n-max", "3",
            "--n-avg", "2",
            "--settle-extra-s", "0.3",
            "--radius-factor", "1.2,2.4",
            "--radii", "150,250",
            "--amps", "3,7",
            "--outlier-factor", "2.5",
            "--coverage-tol", "2",
            "--radius-scan", "130,210,310",
            "--n-iter", "4",
            "--gain", "0.6",
            "--leak", "0.25",
            "--quick",
            "--save-phase",
            "--no-export-correction",
            "--export-dir", "{out2}",
            "-o", "{out}",
        ),
    ),
    Case(
        "slm_zernike_sweep_probe",
        "main",
        (
            "--out", "{out}",
            "--slm-number", "3",
            "--slm-wavelength", "532",
            "--cam-type", "miicam",
            "--cam-id", "2",
            "--exposure-ms", "4.5",
            "--pupil-center", "900,640",
            "--zernike-radius", "320",
            "--sweep-tilt", "-2.0,2.0",
            "--sweep-defocus", "-5.0,-3.0,-1.0,1.0,3.0,5.0",
            "--sweep-astig", "-4.0,-2.0,2.0,4.0",
            "--sweep-coma", "-2.0,-1.0,1.0,2.0",
            "--sweep-spherical", "-2.5,-1.0,1.0,2.5",
            "--sweep-ramps", "96,192,384,768",
            "--frames", "3",
            "--discard", "2",
            "--settle-s", "0.75",
            "--stable-tol", "0.03",
            "--max-wait-s", "8.0",
            "--no-save-frames",
            "--no-hw",
        ),
        stop="offline",
    ),
    Case(
        # ``--verify`` / ``--export-correction`` are mode switches: their first
        # statement in ``main`` is ``return <offline routine>``, so no argv can
        # both set them and still reach the hardware path.  Structurally
        # re-verified by :func:`_diverges_before_device_open`.
        "slm_zernike_response",
        "main",
        (
            "--slm-number", "3",
            "--slm-wavelength", "1064",
            "--wfs-exposure-ms", "5.5",
            "--wfs-order", "8",
            "--n-max", "3",
            "--zernike-radius", "250.0",
            "--amplitude-rad", "7.0",
            "--n-avg", "3",
            "--n-cycles", "4",
            "--shift-x", "12",
            "--shift-y", "-9",
            "--exclude-tip-tilt",
            "--settle-extra-s", "0.3",
            "-o", "{out}",
            "--out-correction", "{out2}",
            "--w-file", "{out3}",
            "--export-shift",
            "--export-shift-x", "5",
            "--export-shift-y", "-4",
        ),
        exempt=frozenset({"verify_path", "export_h5"}),
    ),
)

#: Number of ``@click.command``-decorated functions the guard must cover.
EXPECTED_COMMAND_COUNT = 17

#: Construction symbols counted by test 3.
_SANTEC_RE = re.compile(r"(?<![\w.])Santec\s*\(")
_CAMERA_RE = re.compile(
    r"(?<![\w.])(?:MIICamera|DahengCamera|create_camera|open_camera)\s*\("
)


def _substitute(argv: tuple[str, ...], tmp_path: Path) -> list[str]:
    """Replace ``{out}``-style placeholders with paths under ``tmp_path``."""
    resume = tmp_path / "resume"
    resume.mkdir(parents=True, exist_ok=True)
    (resume / "global_metadata.json").write_text("{}", encoding="utf-8")
    mapping = {
        "{out}": str(tmp_path / "out"),
        "{out2}": str(tmp_path / "out2"),
        "{out3}": str(tmp_path / "out3"),
        "{resume}": str(resume / "global_metadata.json"),
    }
    out: list[str] = []
    for token in argv:
        for key, value in mapping.items():
            token = token.replace(key, value)
        out.append(token)
    return out


def _discover_click_commands() -> set[tuple[str, str]]:
    """``(module, attr)`` of every module-level ``@click.command`` in the package.

    Statically parsed (never imported) so a newly added command cannot slip
    past the guard table unnoticed.
    """
    found: set[tuple[str, str]] = set()
    for path in sorted(TOOLS_SLM_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            for dec in node.decorator_list:
                target = dec.func if isinstance(dec, ast.Call) else dec
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr in {"command", "group"}
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "click"
                ):
                    found.add((path.stem, node.name))
    return found


def _still_at_default(obj: Any, prefix: str = "") -> set[str]:
    """Dotted names of dataclass fields whose value still equals its default.

    Recurses into nested dataclass instances so a ``ClickGroup`` child counts
    once per *leaf* that is still at its declared default.
    """
    out: set[str] = set()
    for fld in fields(obj):
        value = getattr(obj, fld.name)
        name = f"{prefix}{fld.name}"
        if is_dataclass(value) and not isinstance(value, type):
            out.update(_still_at_default(value, prefix=f"{name}."))
        elif fld.default is not MISSING and value == fld.default:
            out.add(name)
    return out


def _tested_names(test: ast.AST) -> set[str]:
    """Names an ``if`` test reads, as bare params *or* ``params.field`` attributes.

    ``with_params`` hands the callback a single dataclass, so a migrated body
    tests ``params.no_daheng`` -- an :class:`ast.Attribute` -- rather than a bare
    ``no_daheng`` (:class:`ast.Name`). Both spellings must resolve, otherwise
    every *structural* exemption (``gates_open`` / ``diverges_before_device_open``)
    silently stops being detected the moment its command is migrated, and the
    honesty check then fails on a field that is still legitimately exempt.
    """
    return {n.id for n in ast.walk(test) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(test) if isinstance(n, ast.Attribute)
    }


def _diverges_before_device_open(module_name: str, attr: str, field: str) -> bool:
    """True if ``attr`` returns/raises on ``field`` before its first device open."""
    source = TOOLS_SLM_DIR / f"{module_name}.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    device_names = {n for _, n in _DEVICE_SOURCES} | _DEVICE_NAMES
    for node in tree.body:
        if not (isinstance(node, ast.FunctionDef) and node.name == attr):
            continue
        device_stmt: int | None = None
        for index, stmt in enumerate(node.body):
            if any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id in device_names
                for call in ast.walk(stmt)
            ):
                device_stmt = index
                break
        for index, stmt in enumerate(node.body):
            if not isinstance(stmt, ast.If) or device_stmt is None:
                continue
            if index >= device_stmt:
                continue
            tested = _tested_names(stmt.test)
            if field not in tested:
                continue
            if any(
                isinstance(inner, (ast.Return, ast.Raise))
                for inner in ast.walk(stmt)
            ):
                return True
    return False


def _gates_device_open(module_name: str, attr: str, field: str) -> bool:
    """True if ``attr``'s device open is nested inside an ``if`` testing ``field``.

    Such a field cannot be given a non-default argv value: flipping it to True
    skips the very open the hardware sentinel relies on, so the command would
    run on without ever reaching the blocked device.
    """
    source = TOOLS_SLM_DIR / f"{module_name}.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    device_names = {n for _, n in _DEVICE_SOURCES} | _DEVICE_NAMES
    for node in tree.body:
        if not (isinstance(node, ast.FunctionDef) and node.name == attr):
            continue
        for stmt in node.body:
            if not isinstance(stmt, ast.If):
                continue
            tested = _tested_names(stmt.test)
            if field not in tested:
                continue
            for inner in stmt.body:
                if any(
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id in device_names
                    for call in ast.walk(inner)
                ):
                    return True
    return False


@pytest.mark.parametrize("case", CASES, ids=[f"{c.module}.{c.attr}" for c in CASES])
def test_no_field_equals_its_default(
    case: Case, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R3 guard: a fixed non-default argv must reach *every* declared field.

    The command is invoked through the real ``CliRunner``, so the value under
    test is what the click parser actually produced — the very ``kwargs`` that
    ``with_params`` pops field names out of.  A field that silently falls back
    to its dataclass default shows up here as a name in
    ``still_at_default``.
    """
    module = importlib.import_module(f"ao_shaping.tools.slm.{case.module}")
    command = getattr(module, case.attr)
    assert isinstance(command, Command), f"{case.module}.{case.attr} is not a Command"

    param_names = [p.name for p in command.params]
    assert len(param_names) == len(set(param_names)), "duplicate click param names"

    delivered: dict[str, Any] = {}
    delegate_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    real_callback = command.callback
    assert real_callback is not None

    def _spy(*args: Any, **kwargs: Any) -> Any:
        """Record what the parser delivered, then run the real callback.

        Calling through (rather than short-circuiting) matters: after the
        ``with_params`` migration the assembly lives *inside* this callback, so
        short-circuiting would skip the very code the guard protects.
        """
        delivered["args"] = args
        delivered["kwargs"] = dict(kwargs)
        return real_callback(*args, **kwargs)

    monkeypatch.setattr(command, "callback", _spy)

    if case.stop == "device":
        _block_device_opens(monkeypatch, module)
        expected: Any = _HardwareBlocked
    elif case.stop == "offline":
        # Still block device opens: if the early-return branch were deleted, the
        # sentinel must surface instead of the command reaching real hardware.
        _block_device_opens(monkeypatch, module)
        expected = None
    else:
        _, delegate_name = case.stop.split(":", 1)
        assert hasattr(module, delegate_name), f"missing delegate {delegate_name}"

        def _delegate(*args: Any, **kwargs: Any) -> Any:
            delegate_calls.append((args, kwargs))
            return {"verdict": {}}

        monkeypatch.setattr(module, delegate_name, _delegate)
        expected = None

    argv = _substitute(case.argv, tmp_path)

    if expected is None:
        result = CliRunner().invoke(command, argv, standalone_mode=False)
        assert result.exit_code == 0, (
            f"{case.module}.{case.attr} exited {result.exit_code}: "
            f"{result.exception!r}\n{result.output}"
        )
        if case.stop != "offline":
            assert delegate_calls, f"delegate {case.stop!r} was never called"
    else:
        blocked = False
        result = None
        try:
            result = CliRunner().invoke(command, argv, standalone_mode=False)
        except _HardwareBlocked:
            blocked = True
        assert blocked, (
            f"{case.module}.{case.attr}: argv never reached a blocked device open "
            f"(exit_code={getattr(result, 'exit_code', None)}, "
            f"exception={getattr(result, 'exception', None)!r}). A non-zero exit_code "
            f"here means click rejected the argv: a @click.option was renamed, "
            f"removed, or its name no longer matches its dest.\nargv={argv}\n"
            f"{getattr(result, 'output', '')}"
        )

    assert delivered, "command callback was never reached"
    kwargs = delivered["kwargs"]

    # Post-migration shape: a single dataclass instance delivered under the
    # ``kw_name``.  Pre-migration shape: one flat kwarg per declared option.
    if (
        len(kwargs) == 1
        and is_dataclass(next(iter(kwargs.values())))
        and not isinstance(next(iter(kwargs.values())), type)
    ):
        config = next(iter(kwargs.values()))
        assert is_dataclass(config), "with_params must deliver a dataclass"
        logger.debug(f"{case.module}.{case.attr}: dataclass delivery")
        still = _still_at_default(config)
    else:
        missing = sorted(set(param_names) - set(kwargs))
        assert not missing, (
            f"{case.module}.{case.attr}: fields never delivered: {missing}"
        )
        # A shadow dataclass whose *declared defaults* come from the click
        # options and whose *values* come from the delivered kwargs: the
        # ``dataclasses.fields()`` comparison the guard is specified in.
        shadow_cls = make_dataclass(
            f"{case.module}_{case.attr}_shadow",
            [(p.name, object, p.default) for p in command.params],
            namespace={"__module__": __name__},
        )
        shadow = shadow_cls()
        for param in command.params:
            setattr(shadow, param.name, kwargs[param.name])
        still = _still_at_default(shadow)

    # The exemption table must stay honest: an exempt field has to be genuinely
    # un-switchable *and* genuinely still at its default, otherwise it should
    # have been given a non-default argv value and dropped from the table.
    for name in sorted(case.exempt):
        assert name in param_names, (
            f"{case.module}.{case.attr}: exempt {name!r} not a param"
        )
        param = next(p for p in command.params if p.name == name)
        flag_only = param.is_flag and not param.secondary_opts and bool(param.default)
        gates_open = _gates_device_open(case.module, case.attr, name)
        diverges = _diverges_before_device_open(case.module, case.attr, name)
        assert flag_only or gates_open or diverges, (
            f"{case.module}.{case.attr}: {name!r} is exempt but is neither an "
            f"unswitchable flag, a device-open gate, nor a pre-device-open mode "
            f"switch — remove it from the exemption table and give it a "
            f"non-default argv value"
        )
        assert name in still, (
            f"{case.module}.{case.attr}: exempt field {name!r} differs from its "
            f"default — it is no longer exempt, drop it from the exemption table"
        )

    guarded = {n for n in still if n.split(".", 1)[0] not in case.exempt}
    assert guarded == set(), (
        f"{case.module}.{case.attr}: fields still equal to their declared "
        f"default after a non-default argv: {sorted(guarded)}\n"
        f"argv={argv}\ndelivered={kwargs}"
    )


def test_every_click_command_has_a_guard_case() -> None:
    """No click command in ``tools/slm`` may escape the R3 guard table."""
    discovered = _discover_click_commands()
    covered = {(case.module, case.attr) for case in CASES}
    assert discovered == covered, (
        "click command inventory drifted from the guard table; "
        f"missing={sorted(discovered - covered)} stale={sorted(covered - discovered)}"
    )
    assert len(covered) == EXPECTED_COMMAND_COUNT


# ---------------------------------------------------------------------------
# Test 2 — the sweep probe's pinned CLI namespace
# ---------------------------------------------------------------------------

SWEEP_ARGV: tuple[str, ...] = (
    "--out", "/tmp/ao_cli_binding_sweep",
    "--slm-number", "3",
    "--slm-wavelength", "532",
    "--cam-type", "miicam",
    "--cam-id", "2",
    "--exposure-ms", "4.5",
    "--pupil-center", "900,640",
    "--zernike-radius", "320",
    "--sweep-tilt=-2.0,2.0",
    "--sweep-defocus=-5.0,-3.0,-1.0,1.0,3.0,5.0",
    "--sweep-astig=-4.0,-2.0,2.0,4.0",
    "--sweep-coma=-2.0,-1.0,1.0,2.0",
    "--sweep-spherical=-2.5,-1.0,1.0,2.5",
    "--sweep-ramps", "96,192,384,768",
    "--frames", "3",
    "--discard", "2",
    "--settle-s", "0.75",
    "--stable-tol", "0.03",
    "--max-wait-s", "8.0",
    "--no-save-frames",
    "--no-hw",
)

EXPECTED_SWEEP_NAMESPACE: dict[str, Any] = {
    "out": "/tmp/ao_cli_binding_sweep",
    "slm_number": 3,
    "slm_wavelength": 532,
    "cam_type": "miicam",
    "cam_id": 2,
    "exposure_ms": 4.5,
    "pupil_center": (900, 640),
    "zernike_radius": 320,
    "sweep_tilt": "-2.0,2.0",
    "sweep_defocus": "-5.0,-3.0,-1.0,1.0,3.0,5.0",
    "sweep_astig": "-4.0,-2.0,2.0,4.0",
    "sweep_coma": "-2.0,-1.0,1.0,2.0",
    "sweep_spherical": "-2.5,-1.0,1.0,2.5",
    "sweep_ramps": "96,192,384,768",
    "frames": 3,
    "discard": 2,
    "settle_s": 0.75,
    "stable_tol": 0.03,
    "max_wait_s": 8.0,
    # click resolves the slash syntax to a real boolean pair, so what argparse
    # spelled as the dest ``"save_frames/__no_save_frames"`` (value ``"0"``) is
    # now the plain flag ``save_frames=False``. The flag *pair* is unchanged.
    "save_frames": False,
    "no_hw": True,
}


def test_sweep_cli_namespace_is_pinned() -> None:
    """Freeze the sweep probe's 21-option click namespace.

    This was the ``argparse`` "before" snapshot taken before the click migration
    (``tools/slm/TODO.md`` R5); it is retained verbatim against the click parser
    so the migration can be proven value-for-value equivalent. The argv sets every
    one of the 21 options to a non-default value, so a silently dropped option
    shows up as a missing/unchanged key. Parsing is pure — no device is touched —
    and ``--no-hw`` is set for good measure.

    One key is *expected* to differ from the old argparse snapshot:
    ``save_frames`` (see :data:`EXPECTED_SWEEP_NAMESPACE`).
    """
    from ao_shaping.tools.slm.slm_zernike_sweep_probe import main

    actual = dict(main.make_context("slm_zernike_sweep_probe", list(SWEEP_ARGV)).params)

    assert set(actual) == set(EXPECTED_SWEEP_NAMESPACE), (
        "click parameter set drifted (an option was added/removed/renamed); "
        f"added={sorted(set(actual) - set(EXPECTED_SWEEP_NAMESPACE))} "
        f"removed={sorted(set(EXPECTED_SWEEP_NAMESPACE) - set(actual))}"
    )
    assert actual == EXPECTED_SWEEP_NAMESPACE

    # Every option must be non-default, otherwise the snapshot is weak.
    defaults = dict(main.make_context("slm_zernike_sweep_probe", []).params)
    assert set(actual) == set(defaults)
    unchanged = sorted(k for k in actual if actual[k] == defaults[k])
    assert unchanged == [], f"sweep argv left these at their default: {unchanged}"


# ---------------------------------------------------------------------------
# Test 3 — device-session call-site census (TODO.md §六)
# ---------------------------------------------------------------------------


def _code_only(path: Path) -> str:
    """Source text with comments and *all* string literals removed."""
    out: list[str] = []
    with open(path, "rb") as handle:
        for tok in tokenize.tokenize(handle.readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            out.append(tok.string)
    return " ".join(out)


def _module_docstring(path: Path) -> str:
    return (
        ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")), clean=False)
        or ""
    )


def _construction_sites(pattern: re.Pattern[str]) -> dict[str, int]:
    """``{file: sites}`` for executable code + the module docstring's usage example.

    Counting convention (reproducible, see the module docstring):
    ``executable construction sites`` (comments/strings stripped) **plus** the
    occurrences inside a module docstring — which is where
    ``slm_snr_probe``'s ``Typical use::`` example documents how a caller opens
    the bench.  Function docstrings and prose are excluded, so a docstring edit
    cannot move the numbers.
    """
    counts: dict[str, int] = {}
    for path in sorted(TOOLS_SLM_DIR.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        hits = len(pattern.findall(_code_only(path)))
        hits += len(pattern.findall(_module_docstring(path)))
        if hits:
            counts[str(path.relative_to(TOOLS_SLM_DIR))] = hits
    return counts


def test_santec_and_camera_call_sites_are_stable() -> None:
    """Freeze the per-file device-session census.

    ``tools/slm/TODO.md`` §六 forbids unifying these sessions behind a
    ``with_slm_bench()`` context manager: settle / slot / dark-frame steps are
    **position sensitive**, and reordering them yields plausible-looking wrong
    data (R1: a measured 3.3x slope error).  Nothing in the suite enforces that,
    so freeze the census here — any refactor that moves or adds a device open
    must come here first.
    """
    santec = _construction_sites(_SANTEC_RE)
    camera = _construction_sites(_CAMERA_RE)

    # 18 -> 21 and 15 -> 18 when slm_abba_probe / slm_drift_probe / slm_floor_probe
    # were added upstream. This guard pins the census so a *refactor* cannot quietly
    # consolidate device sessions; adding a genuinely new probe is a legitimate
    # change and must be re-baselined here deliberately.
    assert sum(santec.values()) == 21, f"Santec( construction sites changed: {santec}"
    assert sum(camera.values()) == 18, f"camera-open call sites changed: {camera}"

    assert santec == {
        "calibration.py": 2,
        "cartographer/slm_cartographer_ui.py": 1,
        "gray_response.py": 1,
        "phase_capture.py": 1,
        "slm_abba_probe.py": 1,
        "slm_beam_extent.py": 1,
        "slm_diagnose.py": 1,
        "slm_drift_probe.py": 1,
        "slm_floor_probe.py": 1,
        "slm_lut_runner.py": 1,
        "slm_panel_locate.py": 1,
        "slm_phase_resolution.py": 1,
        "slm_phase_response.py": 1,
        "slm_snr_probe.py": 1,
        "slm_tilt_probe.py": 1,
        "slm_wfs_probe.py": 1,
        "slm_wfs_reference.py": 1,
        "slm_zernike_correction.py": 1,
        "slm_zernike_response.py": 1,
        "slm_zernike_sweep_probe.py": 1,
    }, f"Santec( per-file census drifted: {santec}"

    assert camera == {
        "calibration.py": 1,
        "gray_response.py": 1,
        "phase_capture.py": 2,
        "slm_abba_probe.py": 1,
        "slm_beam_extent.py": 1,
        "slm_diagnose.py": 1,
        "slm_drift_probe.py": 1,
        "slm_exposure_check.py": 1,
        "slm_floor_probe.py": 1,
        "slm_lut_runner.py": 2,
        "slm_panel_locate.py": 1,
        "slm_phase_resolution.py": 1,
        "slm_phase_response.py": 1,
        "slm_snr_probe.py": 1,
        "slm_tilt_probe.py": 1,
        "slm_zernike_sweep_probe.py": 1,
    }, f"camera-open per-file census drifted: {camera}"

    # ``slm_snr_probe`` is the one module that is *not* a CLI: it exposes no
    # click command and constructs no device, so every caller keeps full control
    # of the settle/slot/dark-frame ordering (the R1 hazard).  Its
    # ``Santec(`` / ``create_camera(`` occurrences are documentation only.
    snr_path = TOOLS_SLM_DIR / "slm_snr_probe.py"
    snr_source = snr_path.read_text(encoding="utf-8")
    assert "import click" not in snr_source, "slm_snr_probe must stay CLI-free"
    assert "@click.command" not in snr_source
    assert not _SANTEC_RE.search(_code_only(snr_path)), (
        "slm_snr_probe must not construct a Santec — it is device-agnostic by "
        "design and takes already-opened instances"
    )
    assert not _CAMERA_RE.search(_code_only(snr_path)), (
        "slm_snr_probe must not open a camera — it is device-agnostic by "
        "design and takes already-opened instances"
    )
    assert _SANTEC_RE.search(_module_docstring(snr_path)), (
        "slm_snr_probe's documented 'Typical use' example must keep showing how "
        "a caller opens the bench"
    )


# ---------------------------------------------------------------------------
# Test 4 — shared params module stays free of runners (TODO.md R5)
# ---------------------------------------------------------------------------


def test_params_module_does_not_import_runners() -> None:
    """Guard the layering invariant documented in ``tools/slm/params.py``.

    ``runners/slm/zernike_matrix_runner.py`` imports ``ao_shaping.tools.slm.*``
    at module level, so any ``ao_shaping.runners`` edge in the shared parameter
    groups closes a real import cycle. ``with_params`` plumbing is pure
    metadata and must never reach for hardware orchestration.
    """
    params_path = TOOLS_SLM_DIR / "params.py"
    tree = ast.parse(params_path.read_text(encoding="utf-8"))

    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            offenders += [
                alias.name for alias in node.names
                if alias.name.startswith("ao_shaping.runners")
            ]
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.startswith("ao_shaping.runners") or (
                node.level and module.startswith("runners")
            ):
                offenders.append(module)

    assert not offenders, (
        "tools/slm/params.py must not import ao_shaping.runners "
        f"(would close the runners -> tools.slm import cycle): {sorted(set(offenders))}"
    )
