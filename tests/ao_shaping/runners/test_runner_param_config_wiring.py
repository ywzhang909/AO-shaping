"""R-35 — every CLI params field must actually reach its optimizer ``Config``.

A params dataclass field is a promise: the user typed a flag, so the value must
land in the ``Config`` the optimizer reads. Nothing enforces that promise, and
the three ways it breaks are all *silent*:

1. **Wrong family.** A field added to ``SlmGsRefineParams`` but wired into
   ``build_model_in_loop_config``. The two families share option names on
   purpose (``--far-field-padding``, ``--pupil-center``, ``--target-side``), so
   a field dropped into the neighbouring builder can look plausible and even
   share a name with a real config field.
2. **Never wired.** The flag parses, the field is populated, the builder just
   does not pass it. The run then silently uses the config default.
3. **Hardcoded or mis-sourced value.** ``far_field_padding=3`` or
   ``far_field_padding=params.pupil_center`` — the right kwarg name carrying
   the wrong number, which no signature check can see.

``tests/ao_shaping/optimizer/wfless/test_slm_model_in_loop.py::TestCliLibraryParity``
already models the *idea* of a CLI-only / library-only split, but it does not
derive the sets: ``test_shared_field_names_are_a_valid_mapping`` computes
``config_fields & param_fields`` and then asserts the result is a subset of both
operands, which is true by construction and therefore cannot fail. The curated
``LIBRARY_ONLY_FIELDS`` / ``CLI_ONLY_FIELDS`` beside it drifted silently — they
still omit ``debug_dir`` and the five ``panel_scan*`` fields added alongside it,
and no assertion compares them against reality.

This module closes that gap by **deriving** the wiring from the source AST and
comparing it against a curated split that has to be updated *deliberately*. The
derivation is static, so it is race-free and needs no device: ``runner_common``
only imports driver modules at module scope, and constructing a config opens
nothing.

No hardware, no torch, no network, no files written.
"""

from __future__ import annotations

import ast
import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import ao_shaping.runners.runner_common as runner_common
import ao_shaping.runners.slm.model_in_loop_runner as model_in_loop_runner
from ao_shaping.optimizer.wfless.slm_gs_refine import SlmGsRefineConfig
from ao_shaping.optimizer.wfless.slm_model_in_loop import SlmModelInLoopConfig
from ao_shaping.runners.runner_common import (
    SlmGsRefineParams,
    SlmModelInLoopParams,
    build_gs_refine_config,
    build_model_in_loop_config,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_SCALARS = (int, float, str, bool, type(None))


@dataclasses.dataclass(frozen=True)
class _Family:
    """One ``params -> builder -> Config`` chain plus its curated exceptions."""

    label: str
    params: type
    config: type
    builder: Callable[..., Any]
    builder_name: str

    #: Modules allowed to consume this family's CLI-only fields. The set is
    #: derived from ``params - builder kwargs`` rather than hand-listed, so it
    #: tracks whichever branch the suite runs on -- the same guard has to hold
    #: whether or not a given family has gained a preflight flag yet.
    consumers: tuple[type, ...]

    #: Fields whose CLI default is a deliberate sentinel rather than a value:
    #: ``slm_wavelength`` defaults to ``0`` ("ask the device") while the Config
    #: defaults to a concrete ``1064``, so forwarding it verbatim is the point.
    sentinel_defaults: frozenset[str] = frozenset()

    def cli_only(self) -> set[str]:
        """Param fields the builder does not forward (they are runner-level)."""
        return _field_names(self.params) - set(_config_call_kwargs(self.builder_name))


GS = _Family(
    label="gs-refine",
    params=SlmGsRefineParams,
    config=SlmGsRefineConfig,
    builder=build_gs_refine_config,
    builder_name="build_gs_refine_config",
    # `save_gs_refine_results` decides whether to write best_far_field.png.
    consumers=(runner_common,),
)

MIL = _Family(
    label="model-in-loop",
    params=SlmModelInLoopParams,
    config=SlmModelInLoopConfig,
    builder=build_model_in_loop_config,
    builder_name="build_model_in_loop_config",
    # The panel scan, when present, is a CLI preflight that runs *before* the
    # optimizer opens its devices, so those fields never belong in the Config;
    # save_best_image is consumed by save_model_in_loop_results.
    consumers=(runner_common, model_in_loop_runner),
    sentinel_defaults=frozenset({"slm_wavelength"}),
)

#: Config fields deliberately unreachable from the CLI. Each is either a
#: callback, a progress knob, or something the library sets on a caller's behalf
#: (``debug_dir`` is injected by the runner via ``dataclasses.replace`` once the
#: debug directory exists; ``step_sink`` is a per-measurement debug hook, which
#: is the same category as ``callback``).
LIBRARY_ONLY: dict[str, set[str]] = {
    GS.label: {
        "callback",
        "gs_relax",
        "lr_min_ratio",
        "progress_every",
        "settle_max_discard",
    },
    MIL.label: {
        "callback",
        "debug_dir",
        "output_subdir",
        "progress_every",
        "sim_aberration",
        "step_sink",
    },
}

FAMILIES = (GS, MIL)


def _config_call_kwargs(builder_name: str) -> dict[str, str]:
    """Return ``{kwarg_name: source}`` for the ``*Config(...)`` call in a builder.

    Read off the AST rather than by calling the builder, so a kwarg whose value
    is a literal is still visible (a call would hide it behind the default).
    """
    tree = ast.parse(Path(runner_common.__file__).read_text(encoding="utf-8"))
    func = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == builder_name
    )
    for node in ast.walk(func):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id.endswith("Config")
        ):
            return {kw.arg: ast.unparse(kw.value) for kw in node.keywords if kw.arg}
    raise AssertionError(f"no *Config(...) call found in {builder_name}()")


def _field_names(cls: type) -> set[str]:
    return {f.name for f in dataclasses.fields(cls)}


def _module_source(module: type) -> str:
    """Return a module's source text (for ``params.<field>`` reachability)."""
    return Path(module.__file__).read_text(encoding="utf-8")


def _defaults(cls: type) -> dict[str, object]:
    return {
        f.name: f.default
        for f in dataclasses.fields(cls)
        if f.default is not dataclasses.MISSING
    }


# ---------------------------------------------------------------------------
# A. the three silent breakages
# ---------------------------------------------------------------------------


class TestWiring:
    """The builder is the only place a CLI value can become physics."""

    @pytest.mark.parametrize("family", FAMILIES, ids=lambda f: f.label)
    def test_builder_passes_only_real_config_fields(self, family: _Family) -> None:
        """Breaks on (1) wrong family: a kwarg the Config does not declare.

        A ``TypeError`` at runtime would catch the obvious case; this pins it
        statically so the failure names the field instead of the traceback.
        """
        unknown = sorted(set(_config_call_kwargs(family.builder_name)) - _field_names(family.config))
        assert not unknown, (
            f"{family.builder_name}() passes {unknown}, which "
            f"{family.config.__name__} does not declare — this is a field wired "
            f"into the wrong Config"
        )

    @pytest.mark.parametrize("family", FAMILIES, ids=lambda f: f.label)
    def test_every_wired_field_reads_its_own_params_field(self, family: _Family) -> None:
        """Breaks on (3): right kwarg name, wrong (or literal) value.

        A builder-local kwarg such as ``record_debug=debug`` is exempt because
        its name is not a params field; everything whose name *is* a params
        field must read ``params.<that same field>``.
        """
        kwargs = _config_call_kwargs(family.builder_name)
        params_fields = _field_names(family.params)
        wrong = {
            name: src
            for name, src in kwargs.items()
            if name in params_fields and f"params.{name}" not in src
        }
        assert not wrong, (
            f"{family.builder_name}() sets {wrong} without reading the matching "
            f"params field — a hardcoded or mis-sourced value"
        )

    @pytest.mark.parametrize("family", FAMILIES, ids=lambda f: f.label)
    def test_params_fields_split_exactly_into_wired_and_cli_only(
        self, family: _Family
    ) -> None:
        """Breaks on (2) never wired: a flag that parses and is then ignored.

        The invariant is *every* param field is either forwarded into the Config
        or read as ``params.<field>`` by one of the family's runner modules --
        a flag that is neither is dead. Deriving the CLI-only set (rather than
        hand-listing it) is what keeps this honest across branches, where the
        same guard has to hold before and after a preflight flag exists.
        """
        sources = [_module_source(m) for m in family.consumers]
        dead = sorted(
            field
            for field in family.cli_only()
            if not any(f"params.{field}" in src for src in sources)
        )
        assert not dead, (
            f"{family.params.__name__} declares {dead}, which "
            f"{family.builder_name}() never passes and no runner module reads "
            f"as params.<field> — a flag that parses and is then ignored"
        )

    @pytest.mark.parametrize("family", FAMILIES, ids=lambda f: f.label)
    def test_config_fields_split_exactly_into_wired_and_library_only(
        self, family: _Family
    ) -> None:
        """A new Config knob is unreachable until the CLI is given a flag for it.

        The inverse of the check above, and the reason ``debug_dir`` went
        missing from the older hand-written ``LIBRARY_ONLY_FIELDS``: adding a
        config field with no CLI route is a decision, so it has to be restated
        here.
        """
        kwargs = set(_config_call_kwargs(family.builder_name))
        unreachable = sorted(_field_names(family.config) - kwargs - LIBRARY_ONLY[family.label])
        assert not unreachable, (
            f"{family.config.__name__} declares {unreachable}, which the CLI "
            f"cannot set — give it a params field or add it to LIBRARY_ONLY"
        )


# ---------------------------------------------------------------------------
# B. the curated exceptions must stay curated
# ---------------------------------------------------------------------------


class TestCuratedExceptions:
    """An exception list is only useful while it is honest."""

    @pytest.mark.parametrize("family", FAMILIES, ids=lambda f: f.label)
    def test_every_cli_only_field_is_read_by_some_runner_module(
        self, family: _Family
    ) -> None:
        """A CLI-only field must still be read somewhere.

        Without this, the CLI-only set becomes a place to hide a dead flag:
        move a field out of the wiring and the wiring test is happy again.
        """
        dead = {
            field: [m.__name__ for m in family.consumers]
            for field in sorted(family.cli_only())
            if not any(
                f"params.{field}" in _module_source(m) for m in family.consumers
            )
        }
        assert not dead, (
            f"{family.label}: {dead} are declared CLI-only but no longer read "
            f"as params.<field> in the module that was supposed to consume them"
        )

    @pytest.mark.parametrize("family", FAMILIES, ids=lambda f: f.label)
    def test_cli_only_and_library_only_do_not_overlap(self, family: _Family) -> None:
        """A name cannot be both unreachable from the CLI and CLI-only."""
        assert not family.cli_only() & LIBRARY_ONLY[family.label]


# ---------------------------------------------------------------------------
# C. values, not just names
# ---------------------------------------------------------------------------


class TestValuesArrive:
    """A correctly-named kwarg carrying the wrong number is still a bug."""

    @pytest.mark.parametrize("family", FAMILIES, ids=lambda f: f.label)
    def test_pure_passthrough_defaults_match_the_config(self, family: _Family) -> None:
        """A field forwarded verbatim must not change its default in transit.

        Only pure passthroughs (``field=params.field``) and scalar defaults are
        compared: a deliberate transform (``pupil_center=parse_point(...)``) or a
        derived value (``focal_length_m=0.0`` meaning "solve it") is not a
        default mismatch, and would otherwise be impossible to express. A field
        listed in ``sentinel_defaults`` is exempt too — ``slm_wavelength``
        forwards ``0`` ("ask the device") precisely *because* it must not
        silently pin a wavelength, so its divergence from the Config default is
        the behaviour, not an accident.
        """
        kwargs = _config_call_kwargs(family.builder_name)
        param_defaults = _defaults(family.params)
        config_defaults = _defaults(family.config)
        mismatched = {
            name: (param_defaults[name], config_defaults.get(name))
            for name, src in kwargs.items()
            if src == f"params.{name}"
            and name not in family.sentinel_defaults
            and name in param_defaults
            and name in config_defaults
            and isinstance(param_defaults[name], _SCALARS)
            and isinstance(config_defaults[name], _SCALARS)
            and param_defaults[name] != config_defaults[name]
        }
        assert not mismatched, (
            f"{family.builder_name}() forwards {mismatched} verbatim, so the CLI "
            f"default silently overrides a different Config default"
        )

    @pytest.mark.parametrize("family", FAMILIES, ids=lambda f: f.label)
    def test_builder_constructs_from_default_params(self, family: _Family) -> None:
        """The whole chain runs on defaults, with no device opened.

        Catches a kwarg that only type-checks against a Config the builder does
        not actually construct, and any ``__post_init__`` validation the CLI
        defaults would trip.
        """
        config = family.builder(family.params(), debug=False)
        assert isinstance(config, family.config)


# ---------------------------------------------------------------------------
# D. the regression that started this
# ---------------------------------------------------------------------------


class TestFarFieldPaddingStaysOnItsOwnFamily:
    """``--far-field-padding`` is exposed by *both* commands, on purpose.

    Both optimizers need it and the calibrated defaults genuinely differ
    (GS 3, model-in-loop 8), so a blanket "this option belongs to one family"
    rule would be wrong. What must hold is that each family reads its *own*
    value and that the value cannot drift between the CLI and the Config it
    feeds -- a silent swap here rescales the far field and every metric with it.
    """

    def test_each_family_exposes_and_forwards_its_own_padding(self) -> None:
        for family, expected in ((GS, 3), (MIL, 8)):
            params_default = _defaults(family.params)["far_field_padding"]
            config_default = _defaults(family.config)["far_field_padding"]
            assert params_default == expected, (
                f"{family.label} CLI --far-field-padding default moved to "
                f"{params_default} (expected {expected})"
            )
            assert config_default == expected, (
                f"{family.config.__name__}.far_field_padding default is "
                f"{config_default}, no longer matching the CLI ({expected})"
            )
            source = _config_call_kwargs(family.builder_name)["far_field_padding"]
            assert source == "params.far_field_padding", (
                f"{family.builder_name}() sets far_field_padding={source}"
            )

    def test_each_builder_routes_padding_to_its_own_config(self) -> None:
        """``SlmGsRefineConfig`` never sees 8, and vice versa."""
        assert build_gs_refine_config(SlmGsRefineParams(), debug=False).far_field_padding == 3
        assert (
            build_model_in_loop_config(SlmModelInLoopParams(), debug=False).far_field_padding
            == 8
        )
