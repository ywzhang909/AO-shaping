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
import ao_shaping.runners.slm.model_in_loop_session as model_in_loop_session
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

    #: Param fields with no ``Config`` counterpart, mapped to the module that
    #: must consume them. Keyed by field so the entry cannot outlive its use.
    cli_only: dict[str, type]


GS = _Family(
    label="gs-refine",
    params=SlmGsRefineParams,
    config=SlmGsRefineConfig,
    builder=build_gs_refine_config,
    builder_name="build_gs_refine_config",
    # `save_gs_refine_results` decides whether to write best_far_field.png.
    cli_only={"save_best_image": runner_common},
)

MIL = _Family(
    label="model-in-loop",
    params=SlmModelInLoopParams,
    config=SlmModelInLoopConfig,
    builder=build_model_in_loop_config,
    builder_name="build_model_in_loop_config",
    # The panel scan is a preflight that runs *before* the optimizer opens its
    # devices, so those five never belong in the Config. It is consumed by the
    # shared session module (which the GUI drives too), not by the click runner.
    cli_only={
        "save_best_image": runner_common,
        "panel_scan": model_in_loop_session,
        "panel_scan_step_px": model_in_loop_session,
        "panel_scan_radius_px": model_in_loop_session,
        "panel_scan_max_flat_drift": model_in_loop_session,
        "panel_scan_max_session_drift": model_in_loop_session,
        "panel_scan_max_session_shift_px": model_in_loop_session,
    },
)

#: Config fields deliberately unreachable from the CLI. Each is either a
#: callback, a progress knob, or something the library sets on a caller's behalf
#: (``debug_dir`` is injected by the runner via ``dataclasses.replace`` once the
#: debug directory exists). ``send_overlay`` (both families) is filled by the GUI
#: session via ``build_send_overlay``; the click runner leaves it ``None``. The
#: three ``time_budget_s`` / ``chunk_epochs`` / ``resume_state`` fields are
#: consumed by the chunk scheduler ``scripts/slm/run_gs_refine_zernike_sweep.py``
#: (per-block ``--chunk-epochs`` / ``--time-budget-s``), not by the
#: ``slm-gs-refine`` click runner.
LIBRARY_ONLY: dict[str, set[str]] = {
    GS.label: {
        "callback",
        "chunk_epochs",
        "gs_relax",
        "lr_min_ratio",
        "phase_disturbance_file",
        "progress_every",
        "resume_state",
        "send_overlay",
        "settle_max_discard",
        "time_budget_s",
    },
    MIL.label: {
        "adaptive_trust_region",
        "callback",
        "debug_dir",
        # 库调用者直接塞 ModelErrorTerms 对象; CLI 走 --error-params (str 路径)
        # 由 __post_init__ 解析填充, 不走 builder 的 kwargs。
        "error_terms",
        # 焦面补偿前置阶段 (focus_preflight, feb4d292 引入): 由 HW runbook 自己的
        # argparse + build_focus_preflight 设置 (设备打开前运行), 与 panel_scan 同类,
        # 不走通用 click builder; 默认全关, 现有 run 的位级行为保持不变。
        "focus_compensation",
        "focus_dir",
        "focus_live_probe",
        "holdout_on_sim",
        "output_subdir",
        "probe_split_hook",
        "progress_every",
        "send_overlay",
        "sim_aberration",
        "step_sink",
        "uncertainty_on_rejection",
        "uncertainty_probe_diversity",
        "uncertainty_probe_ridge",
        "uncertainty_probe_samples",
        "uncertainty_probe_steps",
        "uncertainty_probes",
    },
}

FAMILIES = (GS, MIL)


def _config_call_kwargs(builder_name: str) -> dict[str, str]:
    """Return ``{kwarg_name: source}`` for the ``*Config(...)`` call in a builder.

    Read off the AST rather than by calling the builder, so a kwarg whose value
    is a literal is still visible (a call would hide it behind the default).
    """
    tree = ast.parse(Path(runner_common.__file__).read_text(encoding="utf-8-sig"))
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
        unknown = sorted(
            set(_config_call_kwargs(family.builder_name)) - _field_names(family.config)
        )
        assert not unknown, (
            f"{family.builder_name}() passes {unknown}, which "
            f"{family.config.__name__} does not declare — this is a field wired "
            f"into the wrong Config"
        )

    @pytest.mark.parametrize("family", FAMILIES, ids=lambda f: f.label)
    def test_every_wired_field_reads_its_own_params_field(
        self, family: _Family
    ) -> None:
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

        This is the assertion the older parity test cannot make, because it
        derives the wired set instead of trusting a hand-written list. Adding a
        field without wiring it fails here; adding one *and* wiring it also
        fails here until the split is restated, which is the point.
        """
        kwargs = set(_config_call_kwargs(family.builder_name))
        unwired = sorted(_field_names(family.params) - kwargs - set(family.cli_only))
        assert not unwired, (
            f"{family.params.__name__} declares {unwired}, which "
            f"{family.builder_name}() never passes — either wire it into "
            f"{family.config.__name__} or add it to the family's cli_only map "
            f"with the module that consumes it"
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
        unreachable = sorted(
            _field_names(family.config) - kwargs - LIBRARY_ONLY[family.label]
        )
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
    def test_cli_only_fields_are_consumed_by_their_runner(
        self, family: _Family
    ) -> None:
        """A CLI-only field must still be read somewhere.

        Without this, ``cli_only`` becomes a place to hide a dead flag: move a
        field out of the wiring and the wiring test is happy again. The module
        named in the map is the one that has to read ``params.<field>``.
        """
        dead = {}
        for field, consumer in family.cli_only.items():
            source = Path(consumer.__file__).read_text(encoding="utf-8-sig")
            if f"params.{field}" not in source:
                dead[field] = consumer.__name__
        assert not dead, (
            f"{family.label}: {dead} are declared CLI-only but no longer read "
            f"as params.<field> in the module that was supposed to consume them"
        )

    @pytest.mark.parametrize("family", FAMILIES, ids=lambda f: f.label)
    def test_cli_only_and_library_only_do_not_overlap(self, family: _Family) -> None:
        """A name cannot be both unreachable from the CLI and CLI-only."""
        assert not set(family.cli_only) & LIBRARY_ONLY[family.label]


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
        default mismatch, and would otherwise be impossible to express.
        """
        kwargs = _config_call_kwargs(family.builder_name)
        param_defaults = _defaults(family.params)
        config_defaults = _defaults(family.config)
        mismatched = {
            name: (param_defaults[name], config_defaults.get(name))
            for name, src in kwargs.items()
            if src == f"params.{name}"
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
        assert (
            build_gs_refine_config(SlmGsRefineParams(), debug=False).far_field_padding
            == 3
        )
        assert (
            build_model_in_loop_config(
                SlmModelInLoopParams(), debug=False
            ).far_field_padding
            == 8
        )
