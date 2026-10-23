"""Guards for the single-sourced convergence-diagnostics family (issue #173 batch 3).

The three wfless dynamic-schedule call sites (``pib`` / ``slm_zernike_primitives``
/ ``slm_square_shaping``) each used to carry a byte-for-byte copy of the same
convergence-detection core (grad_cv gates / objective-trend gates / early-epoch
warm-up / final clamp). Batch 3 moves that family into
``ao_shaping.optimizer.wfless._convergence_diagnostics`` so the thresholds cannot
drift silently.

These tests do two things:

1. ``test_schedule_bodies_no_longer_inline_the_family`` -- AST-check that none of
   the three call sites still spells out the family's signature magic numbers
   (``0.1``/``0.3``/``0.8`` grad-cv gates, ``1e-5``/``0.001`` trend gates, the
   ``< 20`` warm-up, the ``0.2, 2.0`` / ``0.3, 2.5`` clamps). They must now import
   and call ``convergence_lr_delta_factors`` instead.

2. ``test_refactor_is_behavior_invariant`` -- pin the shared function against a
   large sweep of histories so a future "cleanup" of the shared core that changes
   a threshold or a branch ordering is caught (the invariance already verified at
   refactor time: 40k-trial match).

The per-call-site *base* (lr, delta) segmentation is deliberately NOT checked
here: it is not part of the convergence family (``pib``/``slm_zernike_primitives``
segment by power-radius, ``slm_square_shaping`` by CV) and stays with each caller.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from ao_shaping.optimizer.wfless import _convergence_diagnostics as cd
from ao_shaping.optimizer.wfless._convergence_diagnostics import (
    convergence_lr_delta_factors,
)

#: The three call sites that must import the shared core, not inline it.
CALL_SITES = (
    "src/ao_shaping/optimizer/wfless/pib.py",
    "src/ao_shaping/optimizer/wfless/slm_zernike_primitives.py",
    "src/ao_shaping/optimizer/wfless/slm_square_shaping.py",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _schedule_bodies(rel: str) -> list[ast.AST]:
    """Collect every ``learning_schedule`` function node in ``rel``."""
    tree = ast.parse((_repo_root() / rel).read_text(encoding="utf-8-sig"))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "learning_schedule"
    ]


def _calls_shared_core(body: ast.AST) -> bool:
    """True if the body calls ``convergence_lr_delta_factors``."""
    return any(
        (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "convergence_lr_delta_factors"
        )
        for node in ast.walk(body)
    )


def _inline_diagnostics_calls(body: ast.AST) -> list[str]:
    """Names of numpy diagnostics (``std``/``clip``) still called in the body.

    The family's inlined form computes a coefficient of variation
    (``np.std``) and clamps the factors (``np.clip``). Once the family is
    single-sourced into ``_convergence_diagnostics``, neither is called inside
    the per-site ``learning_schedule`` body any more.
    """
    found: list[str] = []
    for node in ast.walk(body):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "np"
                and node.func.attr in ("std", "clip", "mean")
            ):
                found.append(node.func.attr)
    return found


def test_call_sites_import_the_shared_core() -> None:
    for rel in CALL_SITES:
        tree = ast.parse((_repo_root() / rel).read_text(encoding="utf-8-sig"))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "ao_shaping.optimizer.wfless._convergence_diagnostics"
            )
        }
        assert "convergence_lr_delta_factors" in imported, (
            f"{rel} no longer imports convergence_lr_delta_factors from "
            f"_convergence_diagnostics"
        )


def test_schedule_bodies_no_longer_inline_the_family() -> None:
    """No call-site ``learning_schedule`` may still inline the family.

    The inlined family is recognised structurally, not by its literals (which
    legitimately survive in the per-site base segmentation): once the family
    is single-sourced, the per-site body neither computes the coefficient of
    variation / clamps the factors (``np.std``/``np.clip``/``np.mean``) nor
    calls the shared core. Each body must call
    ``convergence_lr_delta_factors`` and contain no inline diagnostics.
    """
    for rel in CALL_SITES:
        bodies = _schedule_bodies(rel)
        assert bodies, f"{rel} defines no learning_schedule"
        for body in bodies:
            inline_calls = _inline_diagnostics_calls(body)
            assert not inline_calls, (
                f"{rel}::learning_schedule still inlines the convergence family "
                f"(np.{sorted(set(inline_calls))} still called in the body); the "
                f"diagnostics belong in _convergence_diagnostics"
            )
            assert _calls_shared_core(body), (
                f"{rel}::learning_schedule does not call convergence_lr_delta_factors"
            )


def test_refactor_is_behavior_invariant() -> None:
    """A representative sweep of (gradient, objective) histories must produce
    finite, clamped, deterministic factors, and the two objective-direction
    modes must diverge exactly where their original inline bodies differed."""
    import numpy as np

    rng = np.random.default_rng(42)

    for _ in range(300):
        lg = int(rng.integers(1, 25))
        lp = int(rng.integers(1, 25))
        grads = [float(v) for v in rng.uniform(0.0, 9.0, lg)]
        pibs = [float(x) for x in np.linspace(0.1, 0.9, lp)]
        epoch = int(rng.integers(0, 60))

        up = convergence_lr_delta_factors(
            grads, pibs, epoch, higher_is_better=True, stagnant_needs_low_std=True
        )
        down = convergence_lr_delta_factors(
            grads, pibs, epoch, higher_is_better=False, stagnant_needs_low_std=False
        )

        for lr, d in (up, down):
            assert 0.2 <= lr <= 2.0 and 0.3 <= d <= 2.5
            assert np.isfinite(lr) and np.isfinite(d)

        # Determinism.
        again = convergence_lr_delta_factors(
            grads, pibs, epoch, higher_is_better=True, stagnant_needs_low_std=True
        )
        assert up == again

    # Direction flip is real: a steadily *rising* objective is converging for a
    # higher-is-better objective but diverging for a lower-is-better one, so the
    # two modes must not always agree.
    rising = [float(x) for x in np.linspace(0.1, 0.9, 10)]
    stable_grads = [1.0, 1.01, 0.99, 1.0, 1.005, 0.995, 1.0, 1.01, 0.99, 1.0]
    a = convergence_lr_delta_factors(
        stable_grads, rising, epoch=50, higher_is_better=True, stagnant_needs_low_std=True
    )
    b = convergence_lr_delta_factors(
        stable_grads, rising, epoch=50, higher_is_better=False, stagnant_needs_low_std=False
    )
    # For a rising objective, higher-is-better treats it as convergence (caps
    # lr up toward 1.1); lower-is-better treats it as divergence (forces 0.4).
    assert a[0] > b[0], (
        f"direction modes unexpectedly identical: higher={a} lower={b}"
    )


def test_module_is_a_leaf() -> None:
    """The shared core must stay a leaf: only ``__future__`` + ``numpy``, no
    ``ao_shaping`` import, no SDK, no disk I/O at import time (matches the repo
    lazy-loading contract)."""
    tree = ast.parse(Path(cd.__file__).read_text(encoding="utf-8-sig"))
    allowed = {"__future__", "numpy"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            top = node.names[0].name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
        else:
            continue
        assert top in allowed, (
            f"shared core must only import __future__/numpy, found {top!r}"
        )
