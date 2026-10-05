"""Shared click plumbing for the CLI: the ``with_params`` dataclass-click layer.

``ao_shaping.utils.cli`` is the single home of the ``Annotated[T, option(...)]``
→ click-options mechanism. It sits **below** both ``runners/`` and ``tools/``
(and below ``algorithm/`` and ``drivers/``), so either side can declare
parameters against it without creating an import cycle — which matters because
``runners/slm/zernike_matrix_runner.py`` already imports ``ao_shaping.tools.slm``
at module level.

Re-exports the three public names; the mechanism itself lives in
:mod:`ao_shaping.utils.cli.params`, whose leaf-module invariant (stdlib +
``click`` only, never anything from ``ao_shaping``) is what keeps this package
cycle-free. The parameter *dataclasses* that consume these names stay in
:mod:`ao_shaping.runners.runner_common`, which re-exports ``ClickGroup``,
``option`` and ``with_params`` for backward compatibility.

Deliberately **not** re-exported from :mod:`ao_shaping.utils` — that module is
eager, so widening it would add import side effects for every consumer.
"""

from __future__ import annotations

from ao_shaping.utils.cli.params import ClickGroup, option, with_params

__all__ = ["ClickGroup", "option", "with_params"]
