"""
Backward-compat shim for ``AOClosedLoop``.

Canonical location: :mod:`ao_shaping.optimizer.wf.closed_loop`.
This shim keeps ``ao_shaping.runners.closed_loop`` importable for callers
written before the Wave 1 ``runners/`` → ``optimizer/wf/`` refactor.
"""

from ao_shaping.optimizer.wf.closed_loop import AOClosedLoop  # noqa: F401

__all__ = ["AOClosedLoop"]
