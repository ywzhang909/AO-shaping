"""Which compute device the torch stages use (fit, synthesis, validation).

Why a resolver instead of a literal
-----------------------------------
Every transfer stage that does torch work used to pin ``device="cpu"`` at its
call site. That is a *policy* decision, not a property of the algorithm, and it
silently threw away the workstation GPU. But a GPU is not always present (CI has
none, and the digital-twin machines may not), and an operator comparing a run
against a historical CPU artefact must be able to pin the device deliberately.
So the policy lives in exactly one place:

* ``None`` / ``"auto"`` — CUDA when ``torch.cuda.is_available()`` else CPU.
* ``"cpu"`` / ``"cuda"`` — returned unchanged. An explicit choice is honoured
  even when CUDA is unavailable: torch then fails loudly at the first
  allocation instead of this helper silently downgrading a run the operator
  asked for.
* anything else — ``ValueError``. A typo must not silently become CPU.

Fallback, and what "unchanged default" means
--------------------------------------------
``"auto"`` is the session default, so:

* on a machine **without** a usable CUDA device the resolved value is exactly
  today's ``"cpu"`` — nothing about those runs changes;
* on a machine **with** one the stages follow the hardware, and
  ``--device cpu`` reproduces the historical CPU numbers bit-for-bit.

Every artefact that records the value records the *resolved* one, so a run can
always be reproduced on the device that produced it.

Import policy
-------------
Nothing here imports torch at module scope: the transfer package stays
importable without it (the ``plan`` command and the package's import guards run
device-free), so the availability probe is a function-local import.
"""
from __future__ import annotations

#: Values the CLI / GUI / :class:`TransferSession` accept for ``device``.
DEVICE_CHOICES = ("auto", "cpu", "cuda")


def resolve_device(requested: str | None = None) -> str:
    """Return ``"cuda"`` or ``"cpu"`` for a requested device string.

    Args:
        requested: ``None`` or ``"auto"`` to follow the machine, ``"cpu"`` /
            ``"cuda"`` to pin it, anything else is rejected.

    Returns:
        The device name to hand to torch (``"cpu"`` or ``"cuda"``).

    Raises:
        ValueError: ``requested`` is not one of :data:`DEVICE_CHOICES`.
    """
    if requested is None:
        return "cuda" if cuda_available() else "cpu"
    value = str(requested).strip().lower()
    if value == "auto":
        return "cuda" if cuda_available() else "cpu"
    if value in {"cpu", "cuda"}:
        return value
    raise ValueError(f"unknown device {requested!r}; expected one of {list(DEVICE_CHOICES)}")


def cuda_available() -> bool:
    """``torch.cuda.is_available()``, or ``False`` when torch is not installed."""
    try:
        import torch
    except ImportError:  # torch is an optional extra for this package
        return False
    return bool(torch.cuda.is_available())
