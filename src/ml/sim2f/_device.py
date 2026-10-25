"""Device resolution shared by the :mod:`ml.sim2f` trainers.

``model`` / ``train`` / ``loop`` / ``optics`` each carried their own copy of
this ten-line resolver (``evaluate`` imported ``train``'s). The copies had
drifted in docstring wording, and ``optics``' variant compared
``device == "auto"`` directly, which raises ``TypeError`` instead of
``ValueError`` on a non-string, non-device spec. One implementation here;
``str(spec) == "auto"`` is the defensive form the majority copy used.

This module imports ``torch`` only -- unlike :mod:`ml.sim2f.config`, which is
deliberately stdlib-only so the parser and report generator can read the same
numbers without the optional ``ml`` dependency group.
"""

from __future__ import annotations

import torch

__all__ = ["_resolve_device"]


def _resolve_device(spec: str | torch.device) -> torch.device:
    """Resolve ``"auto"`` to CUDA when available.

    Args:
        spec: ``"auto"``, a device string (``"cpu"``, ``"cuda:1"``) or a
            :class:`torch.device`.

    Returns:
        The concrete :class:`torch.device`.
    """
    if isinstance(spec, torch.device):
        return spec
    if str(spec) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(spec))
