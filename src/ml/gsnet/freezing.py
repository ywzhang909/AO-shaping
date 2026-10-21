"""Freeze strategies for `FourierGSNet`: train only part of the network.

Why this exists
---------------
Transfer learning on the bench has a hard data budget: the operator has to
collect probe frames on the real device, and every probe costs an exposure.
So the useful question is not "how big is the network" but **"which part of
the pretrained network has to move, and which part can stay exactly where
pretraining put it"**.

The network is a stack of unrolled GS layers. Each layer owns one
`ConditionUNet`, and that U-Net splits into four natural groups:

===============  =========================================================
group            modules
===============  =========================================================
``encoder``      ``down1``, ``down2``, ``down3``, ``bottleneck``
``decoder``      ``up1``, ``up2``
``films``        ``cond1``, ``cond2`` (the FiLM conditioning branches)
``outconv``      ``out_conv`` (the 1x1 read-out producing the phase update)
===============  =========================================================

The physics half of every layer (the FFT and the amplitude exchange) has **no
parameters at all**, so the whole parameter budget lives in these four
groups.

A subtlety that is easy to get wrong
------------------------------------
`ConditionUNet` uses `BatchNorm2d`. Calling ``model.train()`` puts every
BatchNorm back into training mode, where the **running statistics keep
updating** and the **affine weight/bias keep being updated** by the
optimiser if they are still trainable. A "frozen" encoder whose BatchNorm
statistics drift is not frozen at all — the network's input scaling changes
underneath the frozen weights, and the same checkpoint produces different
outputs in train and eval mode.

:func:`apply_freeze` therefore does two things: it clears ``requires_grad``
on the frozen parameters **and** puts the BatchNorm modules inside a frozen
group into evaluation mode. Because ``model.train()`` resets the mode of
every submodule, :func:`prepare_for_training` exists to do the two calls in
the right order (``train()`` first, then ``apply_freeze()``) — that is the
call site to use in a training loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

#: The four parameter groups of `ConditionUNet` that can be frozen.
FREEZE_GROUPS: tuple[str, ...] = ("encoder", "decoder", "films", "outconv")

#: Attribute names of `ConditionUNet` belonging to each group.
_GROUP_MODULES: dict[str, tuple[str, ...]] = {
    "encoder": ("down1", "down2", "down3", "bottleneck"),
    "decoder": ("up1", "up2"),
    "films": ("cond1", "cond2"),
    "outconv": ("out_conv",),
}

#: Canonical name of each freeze configuration, in report order.
FREEZE_SPEC_NAMES: tuple[str, ...] = (
    "full",
    "frozen",
    "freeze-encoder",
    "freeze-decoder",
    "freeze-films",
    "freeze-outconv",
)


@dataclass(frozen=True)
class FreezeSpec:
    """Which parameter groups stay frozen during fine-tuning.

    Attributes:
        encoder: Freeze the U-Net encoder and bottleneck.
        decoder: Freeze the U-Net decoder.
        films: Freeze the FiLM conditioning branches.
        outconv: Freeze the 1x1 read-out that produces the phase update.
    """

    encoder: bool = False
    decoder: bool = False
    films: bool = False
    outconv: bool = False

    @classmethod
    def from_name(cls, name: str) -> "FreezeSpec":
        """Build a spec from its canonical name.

        Args:
            name: One of :data:`FREEZE_SPEC_NAMES`.

        Returns:
            The matching spec.

        Raises:
            ValueError: If ``name`` is not a known configuration.
        """
        if name == "full":
            return cls()
        if name == "frozen":
            return cls(encoder=True, decoder=True, films=True, outconv=True)
        if name == "freeze-encoder":
            return cls(encoder=True)
        if name == "freeze-decoder":
            return cls(decoder=True)
        if name == "freeze-films":
            return cls(films=True)
        if name == "freeze-outconv":
            return cls(outconv=True)
        raise ValueError(
            f"unknown freeze spec {name!r}; expected one of {FREEZE_SPEC_NAMES}"
        )

    @property
    def name(self) -> str:
        """The canonical name of this configuration."""
        for name in FREEZE_SPEC_NAMES:
            if FreezeSpec.from_name(name) == self:
                return name
        raise ValueError(f"no canonical name for spec {self!r}")

    def frozen_groups(self) -> tuple[str, ...]:
        """The names of the groups this spec freezes, in canonical order."""
        return tuple(group for group in FREEZE_GROUPS if getattr(self, group))


def _iter_condition_unets(model: nn.Module):
    """Yield every `ConditionUNet` reachable from ``model``.

    Works for `FourierGSNet` (whose layers own a ``phase_net``) and for a
    bare `ConditionUNet` (which is its own "condition net").
    """
    from ml.gsnet.model import ConditionUNet  # noqa: PLC0415

    for module in model.modules():
        if isinstance(module, ConditionUNet):
            yield module


def _group_modules(model: nn.Module, group: str) -> list[nn.Module]:
    """Collect the sub-modules of ``model`` that belong to ``group``."""
    names = _GROUP_MODULES[group]
    collected: list[nn.Module] = []
    for unet in _iter_condition_unets(model):
        for name in names:
            module = getattr(unet, name, None)
            if module is not None:
                collected.append(module)
    return collected


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return ``(trainable, total)`` parameter counts of ``model``."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def group_parameter_counts(model: nn.Module) -> dict[str, int]:
    """Parameter count per freeze group, plus the parameters outside them.

    Args:
        model: A `FourierGSNet` (or any module containing `ConditionUNet`s).

    Returns:
        Mapping with one entry per group in :data:`FREEZE_GROUPS`, an
        ``"other"`` entry for everything that is not part of a group, and a
        ``"total"`` entry.
    """
    counts: dict[str, int] = {}
    for group in FREEZE_GROUPS:
        counts[group] = sum(
            p.numel() for module in _group_modules(model, group)
            for p in module.parameters()
        )
    grouped = sum(counts.values())
    counts["other"] = sum(p.numel() for p in model.parameters()) - grouped
    counts["total"] = sum(p.numel() for p in model.parameters())
    return counts


def apply_freeze(model: nn.Module, spec: FreezeSpec) -> dict[str, int]:
    """Freeze the groups named by ``spec`` and report the parameter counts.

    Frozen parameters get ``requires_grad = False``; BatchNorm modules inside
    a frozen group are switched to evaluation mode so their running
    statistics stop drifting. Calling this repeatedly is safe.

    ⚠️ ``model.train()`` resets the mode of *every* submodule, so a training
    loop must call :func:`prepare_for_training` (or re-apply this function
    after each ``train()`` call) — otherwise the frozen BatchNorm statistics
    start moving again.

    Args:
        model: The network to modify in place.
        spec: Which groups to freeze.

    Returns:
        ``{"trainable": int, "total": int, "frozen": int}``.
    """
    frozen_groups = set(spec.frozen_groups())
    for group in FREEZE_GROUPS:
        modules = _group_modules(model, group)
        if group not in frozen_groups:
            for module in modules:
                for parameter in module.parameters():
                    parameter.requires_grad = True
            continue
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad = False
            for submodule in module.modules():
                if isinstance(submodule, nn.modules.batchnorm._BatchNorm):
                    submodule.eval()
    trainable, total = count_parameters(model)
    return {
        "trainable": trainable,
        "total": total,
        "frozen": total - trainable,
    }


def prepare_for_training(model: nn.Module, spec: FreezeSpec) -> dict[str, int]:
    """Put ``model`` in training mode and apply the freeze spec.

    This is the call site a training loop should use: ``model.train()``
    first (it resets the mode of every submodule), then :func:`apply_freeze`
    (which puts the frozen BatchNorms back into evaluation mode).

    Args:
        model: The network to prepare.
        spec: Which groups to freeze.

    Returns:
        ``{"trainable": int, "total": int, "frozen": int}``.
    """
    model.train()
    return apply_freeze(model, spec)


def freeze_state(model: nn.Module) -> dict[str, bool]:
    """Return the ``requires_grad`` state of every parameter, keyed by name.

    Useful to assert in tests that a freeze spec did what it claimed.
    """
    return {name: p.requires_grad for name, p in model.named_parameters()}
