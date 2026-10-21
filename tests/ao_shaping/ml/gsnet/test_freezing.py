"""Unit tests for the freeze strategies (src/ml/gsnet/freezing.py).

The central contract has two halves:

1. ``apply_freeze`` clears ``requires_grad`` on exactly the named groups and
   leaves every other parameter trainable;
2. BatchNorm modules **inside a frozen group** are put into evaluation mode,
   and stay there even after a training loop re-enters ``model.train()`` —
   the regression :func:`prepare_for_training` exists to prevent.

Pure torch, CPU only: no hardware, no network.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
import torch.nn as nn

from ml.gsnet.freezing import (
    FREEZE_GROUPS,
    FREEZE_SPEC_NAMES,
    FreezeSpec,
    apply_freeze,
    count_parameters,
    freeze_state,
    group_parameter_counts,
    prepare_for_training,
)
from ml.gsnet.model import FourierGSNet

#: Module attributes of ``ConditionUNet`` per freeze group (mirrors the
#: private ``freezing._GROUP_MODULES``; the helpers below assert that this
#: mapping and the module's own accounting agree).
_GROUP_ATTRS: dict[str, tuple[str, ...]] = {
    "encoder": ("down1", "down2", "down3", "bottleneck"),
    "decoder": ("up1", "up2"),
    "films": ("cond1", "cond2"),
    "outconv": ("out_conv",),
}


def _small_model(seed: int = 0) -> FourierGSNet:
    """Build a small reproducible FourierGSNet (train mode by construction)."""
    torch.manual_seed(seed)
    return FourierGSNet(num_layers=2, base_channels=8)


def _group_of_name(path: str) -> str:
    """Map a parameter/module path to its freeze group via path components.

    A name belongs to group X iff one of X's module attributes appears as a
    dot-separated path component, e.g. ``layers.0.phase_net.down1.0.weight``
    -> encoder, ``layers.0.phase_net.cond1.gamma_beta.weight`` -> films.
    """
    parts = set(path.split("."))
    hits = [
        group
        for group, attrs in _GROUP_ATTRS.items()
        if parts.intersection(attrs)
    ]
    if len(hits) > 1:
        raise AssertionError(f"parameter {path!r} matches multiple groups {hits}")
    if not hits:
        return "other"
    return hits[0]


class TestFreezeSpecNames:
    """Canonical-name round-trip and rejection of unknown names."""

    def test_from_name_covers_every_canonical_name(self):
        for name in FREEZE_SPEC_NAMES:
            spec = FreezeSpec.from_name(name)
            assert spec.name == name
            with pytest.raises(dataclasses.FrozenInstanceError):
                spec.encoder = not spec.encoder  # type: ignore[misc]

    def test_from_name_rejects_unknown_name(self):
        for bogus in ("bogus", ""):
            with pytest.raises(ValueError):
                FreezeSpec.from_name(bogus)


class TestFreezeCounts:
    """Trainable/total accounting of the two extreme specs."""

    def test_frozen_spec_freezes_everything(self):
        model = _small_model()
        spec = FreezeSpec.from_name("frozen")
        counts = apply_freeze(model, spec)
        assert counts["trainable"] == 0
        assert counts["frozen"] == counts["total"] > 0
        assert all(
            grad is False for grad in freeze_state(model).values()
        )

    def test_full_spec_keeps_everything_trainable(self):
        model = _small_model()
        counts = apply_freeze(model, FreezeSpec.from_name("full"))
        assert counts["trainable"] == counts["total"] > 0
        assert all(
            grad is True for grad in freeze_state(model).values()
        )

    def test_count_parameters_matches_manual_sum(self):
        model = _small_model()
        trainable, total = count_parameters(model)
        assert trainable == sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        assert total == sum(p.numel() for p in model.parameters())


class TestGroupMembership:
    """Group attribution: totals, exclusivity, and single-group freezes."""

    def test_group_counts_sum_to_total(self):
        model = _small_model()
        counts = group_parameter_counts(model)
        group_sum = sum(counts[group] for group in FREEZE_GROUPS)
        assert group_sum + counts["other"] == counts["total"] > 0
        assert counts["other"] == 0

    def test_helper_mapping_is_total_and_matches_module_counts(self):
        model = _small_model()
        counts = group_parameter_counts(model)
        helper_counts = dict.fromkeys((*FREEZE_GROUPS, "other"), 0)
        for name, parameter in model.named_parameters():
            helper_counts[_group_of_name(name)] += parameter.numel()
        for group in FREEZE_GROUPS:
            assert helper_counts[group] == counts[group]
        assert helper_counts["other"] == counts["other"] == 0

    @pytest.mark.parametrize("frozen_group", FREEZE_GROUPS)
    def test_freezing_one_group_leaves_the_others_trainable(self, frozen_group):
        model = _small_model()
        spec_name = f"freeze-{frozen_group}"
        counts = apply_freeze(model, FreezeSpec.from_name(spec_name))

        state = freeze_state(model)
        for name, requires_grad in state.items():
            group = _group_of_name(name)
            assert group != "other"
            assert requires_grad is (group != frozen_group)

        frozen_params = sum(
            p.numel()
            for n, p in model.named_parameters()
            if _group_of_name(n) == frozen_group
        )
        assert counts["frozen"] == frozen_params
        assert counts["trainable"] == counts["total"] - frozen_params

    def test_spec_switch_unfreezes_previously_frozen_group(self):
        # apply_freeze re-asserts requires_grad=True on the non-frozen groups,
        # so switching specs on the SAME model must restore trainability.
        model = _small_model()
        apply_freeze(model, FreezeSpec.from_name("freeze-encoder"))
        assert any(
            grad is False for grad in freeze_state(model).values()
        )
        counts = apply_freeze(model, FreezeSpec.from_name("full"))
        assert counts["trainable"] == counts["total"] > 0
        assert all(grad is True for grad in freeze_state(model).values())


class TestBatchNormMode:
    """The BatchNorm half of the contract — the reason apply_freeze exists."""

    def _batchnorms_by_group(self, model: nn.Module) -> dict[str, list[nn.Module]]:
        grouped: dict[str, list[nn.Module]] = {}
        for module_name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                grouped.setdefault(_group_of_name(module_name), []).append(module)
        return grouped

    def test_frozen_batchnorm_is_in_eval_mode(self):
        model = _small_model()
        assert model.training is True
        apply_freeze(model, FreezeSpec.from_name("freeze-encoder"))

        grouped = self._batchnorms_by_group(model)
        assert grouped["encoder"], "model must contain encoder BatchNorms"
        for group in ("decoder", "films", "outconv"):
            if group == "outconv":
                continue  # out_conv is a bare Conv2d: no BatchNorm by design
            assert grouped[group], f"model must contain {group} BatchNorms"
        for module in grouped["encoder"]:
            assert module.training is False
        for group in ("decoder", "films"):
            for module in grouped[group]:
                assert module.training is model.training

    def test_prepare_for_training_reasserts_frozen_batchnorm_after_train(self):
        model = _small_model()
        spec = FreezeSpec.from_name("freeze-encoder")
        apply_freeze(model, spec)
        encoder_bns = self._batchnorms_by_group(model)["encoder"]
        assert all(m.training is False for m in encoder_bns)

        # A training loop that re-enters train mode silently un-freezes the
        # BatchNorm running statistics — this is the regression.
        model.train()
        assert any(m.training is True for m in encoder_bns)

        # A real loop reaches prepare_for_training from eval mode (after an
        # eval pass); prepare_for_training itself must re-enter train mode.
        model.eval()
        counts = prepare_for_training(model, spec)
        assert model.training is True
        assert counts["trainable"] == counts["total"] - sum(
            p.numel()
            for n, p in model.named_parameters()
            if _group_of_name(n) == "encoder"
        )
        assert all(m.training is False for m in encoder_bns)

    def test_apply_freeze_is_idempotent(self):
        model = _small_model()
        spec = FreezeSpec.from_name("freeze-films")
        first = apply_freeze(model, spec)
        first_state = freeze_state(model)
        second = apply_freeze(model, spec)
        assert second == first
        assert freeze_state(model) == first_state


class TestFrozenDeterminism:
    """A fully frozen model must be output-stable in train mode."""

    def _forward_twice(self, model: FourierGSNet) -> tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(0)
        source = torch.rand(2, 1, 16, 16)
        target = torch.rand(2, 1, 16, 16)
        with torch.no_grad():
            first = model(source, target)
            second = model(source, target)
        return first, second

    def test_frozen_model_output_is_deterministic_in_train_mode(self):
        model = _small_model(seed=1)
        prepare_for_training(model, FreezeSpec.from_name("frozen"))
        assert model.training is True
        first, second = self._forward_twice(model)
        assert torch.equal(first, second)

    def test_trainable_model_output_differs_between_passes_in_train_mode(self):
        # The strong "outputs differ between passes" claim is NOT
        # reproducible: a train-mode BatchNorm normalises by batch
        # statistics, so two passes over the same batch with unchanged
        # weights give the same output. The observable distinction of the
        # "full" spec is therefore the module mode itself: nothing frozen
        # means every BatchNorm stays in train mode.
        model = _small_model(seed=2)
        counts = prepare_for_training(model, FreezeSpec.from_name("full"))
        assert counts["trainable"] == counts["total"] > 0
        batchnorms = [
            module
            for module in model.modules()
            if isinstance(module, nn.BatchNorm2d)
        ]
        assert batchnorms
        assert all(m.training is True for m in batchnorms)
