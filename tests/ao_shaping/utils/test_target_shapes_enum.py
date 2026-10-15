"""Tests for the canonical closed-loop shaping ``TargetShape`` vocabulary.

Pins the contract of :mod:`ao_shaping.utils.image.target.shapes`: exactly five
members, string-comparable, labels covering the members, and the coercion
helper.  Also guards the GUI re-export so ``shaping_controls`` keeps exposing
``MIL_TARGET_SHAPE_LABELS`` for existing callers.
"""

from __future__ import annotations

import pytest

from ao_shaping.utils.image.target.shapes import (
    TARGET_SHAPE_LABELS,
    TargetShape,
    coerce_target_shape,
    target_shape_options,
)


class TestTargetShapeEnum:
    def test_members_are_exactly_the_five_supported(self) -> None:
        assert {member.value for member in TargetShape} == {
            "square",
            "rectangle",
            "triangle",
            "star",
            "circle",
        }

    def test_members_compare_equal_to_raw_strings(self) -> None:
        assert TargetShape.SQUARE == "square"
        assert TargetShape.CIRCLE == "circle"


class TestLabels:
    def test_labels_cover_exactly_the_members(self) -> None:
        assert set(TARGET_SHAPE_LABELS) == {member.value for member in TargetShape}
        assert all(
            isinstance(label, str) and label for label in TARGET_SHAPE_LABELS.values()
        )

    def test_target_shape_options_order_matches_labels(self) -> None:
        assert target_shape_options() == list(TARGET_SHAPE_LABELS)


class TestCoerce:
    def test_coerce_accepts_enum_and_string(self) -> None:
        assert coerce_target_shape(TargetShape.TRIANGLE) is TargetShape.TRIANGLE
        assert coerce_target_shape("triangle") is TargetShape.TRIANGLE

    def test_coerce_rejects_unknown(self) -> None:
        with pytest.raises(
            ValueError,
            match=r"valid values: square, rectangle, triangle, star, circle",
        ):
            coerce_target_shape("hexagon")


class TestGuiReExport:
    def test_shaping_controls_still_exposes_mil_labels(self) -> None:
        from ao_shaping.gui.slm.shaping_controls import MIL_TARGET_SHAPE_LABELS

        assert MIL_TARGET_SHAPE_LABELS == TARGET_SHAPE_LABELS
