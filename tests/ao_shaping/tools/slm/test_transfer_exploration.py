"""Exploration accepts uphill moves without overwriting the measured champion."""
import pytest

from ao_shaping.tools.slm.transfer.exploration import NonmonotoneAcceptance


def test_uphill_move_keeps_champion_and_can_reach_a_better_endpoint():
    policy = NonmonotoneAcceptance(.4, temperature=1., relative_budget=.25, seed=2)
    uphill = policy.update(.4, .42)
    assert uphill["accepted"] and uphill["uphill"] and not uphill["improved_best"]
    assert policy.best == .4
    endpoint = policy.update(.42, .35)
    assert endpoint["accepted"] and endpoint["improved_best"]
    assert policy.best == .35


def test_guard_rejects_excessive_deterioration_even_at_high_temperature():
    policy = NonmonotoneAcceptance(.4, temperature=100., relative_budget=.1)
    assert not policy.update(.43, .45)["accepted"]
    assert policy.best == .4


def test_zero_temperature_matches_strict_rule_and_randomness_is_replayable():
    strict = NonmonotoneAcceptance(.4, temperature=0.)
    assert not strict.update(.4, .41)["accepted"]
    assert strict.update(.4, .39)["accepted"]
    a = NonmonotoneAcceptance(.4, temperature=.02, seed=42)
    b = NonmonotoneAcceptance(.4, temperature=.02, seed=42)
    assert a.update(.4, .41) == b.update(.4, .41)
    with pytest.raises(ValueError):
        a.update(.4, float("nan"))
