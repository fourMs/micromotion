"""Equivalence tests, against constructed answers and against the four outcomes they must separate."""

import numpy as np
import pytest

from micromotion import equivalence as eq


def test_a_genuinely_tiny_difference_is_declared_equivalent():
    rng = np.random.default_rng(0)
    a = rng.normal(5.0, 0.4, 200)
    b = a + rng.normal(0.01, 0.05, 200)          # a hair apart, far inside the bound
    r = eq.tost_paired(a, b, bound=0.5)
    assert r["equivalent"]
    assert r["verdict"] == "equivalent"
    assert -0.5 < r["ci_low"] and r["ci_high"] < 0.5


def test_a_real_difference_is_not_declared_equivalent():
    rng = np.random.default_rng(1)
    a = rng.normal(5.0, 0.4, 200)
    b = a - 1.0 + rng.normal(0, 0.05, 200)       # a whole unit, twice the bound
    r = eq.tost_paired(a, b, bound=0.5)
    assert not r["equivalent"]
    assert r["verdict"] == "effect"


def test_too_few_observations_are_inconclusive_rather_than_equivalent():
    """The failure mode this exists to prevent: reading a small sample's null as 'no effect'."""
    rng = np.random.default_rng(2)
    a = rng.normal(5.0, 1.0, 6)
    b = a + rng.normal(0.0, 1.0, 6)
    r = eq.tost_paired(a, b, bound=0.2)
    assert not r["equivalent"]
    assert r["p_difference"] > 0.05              # not significant either
    assert r["verdict"] == "inconclusive"


def test_a_detectable_but_trivial_difference_gets_its_own_verdict():
    rng = np.random.default_rng(3)
    a = rng.normal(5.0, 0.3, 5000)
    b = a - 0.05 + rng.normal(0, 0.01, 5000)     # real at this n, well inside the bound
    r = eq.tost_paired(a, b, bound=0.5)
    assert r["equivalent"] and r["p_difference"] < 0.05
    assert r["verdict"] == "trivial"


def test_independent_samples_agree_with_the_paired_case_on_the_same_means():
    rng = np.random.default_rng(4)
    a = rng.normal(5.0, 0.5, 300)
    b = rng.normal(5.0, 0.5, 300)
    r = eq.tost_independent(a, b, bound=0.4)
    assert r["equivalent"]
    assert r["n"] == (300, 300)


def test_correlation_equivalence_separates_a_null_from_an_underpowered_one():
    tiny_big_n = eq.equivalence_correlation(r=0.01, n=1000, bound=0.2)
    tiny_small_n = eq.equivalence_correlation(r=0.01, n=20, bound=0.2)
    assert tiny_big_n["equivalent"], "r=0.01 on n=1000 is bounded well inside 0.2"
    assert not tiny_small_n["equivalent"], "the same r on n=20 decides nothing"
    assert tiny_small_n["verdict"] == "inconclusive"


def test_correlation_equivalence_rejects_a_correlation_larger_than_the_bound():
    r = eq.equivalence_correlation(r=0.5, n=200, bound=0.2)
    assert not r["equivalent"]
    assert r["verdict"] == "effect"


def test_the_bound_must_be_a_positive_effect_worth_caring_about():
    rng = np.random.default_rng(5)
    a, b = rng.normal(size=50), rng.normal(size=50)
    with pytest.raises(ValueError):
        eq.tost_paired(a, b, bound=0.0)
    with pytest.raises(ValueError):
        eq.equivalence_correlation(r=0.1, n=100, bound=1.5)


def test_interpret_names_the_bound_in_every_verdict():
    rng = np.random.default_rng(6)
    a = rng.normal(5.0, 0.4, 200)
    for b_ in (a + 0.01 + rng.normal(0, 0.02, 200), a - 1.0 + rng.normal(0, 0.02, 200)):
        s = eq.interpret(eq.tost_paired(a, b_, bound=0.5))
        assert "0.5" in s, s

    # An inconclusive case built rather than drawn: eight pairs whose differences have a spread
    # comparable to the bound cannot resolve it either way, and the sentence must say so.
    diff = np.array([-0.80, 0.70, -0.50, 0.60, -0.40, 0.30, 0.50, -0.40])
    base = rng.normal(5.0, 0.4, len(diff))
    r = eq.tost_paired(base + diff, base, bound=0.2)
    assert r["verdict"] == "inconclusive", r
    assert "cannot decide" in eq.interpret(r)
    assert "0.2" in eq.interpret(r)
