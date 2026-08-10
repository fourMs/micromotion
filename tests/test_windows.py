"""Window equalisation, against imbalances that are known because they were imposed."""

import numpy as np
import pytest

import micromotion as mm
from micromotion import windows


def schedule(lengths, conditions, start=0.0, gap=5.0):
    """Segments laid end to end, so the durations are exactly what was asked for."""
    onset, offset, t = [], [], start
    for L in lengths:
        onset.append(t)
        offset.append(t + L)
        t += L + gap
    return np.array(onset), np.array(offset), np.array(conditions)


def test_balance_sees_an_imposed_imbalance():
    on, off, cond = schedule([60, 60, 45, 45], ["music", "music", "silence", "silence"])
    b = windows.balance(on, off, cond)
    assert not b
    assert b.ratio == pytest.approx(60 / 45)
    assert b.by_condition["music"][1] == pytest.approx(60.0)


def test_balance_passes_a_balanced_schedule():
    on, off, cond = schedule([30, 30, 30, 30], ["a", "a", "b", "b"])
    b = windows.balance(on, off, cond)
    assert b
    assert b.ratio == pytest.approx(1.0)


def test_balance_tolerance_is_honoured():
    """Rounding a segment table to whole seconds must not read as a confound."""
    on, off, cond = schedule([30, 30, 31, 31], ["a", "a", "b", "b"])
    assert windows.balance(on, off, cond)                      # within the 5% default
    assert not windows.balance(on, off, cond, tol=0.0)


def test_equalise_makes_every_window_the_shortest():
    on, off, _ = schedule([60, 45, 90], ["m", "s", "m"])
    new = windows.equalise(on, off)
    assert np.allclose(new - on, 45.0)


def test_equalise_leaves_onsets_alone():
    on, off, _ = schedule([60, 45], ["m", "s"])
    new = windows.equalise(on, off)
    assert new.shape == on.shape
    assert np.all(new > on)


def test_equalise_balances_what_balance_rejected():
    on, off, cond = schedule([60, 60, 45, 45], ["music", "music", "silence", "silence"])
    assert not windows.balance(on, off, cond)
    assert windows.balance(on, windows.equalise(on, off), cond)


def test_a_cap_per_group_keeps_the_long_group_long():
    """The lesson a flat cap teaches the hard way: it discards the long group's signal.

    Two editions, one recording 60 s segments and one recording 20 s. Capping per group leaves the
    first at 60 and equalises within each; a flat cap would cut both to 20 and throw away two
    thirds of the first edition for no gain, since it was already internally balanced.
    """
    on, off, _ = schedule([60, 60, 20, 20], ["m", "s", "m", "s"])
    by = np.array(["A", "A", "B", "B"])
    per_group = windows.equalise(on, off, by=by)
    assert np.allclose((per_group - on)[:2], 60.0)
    assert np.allclose((per_group - on)[2:], 20.0)

    flat = windows.equalise(on, off)
    assert np.allclose(flat - on, 20.0)


def test_group_cap_uses_each_groups_own_shortest():
    on, off, _ = schedule([60, 45, 30, 20], ["m", "s", "m", "s"])
    by = np.array(["A", "A", "B", "B"])
    new = windows.equalise(on, off, by=by)
    assert np.allclose((new - on)[:2], 45.0)
    assert np.allclose((new - on)[2:], 20.0)


def test_explicit_cap_overrides_the_shortest():
    on, off, _ = schedule([60, 45], ["m", "s"])
    assert np.allclose(windows.equalise(on, off, cap_s=30.0) - on, 30.0)


def test_a_cap_longer_than_a_segment_does_not_extend_it():
    on, off, _ = schedule([60, 45], ["m", "s"])
    new = windows.equalise(on, off, cap_s=120.0)
    assert np.allclose(new - on, [60.0, 45.0])


def test_the_measured_effect_of_unequal_windows_is_real():
    """Not a property of the API but of the world, and the reason the API exists.

    A band-limited quantity of motion measured over a long window differs from the same signal
    measured over a short one, because the median settles as the window grows. If that were not so,
    none of this would matter.
    """
    rng = np.random.default_rng(0)
    fs = 100.0
    x = np.cumsum(rng.normal(0, 1, 12000)).reshape(-1, 1) * np.ones((1, 3))
    v = mm.speed_from_position(x, fs, unit="mm")
    short = [float(np.median(v[i:i + int(15 * fs)])) for i in range(0, 6000, 1500)]
    long = [float(np.median(v[i:i + int(60 * fs)])) for i in range(0, 6000, 1500)]
    assert np.std(short) > np.std(long)


def test_rejects_mismatched_shapes():
    on, off, cond = schedule([30, 30], ["a", "b"])
    with pytest.raises(ValueError):
        windows.balance(on, off, cond[:1])
    with pytest.raises(ValueError):
        windows.equalise(on, off[:1])
    with pytest.raises(ValueError):
        windows.equalise(on, off, by=np.array(["A"]))


def test_rejects_a_segment_that_ends_before_it_starts():
    with pytest.raises(ValueError):
        windows.balance(np.array([10.0]), np.array([5.0]), np.array(["a"]))


def test_pooling_hides_an_imbalance_that_by_group_reveals():
    """The reason `balance` takes `by`: a pooled ratio averages the imbalances away.

    Two groups. One is severely confounded, nine to one; the other is balanced and larger. Pooled,
    the medians nearly agree and the check passes. Per group, the bad one is unmissable. This is
    the shape the real corpus had, where six editions pooled to 1.07 while one sat at 9.00.
    """
    on, off, cond = schedule(
        [90, 10] + [30] * 8,
        ["silence", "music"] + ["silence", "music"] * 4,
    )
    by = np.array(["bad", "bad"] + ["good"] * 8)

    assert windows.balance(on, off, cond)                    # pooled: passes, wrongly
    per = windows.balance(on, off, cond, by=by)
    assert not per                                           # per group: caught
    assert per.groups["bad"].ratio == pytest.approx(9.0)
    assert per.groups["good"].balanced
    assert per.ratio == pytest.approx(9.0)                   # worst group's, not the average


def test_equalise_by_group_fixes_what_by_group_balance_found():
    on, off, cond = schedule([90, 10] + [30] * 8,
                             ["silence", "music"] + ["silence", "music"] * 4)
    by = np.array(["bad", "bad"] + ["good"] * 8)
    fixed = windows.equalise(on, off, by=by)
    assert windows.balance(on, fixed, cond, by=by)
    assert np.allclose((fixed - on)[2:], 30.0)               # the good group keeps its length
