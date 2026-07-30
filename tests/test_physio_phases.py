"""Respiratory phase decomposition.

The reference implementation, Finn Upham's ``respy``, cannot be used as a test oracle here: its
``Resp_phases`` returns empty columns under pandas copy-on-write. So these check the properties
the decomposition has to have, on a synthetic breath whose ground truth is known by construction.
"""

import numpy as np
import pytest

import micromotion as mm


def synth_breath(n_breaths=20, fs=25.0, period=4.0, insp_frac=0.35, noise=0.0, seed=0):
    """A breathing waveform with a short inspiration and a long expiration-plus-pause.

    Real quiet breathing is asymmetric -- inhale is roughly a third of the cycle -- and a
    symmetric sine would let a broken phase split pass.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, n_breaths * period, 1 / fs)
    phase = (t % period) / period
    y = np.where(phase < insp_frac,
                 -np.cos(np.pi * phase / insp_frac),
                 np.cos(np.pi * (phase - insp_frac) / (1 - insp_frac)))
    return t, y + noise * rng.standard_normal(len(t)), fs


def test_finds_the_right_number_of_breaths():
    _, y, fs = synth_breath(n_breaths=20)
    out = mm.respiration_onsets(y, fs)
    assert 17 <= out["n_breaths"] <= 20      # ends may be trimmed as incomplete


def test_onsets_are_evenly_spaced_at_the_synthetic_period():
    _, y, fs = synth_breath(n_breaths=20, period=4.0)
    out = mm.respiration_onsets(y, fs)
    gaps = np.diff(out["inspiration_s"])
    assert abs(np.median(gaps) - 4.0) < 0.2


def test_phases_partition_the_cycle():
    _, y, fs = synth_breath(n_breaths=20)
    p = mm.respiratory_phases(y, fs)
    # each breath shares two samples: where inspiration ends expiration begins, and vice versa
    both = p["inspiration"] & p["expiration"]
    assert both.sum() <= 2 * p["n_breaths"] + 2, "phases overlap by more than their boundaries"
    covered = (p["inspiration"] | p["expiration"]).mean()
    assert covered > 0.85, f"phases cover only {covered:.0%} of the recording"


def test_inspiration_is_the_shorter_phase():
    _, y, fs = synth_breath(insp_frac=0.35)
    p = mm.respiratory_phases(y, fs)
    assert p["inspiration"].sum() < p["expiration"].sum()


def test_high_flow_is_a_subset_of_its_phase():
    _, y, fs = synth_breath()
    p = mm.respiratory_phases(y, fs)
    for phase in ("inspiration", "expiration"):
        assert not (p[f"{phase}_high"] & ~p[phase]).any()
        assert not (p[f"{phase}_v"] & ~p[phase]).any()


def test_post_expiration_pause_sits_inside_expiration():
    _, y, fs = synth_breath()
    p = mm.respiratory_phases(y, fs)
    assert p["post_expiration"].any(), "no pause found at all"
    assert not (p["post_expiration"] & ~p["expiration"]).any()


def test_pause_velocity_is_lower_than_the_rest_of_expiration():
    """The pause is defined by the rate having slowed, so this is its defining property."""
    _, y, fs = synth_breath()
    p = mm.respiratory_phases(y, fs)
    v = np.abs(p["velocity"])
    rest = p["expiration"] & ~p["post_expiration"]
    assert v[p["post_expiration"]].mean() < v[rest].mean()


def test_survives_nans():
    _, y, fs = synth_breath()
    y = y.copy()
    y[100:140] = np.nan
    p = mm.respiratory_phases(y, fs)
    assert p["n_breaths"] > 10
    assert np.isfinite(p["normalised"]).all()


def test_noise_does_not_multiply_the_breath_count():
    """The baseline-crossing requirement exists to reject rises that are not breaths."""
    _, clean, fs = synth_breath(noise=0.0)
    _, noisy, _ = synth_breath(noise=0.15, seed=3)
    a = mm.respiration_onsets(clean, fs)["n_breaths"]
    b = mm.respiration_onsets(noisy, fs)["n_breaths"]
    assert b <= a * 1.5, f"noise inflated the count from {a} to {b}"


def test_rejects_non_1d_input():
    with pytest.raises(ValueError):
        mm.respiration_onsets(np.zeros((10, 2)), 25.0)


def test_rejects_too_short_input():
    with pytest.raises(ValueError):
        mm.respiration_onsets(np.array([np.nan, 1.0]), 25.0)
