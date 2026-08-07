"""The checks, each tested against a reconstruction of the failure it exists to catch."""

import numpy as np
import pytest

import micromotion as mm
from micromotion import validate


def _head(n=4000, fs=200.0, seed=0):
    """A plausible standing head marker: 1.7 m up, swaying by a few millimetres."""
    rng = np.random.default_rng(seed)
    sway = np.cumsum(rng.normal(0, 0.05, (n, 3)), axis=0)
    return sway + np.array([0.0, 0.0, 1700.0])


def test_zero_triplets_are_caught_and_cost_a_factor_of_ten_in_path_length():
    """The 2021 failure: 93 gap frames read as a point on the floor.

    The check must fire, and the reason it matters is reproduced here — the median hardly
    moves, so nothing looks wrong, while the path length inflates by an order of magnitude.
    """
    x = _head()
    x[1000:1093] = 0.0

    found = validate.zero_triplets(x, where="sverm_2021")
    assert len(found) == 1
    assert found[0].severity == "error"
    assert "93" in found[0].message

    clean = x.copy()
    clean[(x == 0.0).all(axis=1)] = np.nan
    clean = mm.interpolate_gaps(clean)
    naive = np.nansum(np.linalg.norm(np.diff(x, axis=0), axis=1))
    true = np.nansum(np.linalg.norm(np.diff(clean, axis=0), axis=1))
    assert naive > 10 * true


def test_real_optical_data_does_not_trip_the_zero_check():
    assert validate.zero_triplets(_head()) == []


def test_an_all_nan_series_is_an_error_not_a_missing_sensor():
    """A band-pass across an unbridgeable gap empties the series and looks like absence."""
    x = _head()
    x[:] = np.nan
    found = validate.finite_fraction(x)
    assert [f.severity for f in found] == ["error"]


def test_a_thin_series_warns_rather_than_fails():
    x = _head()
    x[500:3000] = np.nan
    found = validate.finite_fraction(x, min_finite=0.8)
    assert [f.severity for f in found] == ["warning"]


def test_longest_finite_span_finds_the_measurable_part():
    """The Sverm 2012 case: a marker lost part-way and never regained."""
    x = _head(n=1000)
    x[800:] = np.nan
    start, length = validate.longest_finite_span(x)
    assert (start, length) == (0, 800)

    y = _head(n=1000)
    y[:300] = np.nan
    assert validate.longest_finite_span(y) == (300, 700)


def test_backward_timestamps_are_an_error_and_duplicates_a_warning():
    """The balance-board clock: 83 backward steps and 123111 duplicates, in miniature."""
    t = np.arange(1000, dtype=float) / 100
    t[500] = t[499] - 0.01
    assert any(f.check == "timestamps" and f.severity == "error"
               for f in validate.timestamps(t))

    t2 = np.repeat(np.arange(500, dtype=float) / 100, 2)
    assert any(f.severity == "warning" for f in validate.timestamps(t2))


def test_a_clean_clock_passes():
    assert validate.timestamps(np.arange(2000, dtype=float) / 200) == []


def test_rate_disagreement_is_caught_at_the_scale_the_ax3_units_showed():
    """The 2024 championship: 191.29 Hz of nominal 200, a property of the device."""
    t = np.arange(63000, dtype=float) / 191.29
    found = validate.rate_agreement(t, documented_hz=200.0)
    assert len(found) == 1 and found[0].severity == "error"
    assert "191" in found[0].message

    assert validate.rate_agreement(np.arange(2000, dtype=float) / 200, 200.0) == []


def test_the_sixteen_bit_ceiling_is_recognised():
    """The C3D truncation that nearly cost seven sessions their last thirty-two seconds."""
    assert validate.frame_count(65535)[0].severity == "error"
    assert validate.frame_count(65536)[0].severity == "error"
    assert validate.frame_count(72000) == []


def test_held_samples_reveal_a_lower_true_rate():
    """The Delsys accelerometers, stored at 2000 Hz with every value repeated."""
    x = np.repeat(_head(n=200), 100, axis=0)
    assert any(f.check == "held_samples" for f in validate.held_samples(x))
    assert validate.held_samples(_head()) == []


def test_duplicate_files_are_found_by_content_not_by_name(tmp_path):
    """The rename leftover: same bytes, different name, in no manifest."""
    a = tmp_path / "sverm_2014_sverm30001.tsv"
    b = tmp_path / "sverm_2014_sverm3_0001.tsv"
    c = tmp_path / "other.tsv"
    a.write_text("Time\tx\n0\t1\n")
    b.write_text("Time\tx\n0\t1\n")
    c.write_text("Time\tx\n0\t2\n")
    found = validate.duplicate_files([a, b, c])
    assert len(found) == 1 and found[0].severity == "error"
    assert "sverm_2014_sverm3_0001.tsv" in found[0].message


def test_validate_series_runs_the_applicable_checks_together():
    x = _head()
    x[100:200] = 0.0
    t = np.arange(len(x), dtype=float) / 200
    found = validate.validate_series(x, t=t, documented_hz=200.0, where="rec")
    assert {f.check for f in found} >= {"zero_triplets"}
    assert all(f.where == "rec" for f in found)


def test_raise_on_error_reports_every_error_at_once():
    findings = [
        validate.Finding("a", "error", "first"),
        validate.Finding("b", "warning", "ignored"),
        validate.Finding("c", "error", "second"),
    ]
    assert len(validate.errors(findings)) == 2
    with pytest.raises(ValueError) as e:
        validate.raise_on_error(findings)
    assert "first" in str(e.value) and "second" in str(e.value)
    validate.raise_on_error([validate.Finding("b", "warning", "fine")])


def _speed(n, fs, base=4.0, seed=1):
    rng = np.random.default_rng(seed)
    return np.abs(rng.normal(base, base * 0.15, n))


def test_edge_motion_catches_people_walking_into_position():
    """The 2017 and 2018 Sverm exports: untrimmed, opening with the participants settling."""
    fs = 200.0
    v = _speed(int(400 * fs), fs)
    v[: int(10 * fs)] *= 5
    found = validate.edge_motion(v, fs, where="standstill0012")
    assert len(found) == 1
    assert found[0].severity == "warning" and "first" in found[0].message


def test_edge_motion_catches_a_moving_end_too():
    fs = 200.0
    v = _speed(int(400 * fs), fs)
    v[-int(10 * fs):] *= 6
    found = validate.edge_motion(v, fs)
    assert len(found) == 1 and "last" in found[0].message


def test_a_properly_trimmed_recording_is_silent():
    """The 2021 export, trimmed to begin after everyone had settled."""
    fs = 200.0
    assert validate.edge_motion(_speed(int(400 * fs), fs), fs) == []


def test_edge_motion_declines_to_judge_a_short_recording():
    """Under the baseline window there is no settled interior to compare against."""
    fs = 200.0
    assert validate.edge_motion(_speed(int(30 * fs), fs), fs) == []


def test_settling_time_measures_what_to_trim():
    """A recording that opens moving and closes settled."""
    fs = 200.0
    v = _speed(int(400 * fs), fs)
    v[: int(25 * fs)] *= 6
    head, tail = validate.settling_time(v, fs)
    assert 20 <= head <= 30
    assert tail == 0.0


def test_settling_time_is_zero_at_both_ends_when_already_trimmed():
    fs = 200.0
    assert validate.settling_time(_speed(int(400 * fs), fs), fs) == (0.0, 0.0)


def test_settling_time_reports_the_ceiling_when_it_never_settles():
    fs = 200.0
    v = _speed(int(400 * fs), fs)
    v[: int(150 * fs)] *= 8
    head, _ = validate.settling_time(v, fs, max_s=60.0)
    assert head == 60.0


# --------------------------------------------------------------- marker_average

def _rigid_markers(n=600, dead=(), gap_frac=0.0, seed=0):
    """Three markers on one rigid segment: same motion, small fixed offsets."""
    rng = np.random.default_rng(seed)
    base = np.cumsum(rng.standard_normal((n, 3)) * 0.5, axis=0) + [0, 0, 1800]
    out = {}
    for i, name in enumerate(("HF", "HL", "HR")):
        x = base + [10 * i, 5 * i, 0]
        if name in dead:
            x = np.zeros_like(x)
        elif gap_frac:
            x = x.copy()
            x[: int(gap_frac * n)] = 0.0
        out[name] = x
    return out


def test_marker_average_clean_set_passes():
    assert mm.validate.marker_average(_rigid_markers()) == []


def test_marker_average_flags_a_dead_marker():
    f = mm.validate.marker_average(_rigid_markers(dead=("HF",)))
    assert len(f) == 1 and f[0].severity == "error"
    assert "1 of 3" in f[0].message


def test_marker_average_reports_the_bias_factor():
    """One dead marker of three understates amplitude by a third — that is the whole point."""
    f = mm.validate.marker_average(_rigid_markers(dead=("HF",)))
    assert "33.3 %" in f[0].message and "0.667" in f[0].message


def test_marker_average_bias_matches_a_real_average():
    """The predicted factor should match what a naive nanmean actually does."""
    m = _rigid_markers(dead=("HF",))
    naive = np.nanmean(np.stack(list(m.values())), axis=0)
    repaired = []
    for x in m.values():
        y = x.astype(float).copy()
        y[(y == 0.0).all(axis=1)] = np.nan
        if np.isfinite(y).any():
            repaired.append(y)
    good = np.nanmean(np.stack(repaired), axis=0)
    ratio = np.ptp(naive[:, 0]) / np.ptp(good[:, 0])
    assert abs(ratio - 2 / 3) < 0.02


def test_marker_average_warns_on_a_partly_gapped_marker():
    f = mm.validate.marker_average(_rigid_markers(gap_frac=0.7))
    assert f and all(x.severity == "warning" for x in f)


def test_marker_average_empty_input():
    assert mm.validate.marker_average({}) == []


# ----------------------------------------------------- implausible_position

def _standing(n=600, height=1650.0, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 3)) * 5.0
    x[:, 2] += height
    return x


def test_implausible_position_accepts_a_normal_recording():
    assert mm.validate.implausible_position(_standing()) == []


def test_implausible_position_flags_a_marker_below_the_floor():
    x = _standing()
    x[100:110, 2] = -139.0
    f = mm.validate.implausible_position(x)
    assert len(f) == 1 and f[0].severity == "error"
    assert "10 of 600" in f[0].message


def test_implausible_position_catches_what_zero_triplets_misses():
    """The near miss: close to the origin, but not exactly on it."""
    x = _standing()
    x[50:60] = [-27.2, -124.4, -88.9]          # a real reconstruction near the origin
    assert mm.validate.zero_triplets(x) == []   # not zeros, so the sentinel check is silent
    assert len(mm.validate.implausible_position(x)) == 1


def test_implausible_position_skips_a_low_marker():
    """A foot marker has no standing-height expectation to test against."""
    x = _standing(height=30.0)
    x[10:20, 2] = -500.0
    assert mm.validate.implausible_position(x) == []


def test_implausible_position_flags_impossibly_high_too():
    x = _standing()
    x[200:205, 2] = 9000.0
    assert len(mm.validate.implausible_position(x)) == 1


def test_implausible_position_needs_enough_samples():
    assert mm.validate.implausible_position(_standing(n=50)) == []


# ----------------------------------------------------------- marker_noise

def _sway(n=6000, fs=100.0, mm_amp=6.0, noise_mm=0.0, seed=1):
    """A standing head: slow sway in the micromotion band, plus optional per-sample jitter."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    x = np.column_stack([
        mm_amp * np.sin(2 * np.pi * 0.5 * t),
        mm_amp * np.sin(2 * np.pi * 0.4 * t + 1.0),
        1650.0 + 0.5 * np.sin(2 * np.pi * 0.3 * t),
    ])
    if noise_mm:
        x = x + rng.standard_normal(x.shape) * noise_mm
    return x


def test_marker_noise_accepts_a_clean_recording():
    assert mm.validate.marker_noise(_sway(), 100.0) == []


def test_marker_noise_flags_a_jittering_marker():
    """The failure the other two checks cannot see: every sample plausible, none of them still."""
    x = _sway(noise_mm=1.5)
    assert mm.validate.zero_triplets(x) == []            # no dropped frames
    assert mm.validate.implausible_position(x) == []      # never leaves head height
    f = mm.validate.marker_noise(x, 100.0)
    assert len(f) == 1 and f[0].severity == "error"
    assert "marker jitter" in f[0].message


def test_marker_noise_leaves_a_median_measure_alone():
    """Why it matters: the jitter destroys a summing measure and not a median one."""
    clean, noisy = _sway(), _sway(noise_mm=1.5)
    q_clean = float(np.median(mm.speed_from_position(clean, 100.0, unit="mm")))
    q_noisy = float(np.median(mm.speed_from_position(noisy, 100.0, unit="mm")))
    assert abs(q_noisy - q_clean) / q_clean < 0.5        # band-limited measure barely moves
    path_clean = np.linalg.norm(np.diff(clean, axis=0), axis=1).sum()
    path_noisy = np.linalg.norm(np.diff(noisy, axis=0), axis=1).sum()
    assert path_noisy > 5 * path_clean                   # the summing measure is wrecked


def test_marker_noise_too_short_to_judge():
    assert mm.validate.marker_noise(_sway(n=20), 100.0) == []
