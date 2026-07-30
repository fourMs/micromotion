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
