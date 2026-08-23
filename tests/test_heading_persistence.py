"""Heading persistence: known answers at both ends of the scale, and in the middle.

The descriptor measures how much a trace's direction changes from step to step, and the
intuition about its sign runs backwards. A sway that reverses along one line reads near +1,
because it holds its heading for the whole of each excursion and turns only at the ends.
Reaching -1 takes a zigzag that flips at the sampling rate. These tests pin that, because a
reader who assumes "reversing means negative" would otherwise change the code to match.
"""
import numpy as np
import pytest

from micromotion import heading_persistence


def _sway_along_a_line(n=500, cycles=5):
    t = np.linspace(0, cycles, n)
    return np.column_stack([np.sin(2 * np.pi * t), np.zeros(n)])


class TestKnownAnswers:
    def test_a_straight_walk_is_maximally_persistent(self):
        xy = np.column_stack([np.linspace(0, 10, 500), np.zeros(500)])
        assert heading_persistence(xy)["persistence"] == pytest.approx(1.0, abs=1e-9)

    def test_a_zigzag_that_flips_every_step_is_minus_one(self):
        xy = np.column_stack([np.arange(400) % 2, np.zeros(400)])
        assert heading_persistence(xy)["persistence"] == pytest.approx(-1.0, abs=1e-9)

    def test_a_random_walk_is_about_zero(self):
        xy = np.random.default_rng(0).standard_normal((2000, 2)).cumsum(axis=0)
        assert abs(heading_persistence(xy)["persistence"]) < 0.1

    def test_sway_along_one_line_reads_near_plus_one_not_minus_one(self):
        """The reversal happens at the turning points only, a few steps among hundreds."""
        assert heading_persistence(_sway_along_a_line())["persistence"] > 0.9


class TestStraightness:
    def test_a_straight_walk_is_one(self):
        xy = np.column_stack([np.linspace(0, 10, 500), np.zeros(500)])
        assert heading_persistence(xy)["straightness"] == pytest.approx(1.0, abs=1e-9)

    def test_returning_to_the_start_is_zero(self):
        """Distance travelled without getting anywhere, which is what standing still is."""
        assert heading_persistence(_sway_along_a_line())["straightness"] < 1e-6

    def test_a_random_walk_lies_between(self):
        xy = np.random.default_rng(1).standard_normal((2000, 2)).cumsum(axis=0)
        s = heading_persistence(xy)["straightness"]
        assert 0.0 < s < 1.0


class TestDegenerateInput:
    def test_constant_step_speed_is_not_filtered_away(self):
        """A strict comparison against the percentile excluded every step of a trace whose
        steps are all the same length, which is exactly the zigzag case."""
        xy = np.column_stack([np.arange(400) % 2, np.zeros(400)])
        assert np.isfinite(heading_persistence(xy)["persistence"])

    @pytest.mark.parametrize("n", [0, 1, 2])
    def test_too_few_samples_gives_nan_rather_than_raising(self, n):
        out = heading_persistence(np.zeros((n, 2)))
        assert np.isnan(out["persistence"]) and np.isnan(out["straightness"])

    def test_non_finite_rows_are_dropped(self):
        xy = np.column_stack([np.linspace(0, 10, 500), np.zeros(500)])
        holed = xy.copy()
        holed[100] = np.nan
        assert heading_persistence(holed)["persistence"] == pytest.approx(
            heading_persistence(xy)["persistence"], abs=1e-6)

    def test_a_stationary_trace_does_not_divide_by_zero(self):
        out = heading_persistence(np.zeros((100, 2)))
        assert np.isnan(out["straightness"])
