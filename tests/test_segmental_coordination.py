"""Segmental coordination: many markers on ONE axis, and what their agreement means.

The distinction these pin is between projecting each marker onto its own principal axis
and projecting all of them onto a shared one. Only the second answers "do these segments
move together": two markers swaying along axes 90 degrees apart, each projected onto its
own axis, correlate perfectly while sharing no direction of motion at all.
"""
import numpy as np
import pytest

from micromotion import segmental_coordination, shared_axis_projection


def _along(direction, signal):
    """A marker whose horizontal trace runs along `direction`, scaled by `signal`."""
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    return np.asarray(signal, float)[:, None] * d[None, :]


def _wave(n=600, freq=3.0, seed=None):
    t = np.linspace(0, 1, n)
    x = np.sin(2 * np.pi * freq * t)
    if seed is not None:
        x = x + 0.01 * np.random.default_rng(seed).standard_normal(n)
    return x


# ---------------------------------------------------------------------------
# shared_axis_projection
# ---------------------------------------------------------------------------


class TestSharedAxisProjection:

    def test_projects_every_marker_onto_the_reference_axis(self):
        """A marker on the reference's axis keeps its amplitude; one across it loses it."""
        s = _wave()
        markers = {"ref": _along([1.0, 0.0], s),
                   "same": _along([1.0, 0.0], 0.5 * s),
                   "across": _along([0.0, 1.0], s)}

        out = shared_axis_projection(markers, reference="ref")

        np.testing.assert_allclose(out["projection"]["ref"], s - s.mean(), atol=1e-9)
        np.testing.assert_allclose(out["projection"]["same"], 0.5 * (s - s.mean()), atol=1e-9)
        assert np.std(out["projection"]["across"]) < 1e-9

    def test_the_axis_is_the_reference_marker_s_own(self):
        """Not each marker's own axis -- that is what principal_axis_projection does."""
        s = _wave()
        markers = {"ref": _along([1.0, 0.0], s), "across": _along([0.0, 1.0], s)}

        out = shared_axis_projection(markers, reference="ref")

        assert out["axis_deg"] == pytest.approx(0.0, abs=1e-6) or \
               out["axis_deg"] == pytest.approx(180.0, abs=1e-6)

    def test_two_markers_at_right_angles_do_not_correlate_on_a_shared_axis(self):
        """The whole point: per-marker axes would call these identical."""
        s = _wave()
        markers = {"ref": _along([1.0, 0.0], s), "across": _along([0.0, 1.0], s)}

        out = shared_axis_projection(markers, reference="ref")
        a, b = out["projection"]["ref"], out["projection"]["across"]

        assert np.std(b) < 1e-9          # nothing of it survives on the shared axis
        assert np.std(a) > 0.5

    def test_a_missing_reference_marker_says_which_one(self):
        markers = {"a": _along([1.0, 0.0], _wave())}
        with pytest.raises(KeyError, match="head"):
            shared_axis_projection(markers, reference="head")


# ---------------------------------------------------------------------------
# segmental_coordination
# ---------------------------------------------------------------------------


class TestSegmentalCoordination:

    def test_a_rigid_body_has_one_effective_degree_of_freedom(self):
        """Every segment the same motion on one axis: PC1 ~ 1, effective DOF ~ 1."""
        s = _wave()
        markers = {n: _along([1.0, 0.0], k * s)
                   for n, k in [("head", 1.0), ("chest", 0.8), ("hip", 0.5), ("knee", 0.2)]}

        out = segmental_coordination(markers, reference="head")

        assert out["pc1_fraction"] == pytest.approx(1.0, abs=1e-6)
        assert out["effective_dof"] == pytest.approx(1.0, abs=1e-6)

    def test_independent_segments_raise_the_effective_degrees_of_freedom(self):
        """Four segments moving independently along the axis: effective DOF near four."""
        rng = np.random.default_rng(0)
        markers = {n: _along([1.0, 0.0], rng.standard_normal(4000))
                   for n in ["head", "chest", "hip", "knee"]}

        out = segmental_coordination(markers, reference="head")

        assert out["effective_dof"] > 3.5
        assert out["pc1_fraction"] < 0.4

    def test_amplitude_ratio_reports_which_segment_sways_more(self):
        """The inverted-pendulum test is a ratio; above 1 means the first sways more."""
        s = _wave()
        markers = {"head": _along([1.0, 0.0], 2.0 * s), "hip": _along([1.0, 0.0], 1.0 * s)}

        out = segmental_coordination(markers, reference="head",
                                     ratios=[("head", "hip")])

        assert out["amplitude_ratio"][("head", "hip")] == pytest.approx(2.0, rel=1e-6)

    def test_correlations_are_nan_aware_and_report_their_n(self):
        """A dropout in one marker must not void the pair, and n must say what was used."""
        s = _wave(seed=1)
        head = _along([1.0, 0.0], s)
        hip = _along([1.0, 0.0], 0.9 * s)
        hip[100:150] = np.nan                       # a dropout in the hip only

        out = segmental_coordination({"head": head, "hip": hip}, reference="head")

        r = out["correlation"][("head", "hip")]
        assert np.isfinite(r) and r > 0.99
        assert out["n"][("head", "hip")] == len(s) - 50

    def test_reference_mask_is_the_default_and_own_mask_is_offered(self):
        """The study masks every marker by the reference's valid samples; both are available."""
        s = _wave(seed=2)
        head = _along([1.0, 0.0], s)
        hip = _along([1.0, 0.0], 0.9 * s)
        head[200:260] = np.nan                      # invalid on the REFERENCE

        ref_masked = segmental_coordination({"head": head, "hip": hip}, reference="head",
                                            mask="reference")
        own_masked = segmental_coordination({"head": head, "hip": hip}, reference="head",
                                            mask="own")

        # hip is finite throughout, so its own mask keeps every sample ...
        assert own_masked["n_finite"]["hip"] == len(s)
        # ... while the reference mask drops the frames the head lost.
        assert ref_masked["n_finite"]["hip"] == len(s) - 60

    def test_an_unknown_mask_mode_is_refused(self):
        s = _wave()
        markers = {"head": _along([1.0, 0.0], s), "hip": _along([1.0, 0.0], s)}
        with pytest.raises(ValueError, match="mask"):
            segmental_coordination(markers, reference="head", mask="whatever")

    def test_effective_dof_uses_the_values_not_their_ranks(self):
        """Pinned because the choice is load-bearing and invisible in most data.

        effective_dimensionality defaults to rank=True, which is right for heavy-tailed
        descriptors and wrong for sway signals: ranking discards amplitude, and amplitude is
        the whole content of a projection. On a segment that is a monotone but non-linear
        function of another, ranks call the two identical while their actual covariance does
        not -- so the expected answer here is computed from the values.
        """
        s = _wave(n=1200, freq=2.0)
        markers = {"head": _along([1.0, 0.0], s),
                   "chest": _along([1.0, 0.0], np.sign(s) * np.abs(s) ** 5)}

        out = segmental_coordination(markers, reference="head")

        cols = np.column_stack([s - s.mean(),
                                (np.sign(s) * np.abs(s) ** 5) - (np.sign(s) * np.abs(s) ** 5).mean()])
        ev = np.linalg.eigvalsh(np.corrcoef(cols, rowvar=False))[::-1]
        assert out["pc1_fraction"] == pytest.approx(ev[0] / ev.sum(), rel=1e-9)
        assert out["effective_dof"] == pytest.approx(ev.sum() ** 2 / (ev ** 2).sum(), rel=1e-9)
        # and the rank answer, which this must NOT be, is a different number
        assert out["pc1_fraction"] < 0.98

    def test_a_marker_can_be_ratioed_without_entering_the_reduction(self):
        """A derived marker (a midpoint, say) must not silently add a column to the PCA.

        Found against real data: the still-standing analysis ratios the head against the
        MIDPOINT of the two hips, and passing that midpoint in as a marker changed the
        effective degrees of freedom, because it entered the reduction as a ninth segment
        that is a linear combination of two others already in it.
        """
        s = _wave(seed=3)
        markers = {"head": _along([1.0, 0.0], s),
                   "hipL": _along([1.0, 0.0], 0.5 * s),
                   "hipR": _along([1.0, 0.0], 0.4 * s)}
        segments = list(markers)
        markers["hipmid"] = 0.5 * (markers["hipL"] + markers["hipR"])

        without = segmental_coordination({k: markers[k] for k in segments}, reference="head")
        with_ratio = segmental_coordination(markers, reference="head",
                                            ratios=[("head", "hipmid")], reduce=segments)

        assert with_ratio["effective_dof"] == pytest.approx(without["effective_dof"], rel=1e-12)
        assert with_ratio["pc1_fraction"] == pytest.approx(without["pc1_fraction"], rel=1e-12)
        assert with_ratio["used"] == without["used"]
        assert np.isfinite(with_ratio["amplitude_ratio"][("head", "hipmid")])
