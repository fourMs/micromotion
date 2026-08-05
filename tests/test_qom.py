"""Quantity of motion against signals whose answer is known in closed form.

For a single axis moving as x(t) = A sin(2*pi*f*t), the speed is |A*2*pi*f*cos(2*pi*f*t)|
and its mean over whole cycles is A*2*pi*f*(2/pi). Both the acceleration route and the
position route must recover that number, which is what makes them comparable at all.
"""

import numpy as np
import pytest

import micromotion as mm
from micromotion import filters  # noqa: F401


def sine(amp_mm, f, fs, dur_s):
    t = np.arange(0, dur_s, 1 / fs)
    x = amp_mm * np.sin(2 * np.pi * f * t)
    a = -amp_mm * (2 * np.pi * f) ** 2 * np.sin(2 * np.pi * f * t) / 1000.0  # m/s^2
    return t, x, a


def expected_mean_speed(amp_mm, f):
    return amp_mm * 2 * np.pi * f * 2 / np.pi


@pytest.mark.parametrize("f", [0.5, 1.0, 2.0, 3.0])
def test_position_route_recovers_analytic_speed(f):
    fs, amp = 200.0, 3.0
    _, x, _ = sine(amp, f, fs, 120)
    r = mm.qom(x, fs, kind="position", unit="mm")
    trim = r.edge_samples
    got = np.mean(r.speed[trim:-trim])
    assert got == pytest.approx(expected_mean_speed(amp, f), rel=0.02)


@pytest.mark.parametrize("f", [0.5, 1.0, 2.0, 3.0])
def test_acceleration_route_recovers_analytic_speed(f):
    fs, amp = 200.0, 3.0
    _, _, a = sine(amp, f, fs, 120)
    r = mm.qom(a, fs, kind="acceleration", unit="m/s^2")
    trim = r.edge_samples
    got = np.mean(r.speed[trim:-trim])
    assert got == pytest.approx(expected_mean_speed(amp, f), rel=0.02)


def test_the_two_routes_agree_with_each_other():
    """The claim that lets optical and accelerometer collections sit in one table."""
    fs, amp, f = 200.0, 2.5, 1.5
    _, x, a = sine(amp, f, fs, 120)
    rp = mm.qom(x, fs, kind="position", unit="mm")
    ra = mm.qom(a, fs, kind="acceleration", unit="m/s^2")
    trim = max(rp.edge_samples, ra.edge_samples)
    assert np.mean(rp.speed[trim:-trim]) == pytest.approx(
        np.mean(ra.speed[trim:-trim]), rel=0.02
    )


def test_g_units_are_converted():
    """The 9.80665x error that was live in this project until 2026-07-28."""
    fs, amp, f = 200.0, 3.0, 1.0
    _, _, a = sine(amp, f, fs, 120)
    si = mm.qom(a, fs, unit="m/s^2")
    in_g = mm.qom(a / mm.G, fs, unit="g")
    assert in_g.mean_mm_s == pytest.approx(si.mean_mm_s, rel=1e-9)


def test_out_of_band_motion_is_rejected():
    """A 15 Hz tremor is not micromotion and must not enter the measure."""
    fs = 200.0
    _, x_in, _ = sine(3.0, 1.0, fs, 120)
    _, x_out, _ = sine(3.0, 15.0, fs, 120)
    both = mm.qom(x_in + x_out, fs, kind="position")
    only_in = mm.qom(x_in, fs, kind="position")
    assert both.mean_mm_s == pytest.approx(only_in.mean_mm_s, rel=0.05)


def test_drift_does_not_survive_integration():
    """A constant offset on an accelerometer integrates to a ramp that would dominate."""
    fs, amp, f = 200.0, 3.0, 1.0
    _, _, a = sine(amp, f, fs, 300)
    clean = mm.qom(a, fs)
    biased = mm.qom(a + 0.05, fs)
    trim = clean.edge_samples
    assert np.mean(biased.speed[trim:-trim]) == pytest.approx(
        np.mean(clean.speed[trim:-trim]), rel=0.02
    )


def test_cardiac_peak_finds_a_known_heart_rate():
    """1.2 Hz is 72 bpm.

    Validated on real data too: across nine StillStanding365 days the peak found in the
    phone's acceleration magnitude matches the wrist heart rate at a median ratio of 0.99.
    """
    fs = 200.0
    t = np.arange(0, 300, 1 / fs)
    rng = np.random.default_rng(0)
    x = np.sin(2 * np.pi * 1.2 * t) + 0.5 * rng.normal(size=len(t))
    assert mm.cardiac_peak(x, fs) == pytest.approx(1.2, abs=0.03)


def test_compensated_variant_removes_respiration():
    """Breathing at 0.4 Hz is inside the raw band and below the compensated one.

    Twenty-four breaths a minute sits above the 0.3 Hz lower edge, so the raw variant counts
    it as movement, and below the 0.5 Hz edge the compensated variant raises to exclude it.
    Postural motion at 4 Hz survives both.
    """
    fs = 200.0
    t = np.arange(0, 300, 1 / fs)

    def acc(amp_mm, f):
        return -amp_mm * (2 * np.pi * f) ** 2 * np.sin(2 * np.pi * f * t) / 1000.0

    sig = np.column_stack([acc(30.0, 0.4) + acc(0.5, 4.0)] * 3)
    raw = mm.qom(sig, fs, variant="raw")
    comp = mm.qom(sig, fs, variant="compensated")
    assert comp.mean_mm_s < 0.8 * raw.mean_mm_s


def test_position_rejects_accelerometer_only_variants():
    fs = 200.0
    _, x, _ = sine(3.0, 1.0, fs, 60)
    with pytest.raises(ValueError, match="accelerometers only"):
        mm.qom(x, fs, kind="position", variant="tilt_corrected")


def test_binning_flags_the_partial_final_bin():
    """Needs a recording long enough to have a clean middle.

    At the 0.2 Hz lower edge the filter transient runs 40 s at each end, so anything under
    about two minutes is edge all the way through. That is a real constraint of the band, not
    an artefact: a short recording cannot be band-limited this low and still have an interior.
    """
    fs = 100.0
    _, x, _ = sine(3.0, 1.0, fs, 302.5)
    b = mm.qom(x, fs, kind="position").binned(5.0)
    assert b.iloc[-1].edge == "partial"
    assert (b.edge == "ok").sum() > 0


def test_both_integration_rules_are_available_and_differ():
    """Both are in use in the source corpus, and they are not interchangeable.

    The rectangle sum lags the signal by half a sample, so it reads slightly low. The gap is
    about a quarter of a per cent on real data: small, systematic, and not noise.
    """
    fs = 200.0
    t = np.arange(0, 120, 1 / fs)
    rng = np.random.default_rng(0)
    a = np.column_stack([
        -3.0 * (2 * np.pi * 1.0) ** 2 * np.sin(2 * np.pi * 1.0 * t) / 1000.0
        + 0.002 * rng.normal(size=len(t)) for _ in range(3)])
    rect = mm.qom(a, fs, integrate="rectangle").mean_mm_s
    trap = mm.qom(a, fs, integrate="trapezoid").mean_mm_s
    assert rect != trap
    assert abs(trap / rect - 1) < 0.02


def test_unknown_integration_rule_is_refused():
    fs = 100.0
    a = np.zeros((1000, 3))
    with pytest.raises(ValueError, match="unknown rule"):
        mm.qom(a, fs, integrate="simpson")


def test_very_low_band_against_a_high_rate_warns():
    """The silent failure this guard exists for: 0.1-0.5 Hz at 250 Hz.

    In transfer-function form that design has a pole radius of 0.9979 and a measured passband
    gain of 0.84 at 0.15 Hz where it should be 0.99 -- sixteen per cent low, with nothing
    raised and nothing visibly wrong. Second-order sections give 0.9875.
    """
    fs = 250.0
    x = np.sin(2 * np.pi * 0.15 * np.arange(0, 600, 1 / fs))
    with pytest.warns(RuntimeWarning, match="very low against"):
        y = mm.bandpass(x, fs, 0.1, 0.5, order=3)
    assert np.std(y) / np.std(x) == pytest.approx(0.99, abs=0.02)


def test_ordinary_bands_do_not_warn():
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("error")
        mm.bandpass(np.random.default_rng(0).normal(size=4000), 200.0, 0.3, 10.0)


def test_nyquist_margin_is_overridable():
    """The corpus uses 0.999 where this package defaults to 0.99; both must be reachable."""
    fs = 20.0
    t = np.arange(0, 300, 1 / fs)
    rng = np.random.default_rng(0)
    x = np.sin(2 * np.pi * 2.0 * t) + 0.2 * rng.normal(size=len(t))
    a = mm.lowpass(x, fs, 10.0)                     # clamped to 9.9
    b = mm.lowpass(x, fs, 10.0, margin=0.999)       # clamped to 9.99
    assert not np.allclose(a, b)
    assert np.corrcoef(a, b)[0, 1] > 0.999


def test_margin_reaches_bandpass_too():
    fs = 20.0
    x = np.random.default_rng(0).normal(size=4000)
    a = mm.bandpass(x, fs, 0.3, 10.0)
    b = mm.bandpass(x, fs, 0.3, 10.0, margin=0.999)
    assert not np.allclose(a, b)


def test_short_recordings_are_all_edge_at_this_band():
    """The cost of the 0.2 Hz lower edge, made explicit rather than discovered later."""
    fs = 100.0
    _, x, _ = sine(3.0, 1.0, fs, 60.0)
    b = mm.qom(x, fs, kind="position").binned(5.0)
    assert (b.edge == "ok").sum() == 0
    assert mm.filters.edge_transient_samples(fs) / fs == pytest.approx(40.0, abs=1.0)


def test_mgt_band_limited_qom_reads_high_and_we_know_exactly_why():
    """Pin the MGT-versus-micromotion gap, and its attribution, so neither drifts.

    `band_limited_qom` band-limits the position and not the speed derived from it, so the
    velocity keeps energy above the stated upper edge. That, and not the differentiation rule,
    is the whole of the difference. Both are kept: MGT's figures must keep reproducing.
    """
    import numpy as np
    from micromotion import filters

    fs = 200.0
    rng = np.random.default_rng(7)
    pos = np.cumsum(rng.normal(0, 0.05, (24000, 3)), axis=0) + [0.0, 0.0, 1700.0]

    def speed(diff, second):
        p = filters.bandpass(pos, fs, 0.3, 10.0)
        v = np.diff(p, axis=0) * fs if diff == "first" else mm.derivative(p, fs)
        if second:
            v = filters.bandpass(v, fs, 0.3, 10.0)
        return float(np.mean(np.linalg.norm(v, axis=1)))

    mgt_recipe = speed("first", False)
    only_diff_changed = speed("central", False)
    only_second_pass_added = speed("first", True)
    mm_recipe = speed("central", True)

    # The differentiation rule is not where the difference lives.
    assert abs(only_diff_changed - mgt_recipe) / mgt_recipe < 0.005

    # The second band-pass is, and it lowers the result.
    assert only_second_pass_added < mgt_recipe
    assert abs(only_second_pass_added - mm_recipe) / mm_recipe < 0.005

    # The package's own function agrees with its recipe, and MGT's with its own.
    assert mm.speed_from_position(pos, fs, lo=0.3, hi=10.0).mean() == pytest.approx(
        mm_recipe, rel=1e-6)
    mgt, _ = mm.band_limited_qom(pos, fs, lo=0.3, hi=10.0)
    assert float(np.mean(mgt)) == pytest.approx(mgt_recipe, rel=1e-6)


def test_every_band_default_comes_from_filters_BAND():
    """No function may carry its own copy of the band.

    Until 2026-07-31 four functions defaulted to 0.3-15 Hz while `filters.BAND` was
    (0.2, 10.0) and `qom`/`speed_from_*` used it. Two bands in one package means a number can be
    quoted from the wrong one, and the documentation had already drifted to match the wrong one in
    five places. `pose_qom` and `normalized_qom` keep hi=5.0 deliberately -- image-space landmark
    jitter dominates above it -- but their lower edge is the canonical one like everything else.
    """
    import inspect
    import micromotion as mm
    from micromotion import filters

    lo, hi = filters.BAND
    for fn in (mm.band_limited_qom, mm.group_qom, mm.pose_qom, mm.normalized_qom,
               mm.speed_from_position, mm.speed_from_acceleration):
        p = inspect.signature(fn).parameters
        if "lo" in p:
            assert p["lo"].default == lo, f"{fn.__name__} lo={p['lo'].default} != {lo}"
        if "hi" in p and fn not in (mm.pose_qom, mm.normalized_qom):
            assert p["hi"].default == hi, f"{fn.__name__} hi={p['hi'].default} != {hi}"


def test_docstrings_do_not_state_a_band_the_code_does_not_use():
    """The prose must agree with the signature, not only the signature with BAND.

    The test above checks that every default equals `filters.BAND`. It passed for five days while
    two docstrings told the reader the upper edge defaulted to 15.0, which had been true before
    2026-07-31 and was not after. Those strings render straight into the generated API page, so a
    reader of the documentation got the old band from a package whose code used the new one.

    This reads the docstring and compares what it claims against what the function actually does.
    """
    import inspect
    import re
    import micromotion as mm

    checked = 0
    for fn in (mm.band_limited_qom, mm.group_qom, mm.pose_qom, mm.normalized_qom,
               mm.speed_from_position, mm.speed_from_acceleration):
        doc = inspect.getdoc(fn) or ""
        params = inspect.signature(fn).parameters
        for edge in ("lo", "hi"):
            if edge not in params:
                continue
            # "lo (float, optional): ... Defaults to <something>." up to the next parameter.
            m = re.search(rf"^\s*{edge} \(.*?(?=^\s*\w+ \(|\Z)", doc, re.S | re.M)
            if not m:
                continue
            claim = re.search(r"Defaults to ([^.\n]*(?:\.\d+)?)", m.group(0))
            if not claim:
                continue
            text = claim.group(1)
            numbers = [float(x) for x in re.findall(r"\d+\.\d+", text)]
            if not numbers:
                continue          # states the constant by name only, which cannot go stale
            checked += 1
            assert params[edge].default in numbers, (
                f"{fn.__name__} {edge}: docstring says 'Defaults to {text}', "
                f"signature says {params[edge].default}")
    assert checked, "no docstring stated a band edge as a number; the test checked nothing"


# The value 0.15.2 returned for _occluded_group(seed=5); the escape hatch is pinned to it.
# The tolerance is relative and loose enough for filter arithmetic to differ between platforms:
# this same value came out 1.3e-7 lower on one CI runner and 1.3e-8 higher on another, because
# `filtfilt` is not bit-reproducible across scipy builds. A first version pinned it to 1e-9
# absolute and failed on all four Pythons while passing locally, which is a test pinned to a
# machine rather than to a claim. What is being asserted is that the escape hatch still performs
# the old computation, and 1 part in 10^6 says that far more tightly than the 16 per cent bias it
# exists to distinguish itself from.
PRE_1_0_WORN = 76.40629424709411
PRE_1_0_RTOL = 1e-6


def _occluded_group(fs=100.0, n=12000, nm=12, seed=5):
    """A group of markers on a shared trajectory, with realistic short dropouts."""
    import numpy as np
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    base = np.stack([np.sin(2 * np.pi * 0.8 * t), np.cos(2 * np.pi * 0.6 * t),
                     0.3 * np.sin(2 * np.pi * 1.1 * t)], 1) * 20
    clean = np.repeat(base[:, None, :], nm, axis=1) + rng.normal(0, 2, (n, nm, 3))
    occ = clean.copy()
    for m in range(nm):
        i = 0
        while i < n:
            i += int(rng.exponential(0.9 * fs))
            if i >= n:
                break
            gap = int(rng.exponential(0.45 * fs)) + 5
            occ[i:i + gap, m, :] = np.nan
            i += gap
    return clean, occ


def test_group_qom_default_is_not_confounded_by_occlusion():
    """The 1.0 default must recover the unoccluded value; the old one must not.

    Both halves matter. Asserting only that `visible` is close to the truth would pass on an
    implementation that ignored occlusion entirely, so the test also pins that `worn` is
    materially wrong on the same data. That is the behaviour this default was changed to escape.
    """
    import numpy as np
    import micromotion as mm

    clean, occ = _occluded_group()
    truth, _, _ = mm.group_qom(clean, 100.0)
    visible, series, fs_out = mm.group_qom(occ, 100.0)
    worn, _, _ = mm.group_qom(occ, 100.0, normalize="worn")

    assert abs(visible - truth) / truth < 0.03, (visible, truth)
    assert (truth - worn) / truth > 0.08, f"worn={worn} truth={truth}: the old bias is gone?"

    coverage = np.isfinite(occ[:, :, 0]).sum(1)[:len(series)]
    ok = np.isfinite(series)
    r = np.corrcoef(series[ok], coverage[:len(series)][ok])[0, 1]
    assert abs(r) < 0.08, f"the default still tracks marker coverage, r={r}"


def test_group_qom_visible_survives_decimation():
    """At 200 Hz `band_limited_qom` decimates, so the presence mask must move with it.

    A mask left at the input rate would be wrong by the decimation factor and would silently
    mask the wrong frames, which is worse than not masking at all.
    """
    import micromotion as mm

    clean, occ = _occluded_group(fs=200.0, n=24000, seed=9)
    truth, _, _ = mm.group_qom(clean, 200.0)
    visible, _, fs_out = mm.group_qom(occ, 200.0)
    assert fs_out < 200.0, "expected decimation at 200 Hz; this test is no longer exercising it"
    assert abs(visible - truth) / truth < 0.03, (visible, truth, fs_out)


def test_group_qom_worn_reproduces_the_pre_1_0_number():
    """`normalize="worn"` is the escape hatch for published figures, so pin it to a value.

    Computed with the 0.15.2 implementation on this exact input, compared with a relative
    tolerance because `filtfilt` is not bit-reproducible across scipy builds.
    """
    import micromotion as mm

    _, occ = _occluded_group(seed=5)
    worn, _, _ = mm.group_qom(occ, 100.0, normalize="worn")
    assert abs(worn - PRE_1_0_WORN) / PRE_1_0_WORN < PRE_1_0_RTOL, f"{worn} != {PRE_1_0_WORN}"


def test_group_qom_rejects_an_unknown_normalisation():
    import numpy as np
    import pytest
    import micromotion as mm
    with pytest.raises(ValueError):
        mm.group_qom(np.zeros((500, 3, 3)), 100.0, normalize="average")


def test_low_rate_narrows_the_band_and_says_so():
    """A rate below twice the upper edge silently changes the measurement unless it warns.

    8 Hz data cannot carry a 5 Hz upper edge: Nyquist is 4, so the band becomes 0.2-3.96. The
    result is still meaningful, but it is not the same measurement as one from a faster recording
    and must not be compared with it. Analysis at 10 Hz is supported; quiet substitution is not.
    """
    import warnings
    import numpy as np
    import micromotion as mm

    assert mm.effective_band(100.0) == (0.2, 5.0)
    lo, hi = mm.effective_band(8.0)
    assert lo == 0.2 and hi < 5.0

    x = np.random.default_rng(0).normal(size=(600, 3))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        v = mm.speed_from_position(x, 10.0, unit="mm")
    assert np.isfinite(v).all()                      # it still works
    assert any("clamped" in str(x.message) for x in w)

    with warnings.catch_warnings(record=True) as w:   # and is silent when the rate is adequate
        warnings.simplefilter("always")
        mm.speed_from_position(np.random.default_rng(0).normal(size=(6000, 3)), 100.0, unit="mm")
    assert not any("clamped" in str(x.message) for x in w)


def test_velocity_functions_are_the_speed_functions_before_the_norm():
    """The speed helpers must stay defined as the norm of the velocity helpers.

    They were one function each until the velocity form was split out, and the split is only
    safe while the two cannot drift: a descriptor that needs the velocity vector (jerk,
    anything directional) has to agree with the scalar QoM that is reported alongside it.
    """
    rng = np.random.default_rng(11)
    x = rng.normal(size=(3000, 3))
    for speed, velocity, unit in (
        (mm.speed_from_position, mm.velocity_from_position, "mm"),
        (mm.speed_from_acceleration, mm.velocity_from_acceleration, "m/s^2"),
    ):
        v = velocity(x, 100.0, unit)
        assert v.shape == x.shape
        assert np.allclose(speed(x, 100.0, unit), np.linalg.norm(v, axis=1), atol=0, rtol=0)


def test_acceleration_unit_is_applied_not_ignored():
    """g and m/s^2 differ by 9.80665, and a silently ignored unit is a factor-10 error."""
    rng = np.random.default_rng(12)
    x = rng.normal(size=(3000, 3))
    in_g = mm.velocity_from_acceleration(x, 100.0, "g")
    in_si = mm.velocity_from_acceleration(x, 100.0, "m/s^2")
    assert np.allclose(in_g, in_si * mm.G)



@pytest.mark.parametrize("f", [1.0, 3.0, 5.0])
def test_wideband_reaches_where_the_canonical_band_stops(f):
    """WIDEBAND must be a real measurement above BAND's ceiling, not a relabelling.

    Probes stay at or below 5 Hz because a zero-phase fourth-order Butterworth is already
    rolling off well before its nominal corner -- see the rolloff table in
    ``docs/conventions.md``. At 6 Hz the 10 Hz band reads 2 per cent low and at 7 Hz nearly
    9 per cent, which is the filter behaving, not a defect.
    """
    fs, amp = 200.0, 3.0
    _, x, _ = sine(amp, f, fs, 120)
    wide = mm.qom(x, fs, kind="position", unit="mm", band="wideband")
    trim = wide.edge_samples
    assert np.mean(wide.speed[trim:-trim]) == pytest.approx(expected_mean_speed(amp, f), rel=0.02)


def test_the_canonical_band_actually_rejects_above_its_ceiling():
    """A 5 Hz tone must survive the wide band and be cut by the canonical one.

    This is what makes the two bands different measurements rather than two names.
    """
    fs, amp, f = 200.0, 3.0, 5.0
    _, x, _ = sine(amp, f, fs, 120)
    wide = mm.qom(x, fs, kind="position", unit="mm", band="wideband")
    narrow = mm.qom(x, fs, kind="position", unit="mm")
    t = wide.edge_samples
    assert np.mean(narrow.speed[t:-t]) < 0.4 * np.mean(wide.speed[t:-t])


def test_the_canonical_ceiling_is_deliverable_by_every_corpus_rate():
    """The reason the ceiling is 5 Hz: every instrument in the corpus must reach it.

    The slowest is the phone accelerometer at about 15 Hz. If a future change raises the
    ceiling above half that, the band stops being computable on a whole collection.
    """
    slowest_sensor_hz = 14.75
    assert mm.BAND[1] <= slowest_sensor_hz / 2
    assert mm.effective_band(slowest_sensor_hz) == mm.BAND


def test_effective_dimensionality_counts_independent_axes():
    """Three latent axes replicated into nine columns must read as about three, not nine."""
    rng = np.random.default_rng(3)
    latent = rng.normal(size=(600, 3))
    x = np.hstack([latent + 0.01 * rng.normal(size=(600, 3)) for _ in range(3)])
    r = mm.effective_dimensionality(x)
    assert 2.5 < r["participation_ratio"] < 3.5
    assert r["n_for_80"] <= 3

    independent = rng.normal(size=(600, 9))
    assert mm.effective_dimensionality(independent)["participation_ratio"] > 7.0


def test_intraclass_correlation_separates_trait_from_state():
    """A measure driven by the person must give a high ICC; one driven by noise, a low one."""
    # statsmodels is an optional extra, so a plain `pip install micromotion` cannot run this.
    # Without the guard the suite FAILS rather than skips on the default install, which reads as
    # a broken package to anyone checking their own environment.
    pytest.importorskip("statsmodels", reason="intraclass_correlation needs the [mixed] extra")
    rng = np.random.default_rng(4)
    people = np.repeat(np.arange(12), 20)
    offsets = rng.normal(0, 3.0, size=12)[people]
    trait = mm.intraclass_correlation(offsets + rng.normal(0, 0.5, size=len(people)), people)
    state = mm.intraclass_correlation(rng.normal(0, 1.0, size=len(people)), people)
    assert trait["icc"] > 0.8 and not trait["boundary"]
    assert state["icc"] < 0.2
    assert trait["n_groups"] == 12


def test_intraclass_correlation_flags_a_boundary_estimate():
    """Zero between-group variance is the optimiser at an edge, not a measured zero."""
    pytest.importorskip("statsmodels", reason="intraclass_correlation needs the [mixed] extra")
    rng = np.random.default_rng(5)
    people = np.repeat(np.arange(4), 25)
    r = mm.intraclass_correlation(rng.normal(0, 1.0, size=len(people)), people)
    if r["var_between"] <= 1e-9:
        assert r["boundary"]


def test_band_limited_qom_accepts_optical_legacy_band():
    """`lo=None` is a pure low-pass, and the package's own constant must be usable.

    `OPTICAL_LEGACY_BAND` is `(None, 10.0)` and `filters.bandpass` has always honoured a `None`
    lower edge, but `band_limited_qom` compared `0 < lo` and raised a TypeError on it -- so
    `band_limited_qom(x, fs, *OPTICAL_LEGACY_BAND)` failed on a constant the package exports.
    Several analyses in the source corpus work at that band deliberately, because it is what the
    older optical standstill studies used.
    """
    import numpy as np
    import micromotion as mm

    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(size=(4000, 3)), axis=0)

    lp, fs_lp = mm.band_limited_qom(x, 100.0, *mm.OPTICAL_LEGACY_BAND)
    assert lp.size and np.isfinite(lp).all()
    assert fs_lp == 100.0

    # and it must differ from the band-pass, since it keeps the slow drift the band removes
    bp, _ = mm.band_limited_qom(x, 100.0, lo=0.2, hi=10.0)
    assert np.median(lp) > np.median(bp)

    # The upper edge is still validated when there is no lower edge -- but note that an
    # over-high `hi` is *clipped* to 0.9 x Nyquist rather than rejected, which is pre-existing
    # behaviour and the reason this asserts on a non-positive edge instead.
    import pytest
    with pytest.raises(ValueError):
        mm.band_limited_qom(x, 100.0, lo=None, hi=0.0)

    # clipping, stated so the asymmetry above is not mistaken for an oversight
    clipped, _ = mm.band_limited_qom(x, 100.0, lo=None, hi=90.0)
    at_nyquist, _ = mm.band_limited_qom(x, 100.0, lo=None, hi=45.0)
    assert np.allclose(clipped, at_nyquist)
