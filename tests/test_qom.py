"""Quantity of motion against signals whose answer is known in closed form.

For a single axis moving as x(t) = A sin(2*pi*f*t), the speed is |A*2*pi*f*cos(2*pi*f*t)|
and its mean over whole cycles is A*2*pi*f*(2/pi). Both the acceleration route and the
position route must recover that number, which is what makes them comparable at all.
"""

import numpy as np
import pytest

import micromotion as mm


def sine(amp_mm, f, fs, dur_s):
    t = np.arange(0, dur_s, 1 / fs)
    x = amp_mm * np.sin(2 * np.pi * f * t)
    a = -amp_mm * (2 * np.pi * f) ** 2 * np.sin(2 * np.pi * f * t) / 1000.0  # m/s^2
    return t, x, a


def expected_mean_speed(amp_mm, f):
    return amp_mm * 2 * np.pi * f * 2 / np.pi


@pytest.mark.parametrize("f", [0.5, 1.0, 2.0, 5.0])
def test_position_route_recovers_analytic_speed(f):
    fs, amp = 200.0, 3.0
    _, x, _ = sine(amp, f, fs, 120)
    r = mm.qom(x, fs, kind="position", unit="mm")
    trim = r.edge_samples
    got = np.mean(r.speed[trim:-trim])
    assert got == pytest.approx(expected_mean_speed(amp, f), rel=0.02)


@pytest.mark.parametrize("f", [0.5, 1.0, 2.0, 5.0])
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
    fs = 100.0
    _, x, _ = sine(3.0, 1.0, fs, 62.5)
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
