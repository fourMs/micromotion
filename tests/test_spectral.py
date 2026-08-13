"""Spectral helpers and the extra dynamics measures."""

import numpy as np
from scipy import signal
import pytest

from micromotion import dynamics as dy
from micromotion import spectral as sp


def tone(f, fs, dur, amp=1.0, noise=0.0, seed=0):
    t = np.arange(0, dur, 1 / fs)
    x = amp * np.sin(2 * np.pi * f * t)
    if noise:
        x = x + np.random.default_rng(seed).normal(0, noise, len(t))
    return x


def test_spectral_peak_finds_a_tone_and_reports_high_snr():
    r = sp.spectral_peak(tone(1.2, 100.0, 300, noise=0.3), 100.0, (0.7, 2.2))
    assert r["freq"] == pytest.approx(1.2, abs=0.03)
    assert r["snr"] > 10


def pink(fs, dur, seed=0):
    """A 1/f series: falling spectrum, no peak anywhere in it."""
    n = int(fs * dur)
    w = np.random.default_rng(seed).normal(0, 1, n)
    F = np.fft.rfft(w)
    f = np.fft.rfftfreq(n, 1 / fs)
    F[1:] /= f[1:]
    F[0] = 0
    return np.fft.irfft(F, n=n)


BAND = (0.12, 0.40)     # a respiration band, drawn above the knee of a 1/f spectrum


def test_spectral_peak_refuses_a_slope_instead_of_returning_its_lowest_bin():
    """The failure this function was reporting as a measurement.

    A band drawn above the knee of a 1/f spectrum contains no peak, and a bare argmax returns the
    band's lowest bin. That value is not a rhythm, it is the slope, and it is where four analyses
    in the Oslo corpus were reading breathing rates and sway frequencies from -- one of them on
    662 of 930 values, one of them into a book.
    """
    for seed in range(6):
        r = sp.spectral_peak(pink(10.0, 600, seed), 10.0, BAND)
        assert np.isnan(r["freq"]), f"seed {seed} found a peak in a slope"
        assert r["is_peak"] is False


def test_the_snr_does_not_catch_a_slope_which_is_why_the_excess_exists():
    """Tightening an SNR threshold selects the artefact rather than excluding it.

    The band median assumes a flat band. Over 1/f it is dragged down by the high-frequency end,
    so the lowest bins clear any SNR bar without being peaks.
    """
    old = sp.spectral_peak(pink(10.0, 600), 10.0, BAND, require_peak=False)
    assert not np.isnan(old["freq"])
    assert old["snr"] > 3               # would pass the advice the old docstring gave
    assert old["freq"] < BAND[0] + 0.04  # and it is the bottom of the band


def test_rejecting_the_edge_bin_would_not_have_been_enough():
    """The reason the test is "is this a peak" and not "where is it".

    On a falling spectrum, refusing the first bin moves the maximum to the second.
    """
    x = pink(10.0, 600)
    f, p = signal.welch(signal.detrend(x), 10.0, nperseg=int(10.0 * 60))
    pb = p[(f >= BAND[0]) & (f <= BAND[1])]
    assert int(np.argmax(pb)) <= 1                      # at or beside the floor
    assert int(np.argmax(pb[1:])) <= 1                  # and one bin along, the same shape
    assert np.isnan(sp.spectral_peak(x, 10.0, BAND)["freq"])


def test_require_peak_false_restores_the_old_behaviour():
    x = pink(10.0, 600)
    assert np.isnan(sp.spectral_peak(x, 10.0, BAND)["freq"])
    assert not np.isnan(sp.spectral_peak(x, 10.0, BAND, require_peak=False)["freq"])


def test_a_real_rhythm_riding_on_a_slope_still_survives_the_rule():
    """The rule must not throw away the thing it exists to protect.

    A breath is a modest peak on top of a much larger postural slope, which is exactly the shape
    that makes this hard: the raw argmax is still down at the band floor, and the peak is only
    visible once the slope is divided out.
    """
    fs, dur = 10.0, 600
    t = np.arange(0, dur, 1 / fs)
    x = pink(fs, dur, seed=1) + 0.7 * np.sin(2 * np.pi * 0.25 * t)
    raw = sp.spectral_peak(x, fs, BAND, require_peak=False)
    assert raw["freq"] < BAND[0] + 0.02           # the old answer returns the band floor
    r = sp.spectral_peak(x, fs, BAND)
    assert r["is_peak"] is True
    assert r["freq"] == pytest.approx(0.25, abs=0.02)  # the new one finds the rhythm
    assert r["excess"] > 2


def test_the_rule_is_conservative_and_misses_a_rhythm_below_the_slope():
    """What it costs, stated rather than discovered later.

    Dividing out the slope makes a peak findable, not visible from nothing. Below about half the
    amplitude of the case above, a genuine 0.25 Hz rhythm is present and this returns NaN. That is
    the intended direction to fail in -- a missing value rather than a wrong one -- but it is a
    false negative and a rate computed from these will be missing its weakest cases.
    """
    fs, dur = 10.0, 600
    t = np.arange(0, dur, 1 / fs)
    for amp in (0.2, 0.3, 0.4):
        x = pink(fs, dur, seed=1) + amp * np.sin(2 * np.pi * 0.25 * t)
        assert np.isnan(sp.spectral_peak(x, fs, BAND)["freq"])


def test_respiratory_peak_recovers_a_breath_riding_on_much_larger_slow_drift():
    """The case the periodogram version failed.

    It took the global maximum of the in-band spectrum, which on a red background lands near the
    band floor. Measured on Stillness2025's sixteen belts it returned a median 7.5 breaths per
    minute against the belts' own 16.8, and ranked participants at Spearman -0.32 against their own
    breath timing. It is now measured in the time domain, from detect_breaths.
    """
    fs, dur, breath_hz = 25.6, 300.0, 0.25          # 15 breaths per minute
    t = np.arange(0, dur, 1 / fs)
    rng = np.random.default_rng(0)
    drift = 10.0 * np.cumsum(rng.normal(0, 1, t.size)) / np.sqrt(t.size)
    got = sp.respiratory_peak(np.sin(2 * np.pi * breath_hz * t) + drift, fs)
    assert got * 60 == pytest.approx(breath_hz * 60, abs=1.5)


def test_respiratory_peak_agrees_with_detect_breaths():
    fs = 25.6
    t = np.arange(0, 240, 1 / fs)
    x = np.sin(2 * np.pi * 0.3 * t) + 0.05 * np.sin(2 * np.pi * 1.1 * t)
    assert sp.respiratory_peak(x, fs) * 60 == pytest.approx(
        sp.detect_breaths(x, fs)["rate_per_min"], rel=1e-9)


def test_spectral_peak_reports_low_snr_when_there_is_no_rhythm():
    x = np.random.default_rng(0).normal(size=30000)
    assert sp.spectral_peak(x, 100.0, (0.7, 2.2))["snr"] < 5


def test_band_rms_recovers_a_known_amplitude():
    """A sine of amplitude A has RMS A/sqrt(2)."""
    got = sp.band_rms(tone(2.0, 100.0, 200, amp=3.0), 100.0, (0.3, 10.0))
    assert got == pytest.approx(3.0 / np.sqrt(2), rel=0.03)


def test_band_rms_ignores_out_of_band_energy():
    fs = 100.0
    inside = tone(2.0, fs, 200, amp=3.0)
    outside = tone(30.0, fs, 200, amp=10.0)
    assert sp.band_rms(inside + outside, fs, (0.3, 10.0)) == pytest.approx(
        sp.band_rms(inside, fs, (0.3, 10.0)), rel=0.05
    )


def test_band_power_fractions_sum_to_about_one_and_split_correctly():
    fs = 100.0
    x = tone(0.25, fs, 400, amp=1.0) + tone(1.2, fs, 400, amp=1.0)
    fr = sp.band_power_fraction(x, fs, {"resp": (0.1, 0.5), "card": (0.7, 2.2)})
    assert fr["resp"] == pytest.approx(0.5, abs=0.1)
    assert fr["card"] == pytest.approx(0.5, abs=0.1)


def test_mean_frequency_sits_between_two_tones():
    fs = 100.0
    x = tone(1.0, fs, 300) + tone(3.0, fs, 300)
    assert 1.2 < sp.mean_frequency(x, fs, (0.3, 5.0)) < 2.8


def test_mean_frequency_moves_when_energy_shifts_without_the_peak_moving():
    fs = 100.0
    a = tone(1.0, fs, 300, amp=1.0) + tone(4.0, fs, 300, amp=0.2)
    b = tone(1.0, fs, 300, amp=1.0) + tone(4.0, fs, 300, amp=0.8)
    assert sp.mean_frequency(b, fs, (0.3, 6.0)) > sp.mean_frequency(a, fs, (0.3, 6.0))


def test_detect_breaths_counts_a_known_number_of_cycles():
    fs, dur, f = 25.0, 300.0, 0.25          # 15 breaths per minute
    r = sp.detect_breaths(tone(f, fs, dur, amp=5.0, noise=0.2), fs)
    assert r["n_breaths"] == pytest.approx(dur * f, abs=2)
    assert r["rate_per_min"] == pytest.approx(15.0, abs=1.0)


def test_detect_breaths_returns_peaks_and_troughs_in_alternation():
    r = sp.detect_breaths(tone(0.25, 25.0, 200, amp=5.0), 25.0)
    assert abs(len(r["peaks_s"]) - len(r["troughs_s"])) <= 1


# ------------------------------------------------------------------ extra dynamics

def test_apen_is_low_for_a_sine_and_high_for_noise():
    t = np.arange(3000) / 50.0
    assert dy.apen(np.sin(2 * np.pi * 0.5 * t)) < dy.apen(
        np.random.default_rng(0).normal(size=3000)
    )


def test_apen_and_sampen_agree_in_direction():
    rng = np.random.default_rng(0)
    t = np.arange(3000) / 50.0
    regular, noisy = np.sin(2 * np.pi * 0.5 * t), rng.normal(size=3000)
    assert (dy.apen(regular) < dy.apen(noisy)) == (dy.sampen(regular) < dy.sampen(noisy))


def test_dcca_is_near_one_for_identical_series():
    x = np.cumsum(np.random.default_rng(0).normal(size=4000))
    rho = dy.dcca(x, x)["rho"]
    assert np.nanmedian(rho) == pytest.approx(1.0, abs=0.02)


def test_dcca_is_near_zero_for_independent_walks():
    rng = np.random.default_rng(0)
    a, b = np.cumsum(rng.normal(size=6000)), np.cumsum(rng.normal(size=6000))
    assert abs(np.nanmedian(dy.dcca(a, b)["rho"])) < 0.6


def test_dcca_detects_a_shared_component_that_plain_correlation_misses():
    """Two wandering series with a common slow driver."""
    rng = np.random.default_rng(3)
    common = np.cumsum(rng.normal(size=6000))
    a = common + np.cumsum(rng.normal(size=6000)) * 3
    b = common + np.cumsum(rng.normal(size=6000)) * 3
    rho = dy.dcca(a, b)["rho"]
    assert np.nanmax(rho) > 0.3


def test_peak_from_spectrum_is_the_same_rule_as_spectral_peak():
    """The primitive and the convenience wrapper must not drift apart."""
    fs, dur = 10.0, 600
    t = np.arange(0, dur, 1 / fs)
    for x in (pink(fs, dur, 3), pink(fs, dur, 1) + 0.7 * np.sin(2 * np.pi * 0.25 * t)):
        nper = int(min(len(x), fs * 60.0))
        f, p = signal.welch(signal.detrend(x), fs, nperseg=nper)
        a = sp.spectral_peak(x, fs, BAND)
        b = sp.peak_from_spectrum(f, p, BAND)
        assert a["is_peak"] == b["is_peak"]
        assert (np.isnan(a["freq"]) and np.isnan(b["freq"])) or a["freq"] == b["freq"]


def two_tone(fs=50.0, dur=600.0):
    """0.3 Hz at amplitude 1 and 1.2 Hz at amplitude 2: powers 0.5 and 2.0."""
    t = np.arange(0, dur, 1 / fs)
    return np.sin(2 * np.pi * 0.3 * t) + 2.0 * np.sin(2 * np.pi * 1.2 * t)


def test_band_share_recovers_an_analytic_two_tone_share():
    """A share the arithmetic can be checked against by hand.

    The 1.2 Hz tone carries power 2.0 and the 0.3 Hz tone 0.5, so the cardiac band's share of a
    0.1-3.0 Hz denominator is 2.0 / 2.5 = 0.8 exactly.
    """
    s = sp.band_share(two_tone(), 50.0, num_band=(0.7, 2.2), den_band=(0.1, 3.0))
    assert s == pytest.approx(0.8, abs=0.01)


def test_band_share_and_from_spectrum_agree():
    """The primitive and the convenience wrapper must not drift apart."""
    x, fs = two_tone(), 50.0
    f, p = signal.welch(signal.detrend(x), fs, nperseg=int(fs * 60.0))
    a = sp.band_share(x, fs, num_band=(0.7, 2.2), den_band=(0.1, 3.0))
    b = sp.band_share_from_spectrum(f, p, num_band=(0.7, 2.2), den_band=(0.1, 3.0))
    assert a == pytest.approx(b, rel=1e-12)


def test_band_share_bands_cannot_be_passed_silently():
    """The bands are keyword-only, so a call site has to name both.

    That is the point of the function: four shares -- 38, 43, 45 and 58 per cent -- were quoted
    against each other in this corpus while resting on different, sometimes unstated,
    denominators. A share whose bands are not named is not a reportable number.
    """
    with pytest.raises(TypeError):
        sp.band_share(two_tone(), 50.0, (0.7, 2.2), (0.1, 3.0))
    with pytest.raises(TypeError):
        sp.band_share(two_tone(), 50.0, num_band=(0.7, 2.2))          # denominator missing


def test_a_numerator_reaching_outside_the_denominator_raises():
    """The trap the legacy 58 per cent sat next to: a 'share' that is really a ratio.

    Power the denominator does not contain can push the quotient past 1, and a disjoint pair
    measures two different things divided by each other. All three shapes raise.
    """
    x = two_tone()
    for num in [(0.05, 2.2),        # reaches below the denominator
                (0.7, 4.0),         # reaches above it
                (4.0, 6.0)]:        # disjoint from it
        with pytest.raises(ValueError, match="outside the denominator"):
            sp.band_share(x, 50.0, num_band=num, den_band=(0.1, 3.0))
    # and the whole denominator over itself is exactly 1
    s = sp.band_share(x, 50.0, num_band=(0.1, 3.0), den_band=(0.1, 3.0))
    assert s == pytest.approx(1.0, rel=1e-12)


def test_band_share_warns_when_the_rate_cannot_deliver_the_denominator():
    """A denominator edge above Nyquist silently narrows the share's reference.

    At 10 Hz sampling a 0.1-8 Hz denominator is really 0.1-5, so the share is over a band the
    caller did not name. The same call at 50 Hz is silent, which is the check that the warning
    watches the rate and not the call.
    """
    import warnings as _w
    x10 = two_tone(fs=10.0)
    with pytest.warns(RuntimeWarning, match="truncated denominator"):
        sp.band_share(x10, 10.0, num_band=(0.7, 2.2), den_band=(0.1, 8.0))
    with _w.catch_warnings():
        _w.simplefilter("error")
        sp.band_share(two_tone(), 50.0, num_band=(0.7, 2.2), den_band=(0.1, 8.0))


def test_band_share_from_spectrum_warns_when_the_spectrum_stops_short():
    x, fs = two_tone(fs=10.0), 10.0
    f, p = signal.welch(signal.detrend(x), fs, nperseg=int(fs * 60.0))
    with pytest.warns(RuntimeWarning, match="spectrum ends"):
        sp.band_share_from_spectrum(f, p, num_band=(0.7, 2.2), den_band=(0.1, 8.0))


def test_band_share_warns_on_non_finite_input_like_the_filters_do():
    """One NaN makes the whole Welch spectrum NaN, so the share is NaN, not mostly right."""
    import warnings as _w
    x = two_tone()
    x[100] = np.nan
    with pytest.warns(RuntimeWarning, match="non-finite"):
        s = sp.band_share(x, 50.0, num_band=(0.7, 2.2), den_band=(0.1, 3.0))
    assert np.isnan(s)
    with _w.catch_warnings():
        _w.simplefilter("error")
        sp.band_share(two_tone(), 50.0, num_band=(0.7, 2.2), den_band=(0.1, 3.0))


# --- the two spectral-share conventions -------------------------------------------------
#
# This package contained both of them before it said so: spectral.band_share and friends
# integrate with the trapezoid rule over a closed [lo, hi], while physio.spectral_band_fractions
# sums bins over a half-open [lo, hi). Four analysis scripts were carried from the second onto the
# first so that one corpus would hold a single convention, and the move republished a share from
# 58 to 59 per cent, another from 16.6 to 15.1, a fold from 3.1 to 3.2, and 18 of the 24 numbers
# in one table. These tests hold the two conventions apart and pin what each parameter selects.


def exact_grid(n=400, df=1 / 20.0):
    """A frequency grid whose round edges are exactly representable.

    ``np.arange(n) / 20`` puts 0.4, 0.7, 2.2, 3.0 and 5.0 Hz exactly on a bin, which is the
    situation a 60 s Welch window produces on real data and the situation in which closure
    costs a whole bin at each edge. Division rather than multiplication by ``df``, because
    ``14 * 0.05`` is not the double nearest 0.7 and ``14 / 20`` is.
    """
    return np.arange(n) / (1.0 / df)


def red_spectrum():
    """A falling spectrum with a respiratory and a cardiac bump, on the exact grid.

    Shaped like a chest accelerometer at standstill: a 1/f^1.6 background, a breathing bump at
    0.25 Hz and a ballistocardiac one at 1.1 Hz.
    """
    f = exact_grid()
    p = np.zeros_like(f)
    p[1:] = 1.0 / f[1:] ** 1.6
    p += 3.0 * np.exp(-0.5 * ((f - 0.25) / 0.06) ** 2)
    p += 1.5 * np.exp(-0.5 * ((f - 1.1) / 0.15) ** 2)
    p[0] = p[1]
    return f, p


def test_the_round_band_edges_land_exactly_on_bins():
    """Why closure bites at all: the edges analysts choose are bins, not gaps between them.

    At a 60 s window the spacing is 1/60 Hz and 0.40, 0.70, 2.20, 3.0, 5.0 and 8.0 Hz are all
    integer multiples of it, so [lo, hi] holds one more bin than [lo, hi) at both the
    numerator's edge and the denominator's.
    """
    f = np.fft.rfftfreq(int(50.0 * 60.0), 1 / 50.0)
    for edge in (0.4, 0.7, 2.2, 3.0, 5.0, 8.0):
        assert (f == edge).any(), f"{edge} Hz is not on the 60 s Welch grid"


def test_each_convention_parameter_selects_the_arithmetic_it_names():
    """All four combinations, against arithmetic done by hand on a flat spectrum.

    Flat power, a 0.4-0.7 Hz numerator and a 0.1-5.0 Hz denominator on a 0.05 Hz grid. Closed
    holds 7 numerator bins and 99 denominator bins, half-open one fewer of each; the trapezoid
    reduces to the band widths, 0.3 over 4.9 and 0.25 over 4.85.
    """
    f = exact_grid()
    p = np.ones_like(f)
    num, den = (0.4, 0.7), (0.1, 5.0)
    got = {(r, c): sp.band_share_from_spectrum(f, p, num_band=num, den_band=den,
                                               integrate=r, interval=c)
           for r in ("trapezoid", "sum") for c in ("closed", "half_open")}
    assert got[("sum", "closed")] == pytest.approx(7 / 99, rel=1e-12)
    assert got[("sum", "half_open")] == pytest.approx(6 / 98, rel=1e-12)
    assert got[("trapezoid", "closed")] == pytest.approx(0.3 / 4.9, rel=1e-12)
    assert got[("trapezoid", "half_open")] == pytest.approx(0.25 / 4.85, rel=1e-12)

    # On a FLAT band the trapezoid's half-weighted end bins remove exactly one bin's worth, so
    # two of the four coincide. That identity is why the conventions look interchangeable.
    assert got[("trapezoid", "closed")] == pytest.approx(got[("sum", "half_open")], rel=1e-12)


def test_the_quadrature_rule_alone_moves_a_share():
    """Same mask, different rule: 0.230 against 0.207, a tenth of the cardiac share.

    The claim this replaces -- that the two "yield nearly identical results on the uniform
    frequency spacing of Welch" -- was in a docstring in this package. On chest-accelerometer
    standstill data the rule alone moves a respiratory share by up to 0.034 absolute on a share
    of about 0.16.
    """
    f, p = red_spectrum()
    kw = dict(num_band=(0.7, 2.2), den_band=(0.1, 3.0), interval="closed")
    trap = sp.band_share_from_spectrum(f, p, integrate="trapezoid", **kw)
    summed = sp.band_share_from_spectrum(f, p, integrate="sum", **kw)
    assert trap == pytest.approx(0.2298, abs=5e-4)
    assert summed == pytest.approx(0.2073, abs=5e-4)
    assert abs(trap - summed) > 0.02


def test_interval_closure_alone_moves_a_share():
    """Same rule, different closure: 0.685 against 0.658 on a respiratory band.

    Both band edges sit on bins, so closing the interval adds a whole bin to the numerator and
    a whole bin to the denominator. They do not cancel, because the added bins carry very
    different power on a falling spectrum.
    """
    f, p = red_spectrum()
    kw = dict(num_band=(0.1, 0.4), den_band=(0.1, 3.0), integrate="sum")
    closed = sp.band_share_from_spectrum(f, p, interval="closed", **kw)
    half = sp.band_share_from_spectrum(f, p, interval="half_open", **kw)
    assert closed == pytest.approx(0.6854, abs=5e-4)
    assert half == pytest.approx(0.6579, abs=5e-4)
    assert abs(closed - half) > 0.02


def test_the_defaults_are_still_trapezoid_and_closed():
    """The released convention, pinned. Changing it moves every share this package has quoted."""
    f, p = red_spectrum()
    kw = dict(num_band=(0.7, 2.2), den_band=(0.1, 3.0))
    default = sp.band_share_from_spectrum(f, p, **kw)
    assert default == sp.band_share_from_spectrum(f, p, integrate="trapezoid",
                                                  interval="closed", **kw)
    assert default != sp.band_share_from_spectrum(f, p, integrate="sum",
                                                  interval="half_open", **kw)

    x, fs = two_tone(), 50.0
    assert sp.band_share(x, fs, **kw) == sp.band_share(x, fs, integrate="trapezoid",
                                                       interval="closed", **kw)


def test_band_share_forwards_the_convention_to_the_spectrum_it_computes():
    """The Welch front end must not quietly compute a different convention from the primitive."""
    x, fs = two_tone(), 50.0
    f, p = signal.welch(signal.detrend(x), fs, nperseg=int(fs * 60.0))
    for rule in ("trapezoid", "sum"):
        for closure in ("closed", "half_open"):
            a = sp.band_share(x, fs, num_band=(0.7, 2.2), den_band=(0.1, 3.0),
                              integrate=rule, interval=closure)
            b = sp.band_share_from_spectrum(f, p, num_band=(0.7, 2.2), den_band=(0.1, 3.0),
                                            integrate=rule, interval=closure)
            assert a == pytest.approx(b, rel=1e-12), f"{rule}/{closure} disagree"


def test_the_bin_sum_convention_reproduces_spectral_band_fractions_exactly():
    """The older convention, still reachable: the same number from both functions.

    ``spectral_band_fractions`` sums bins over [lo, hi) at numerator and denominator both, so
    ``band_share(..., integrate="sum", interval="half_open")`` on its own spectrum must give the
    identical value. Compared on one spectrum rather than one series because the two functions
    detrend differently -- linear against mean removal -- and that difference is not the share
    rule.
    """
    from scipy.signal import welch

    from micromotion import physio

    # A red series rather than two clean tones: where a spectrum is empty between the bands,
    # every convention returns the same number and the comparison proves nothing.
    fs = 50.0
    x = pink(fs, 600.0) + 0.02 * two_tone(fs, 600.0)
    nperseg = min(len(x), max(8, int(fs * 20)))
    f, p = welch(x - x.mean(), fs, nperseg=nperseg)

    num, den = (0.7, 2.2), (0.1, 3.0)
    theirs = physio.spectral_band_fractions(x, fs, {"cardiac": num}, total_band=den)["cardiac"]
    ours = sp.band_share_from_spectrum(f, p, num_band=num, den_band=den,
                                       integrate="sum", interval="half_open")
    assert ours == pytest.approx(theirs, rel=1e-12)

    # and the default convention does NOT reproduce it, which is the whole point
    default = sp.band_share_from_spectrum(f, p, num_band=num, den_band=den)
    assert abs(default - theirs) > 0.005, (
        f"the two conventions agreed to {abs(default - theirs):.4g} on this spectrum; "
        "pick one with more power near the band edges or the test watches nothing")


def test_an_unknown_rule_or_closure_raises_rather_than_falling_back():
    """A misspelled convention must not silently compute the default one."""
    f, p = red_spectrum()
    kw = dict(num_band=(0.7, 2.2), den_band=(0.1, 3.0))
    for bad in ("trapz", "rectangle", "", None):
        with pytest.raises(ValueError, match="integration rule"):
            sp.band_share_from_spectrum(f, p, integrate=bad, **kw)
        with pytest.raises(ValueError, match="integration rule"):
            sp.band_share(two_tone(), 50.0, integrate=bad, **kw)
    for bad in ("half-open", "open", "left", None):
        with pytest.raises(ValueError, match="unknown interval"):
            sp.band_share_from_spectrum(f, p, interval=bad, **kw)
        with pytest.raises(ValueError, match="unknown interval"):
            sp.band_share(two_tone(), 50.0, interval=bad, **kw)


def test_the_guards_still_fire_under_the_other_convention():
    """The bands, the containment rule and both warnings are convention-independent."""
    import warnings as _w
    other = dict(integrate="sum", interval="half_open")

    with pytest.raises(TypeError):
        sp.band_share(two_tone(), 50.0, (0.7, 2.2), (0.1, 3.0), **other)
    with pytest.raises(ValueError, match="outside the denominator"):
        sp.band_share(two_tone(), 50.0, num_band=(0.05, 2.2), den_band=(0.1, 3.0), **other)
    with pytest.warns(RuntimeWarning, match="truncated denominator"):
        sp.band_share(two_tone(fs=10.0), 10.0, num_band=(0.7, 2.2), den_band=(0.1, 8.0),
                      **other)
    x = two_tone()
    x[100] = np.nan
    with pytest.warns(RuntimeWarning, match="non-finite"):
        assert np.isnan(sp.band_share(x, 50.0, num_band=(0.7, 2.2), den_band=(0.1, 3.0),
                                      **other))
    with _w.catch_warnings():
        _w.simplefilter("error")
        s = sp.band_share(two_tone(), 50.0, num_band=(0.7, 2.2), den_band=(0.1, 3.0), **other)
    assert 0.0 < s < 1.0
