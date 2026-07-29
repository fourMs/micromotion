"""Scaling and nonlinearity measures for postural time series.

These describe how movement is structured in time rather than how much of it there is. They
are the part of this package with reuse value beyond the standstill corpus, and they are
also the part where a subtly wrong implementation produces a plausible number rather than an
error. Every function here is covered by a test against a process whose answer is known in
advance; see ``tests/test_dynamics.py``.

One warning carried over from the corpus. These methods read across scales, so they are the
ones that upsampling corrupts. Resample downwards only, and use
:func:`micromotion.resample.to_rate`, which refuses to do otherwise.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as _signal
from scipy.spatial import cKDTree


# --------------------------------------------------------------------------- surrogates

def iaaft(x, iters: int = 100, rng=None) -> np.ndarray:
    """Iterative amplitude-adjusted Fourier transform surrogate.

    Preserves both the amplitude distribution and the power spectrum while destroying any
    nonlinear structure, so it is the null hypothesis "this series is a linear Gaussian
    process observed through a monotonic transform".
    """
    rng = rng or np.random.default_rng()
    x = np.asarray(x, float)
    n = len(x)
    amp = np.abs(np.fft.rfft(x))
    srt = np.sort(x)
    y = rng.permutation(x)
    for _ in range(iters):
        Y = np.fft.rfft(y)
        y = np.fft.irfft(amp * np.exp(1j * np.angle(Y)), n=n)
        y = srt[np.argsort(np.argsort(y))]
    return y


def phase_surrogate(x, rng=None) -> np.ndarray:
    """Phase-randomised surrogate: preserves the spectrum, not the distribution.

    Cheaper than :func:`iaaft` and a weaker null, since a non-Gaussian amplitude
    distribution alone can make a linear series look nonlinear against it.
    """
    rng = rng or np.random.default_rng()
    x = np.asarray(x, float)
    X = np.fft.rfft(x)
    ph = rng.uniform(0, 2 * np.pi, len(X))
    ph[0] = 0
    if len(x) % 2 == 0:
        ph[-1] = 0
    return np.fft.irfft(np.abs(X) * np.exp(1j * ph), n=len(x))


def circular_shift_surrogate(x, rng=None, margin: float = 0.1) -> np.ndarray:
    """Rotate a series in time.

    Preserves everything about the series itself and destroys only its alignment with
    another, so it is the right null for a correlation between two recordings and the wrong
    one for a property of a single series.
    """
    rng = rng or np.random.default_rng()
    x = np.asarray(x)
    n = len(x)
    k = rng.integers(int(margin * n), int((1 - margin) * n))
    return np.roll(x, k)


def surrogate_test(x, statistic, n: int = 99, method=iaaft, rng=None) -> dict:
    """Compare a statistic against its surrogate distribution.

    Returns the observed value, the surrogate mean and standard deviation, a z score and a
    two-sided p value. The p value uses the standard ``(count + 1) / (n + 1)`` form, which
    cannot return zero.
    """
    rng = rng or np.random.default_rng()
    obs = float(statistic(np.asarray(x, float)))
    null = np.array([statistic(method(x, rng=rng)) for _ in range(n)])
    sd = null.std()
    z = (obs - null.mean()) / (sd + 1e-30)
    p = (np.sum(np.abs(null - null.mean()) >= abs(obs - null.mean())) + 1) / (n + 1)
    return {"observed": obs, "null_mean": float(null.mean()), "null_sd": float(sd),
            "z": float(z), "p": float(p), "n_surrogates": n}


# --------------------------------------------------------------------------- statistics

def trev(x, tau: int = 1) -> float:
    """Time-reversal asymmetry.

    A linear Gaussian process looks the same run backwards; most nonlinear ones do not. The
    statistic is the skew of the lag-``tau`` increments, normalised by their variance.
    """
    x = np.asarray(x, float)
    d = x[tau:] - x[:-tau]
    return float(np.mean(d**3) / (np.mean(d**2) ** 1.5 + 1e-30))


def dfa(x, smin: int | None = None, smax: int | None = None, nsc: int = 18,
        order: int = 1) -> dict:
    """Detrended fluctuation analysis.

    Returns the scaling exponent ``alpha``. For reference, 0.5 is white noise, 1.0 is pink
    noise, and 1.5 is Brownian motion. Values above 0.5 mean the series persists: a
    deviation tends to be followed by more of the same.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    smin = smin or 8
    smax = smax or n // 4
    if n < 100 or smax <= smin:
        return {"alpha": float("nan"), "scales": np.array([]), "F": np.array([])}
    Y = np.cumsum(x - x.mean())
    scales = np.unique(np.round(np.logspace(np.log10(smin), np.log10(smax), nsc)).astype(int))
    F = []
    for s in scales:
        nseg = n // s
        if nseg < 4:
            F.append(np.nan)
            continue
        seg = Y[: nseg * s].reshape(nseg, s)
        t = np.arange(s)
        V = np.polynomial.polynomial.polyvander(t, order)
        coef, *_ = np.linalg.lstsq(V, seg.T, rcond=None)
        resid = seg.T - V @ coef
        F.append(np.sqrt(np.mean(resid**2)))
    F = np.array(F, float)
    m = np.isfinite(F) & (F > 0)
    if m.sum() < 5:
        return {"alpha": float("nan"), "scales": scales, "F": F}
    alpha = float(np.polyfit(np.log(scales[m]), np.log(F[m]), 1)[0])
    return {"alpha": alpha, "scales": scales, "F": F}


def mfdfa(x, qs=None, smin: int = 16, smax: int | None = None, nsc: int = 20,
          order: int = 1) -> dict | None:
    """Multifractal detrended fluctuation analysis.

    Returns ``h(q)``, the singularity spectrum, the generalised Hurst exponent ``h2`` and
    the spectrum ``width``. A width near zero means one scaling exponent describes the whole
    series; a wide spectrum means different parts of it scale differently.

    A width above about 2 for postural data is a signal to check the preprocessing rather
    than to celebrate: widths up to 6.6 were once produced entirely by upsampling.
    """
    qs = np.arange(-5, 5.1, 0.5) if qs is None else np.asarray(qs, float)
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    N = len(x)
    smax = smax or N // 8
    if smax <= smin:
        return None
    Y = np.cumsum(x - x.mean())
    scales = np.unique(np.round(np.logspace(np.log10(smin), np.log10(smax), nsc)).astype(int))
    Fq = np.full((len(qs), len(scales)), np.nan)
    for si, s in enumerate(scales):
        nseg = N // s
        if nseg < 4:
            continue
        seg = Y[: nseg * s].reshape(nseg, s)
        t = np.arange(s)
        V = np.polynomial.polynomial.polyvander(t, order)
        coef, *_ = np.linalg.lstsq(V, seg.T, rcond=None)
        F2 = np.mean((seg.T - V @ coef) ** 2, axis=0)
        F2 = F2[F2 > 0]
        if len(F2) < 4:
            continue
        for qi, q in enumerate(qs):
            if abs(q) < 1e-9:
                Fq[qi, si] = np.exp(0.5 * np.mean(np.log(F2)))
            else:
                Fq[qi, si] = np.mean(F2 ** (q / 2)) ** (1 / q)
    h = np.full(len(qs), np.nan)
    for qi in range(len(qs)):
        m = np.isfinite(Fq[qi])
        if m.sum() >= 5:
            h[qi] = np.polyfit(np.log(scales[m]), np.log(Fq[qi][m]), 1)[0]
    ok = np.isfinite(h)
    if ok.sum() < 5:
        return None
    q, hh = qs[ok], h[ok]
    tau = q * hh - 1
    alpha = np.gradient(tau, q)
    return {
        "qs": q, "h": hh, "alpha": alpha, "f": q * alpha - tau,
        "h2": float(np.interp(2, q, hh)),
        "width": float(np.nanmax(alpha) - np.nanmin(alpha)),
    }


def sda(x, y=None, fs: float = 25.0, maxlag: float = 10.0, nlags: int = 60) -> dict:
    """Stabilogram diffusion analysis, after Collins and De Luca.

    Postural sway behaves like two different processes at two timescales: over short
    intervals it drifts away from where it was, and over longer ones it is pulled back. The
    crossover between them is found by fitting two lines to the log-log mean-square
    displacement and taking their intersection.

    Returns short- and long-term Hurst exponents and diffusion coefficients, and the
    critical time and displacement. ``Hs`` above 0.5 with ``Hl`` below it is the normal
    pattern: open-loop drift, then closed-loop correction.
    """
    x = np.asarray(x, float)
    y = np.zeros_like(x) if y is None else np.asarray(y, float)
    n = len(x)
    L = min(int(maxlag * fs), n - 5)
    if L < 10:
        return {"Hs": float("nan"), "Hl": float("nan")}
    lags = np.unique(np.round(np.logspace(0, np.log10(L), nlags)).astype(int))
    msd = np.array([np.nanmean((x[k:] - x[:-k]) ** 2 + (y[k:] - y[:-k]) ** 2) for k in lags])
    ok = np.isfinite(msd) & (msd > 0)
    out = fit_two_region(lags[ok] / fs, msd[ok])
    out["lags_s"], out["msd"] = lags[ok] / fs, msd[ok]
    return out


def fit_two_region(dt, msd) -> dict:
    """Fit two straight lines to a log-log curve and locate their crossover.

    Generic: any measure with a scaling break can use it.
    """
    ld, lm = np.log10(dt), np.log10(msd)
    best = None
    for i in range(4, len(dt) - 4):
        s1 = np.polyfit(ld[: i + 1], lm[: i + 1], 1)
        s2 = np.polyfit(ld[i:], lm[i:], 1)
        r = (np.sum((lm[: i + 1] - np.polyval(s1, ld[: i + 1])) ** 2)
             + np.sum((lm[i:] - np.polyval(s2, ld[i:])) ** 2))
        if best is None or r < best[0]:
            best = (r, i, s1, s2)
    if best is None:
        return {"Hs": float("nan"), "Hl": float("nan")}
    _, i, s1, s2 = best
    xc = (s2[1] - s1[1]) / (s1[0] - s2[0])
    return {"Hs": s1[0] / 2, "Hl": s2[0] / 2, "Ds": 10 ** s1[1] / 2, "Dl": 10 ** s2[1] / 2,
            "dtc": 10**xc, "msdc": 10 ** np.polyval(s1, xc), "idx": int(i)}


def sampen(x, m: int = 2, r: float | None = None) -> float:
    """Sample entropy: how unpredictable the series is.

    The negative log probability that two segments which match for ``m`` samples still match
    for ``m + 1``. Higher is less regular. ``r`` defaults to 0.2 standard deviations.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < m + 20:
        return float("nan")
    r = 0.2 * np.std(x) if r is None else r
    if r <= 0:
        return float("nan")

    def count(mm):
        emb = np.lib.stride_tricks.sliding_window_view(x, mm)[: n - m]
        tree = cKDTree(emb)
        return tree.count_neighbors(tree, r, p=np.inf) - len(emb)

    a, b = count(m + 1), count(m)
    return float(-np.log(a / b)) if a > 0 and b > 0 else float("nan")


# --------------------------------------------------------------------- embedding and RQA

def ami(x, maxlag: int = 100, bins: int = 32) -> np.ndarray:
    """Average mutual information against lag, for choosing an embedding delay."""
    x = np.asarray(x, float)
    out = []
    for k in range(1, maxlag + 1):
        c, _, _ = np.histogram2d(x[:-k], x[k:], bins=bins)
        p = c / c.sum()
        px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
        m = p > 0
        out.append(float(np.sum(p[m] * np.log(p[m] / (px @ py)[m]))))
    return np.array(out)


def first_ami_minimum(x, maxlag: int = 100, smooth: int = 9) -> int:
    """The conventional embedding delay: the first local minimum of mutual information.

    The curve is smoothed first. Estimated from a histogram it is noisy enough that the
    first local minimum of the raw curve is usually spurious: on a clean sine, whose true
    answer is a quarter period, the unsmoothed rule returns a delay of 2.
    """
    a = ami(x, maxlag)
    if smooth > 1 and len(a) > smooth:
        k = np.ones(smooth) / smooth
        a = np.convolve(a, k, mode="same")
        a[: smooth // 2] = a[smooth // 2]
        a[-(smooth // 2):] = a[-(smooth // 2) - 1]
    for i in range(1, len(a) - 1):
        if a[i] < a[i - 1] and a[i] < a[i + 1]:
            return i + 1
    return int(np.argmin(a)) + 1


def embed(x, dim: int, tau: int) -> np.ndarray:
    """Time-delay embedding. ``x`` may be one series or several columns."""
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[:, None]
    n = len(x) - (dim - 1) * tau
    if n <= 0:
        raise ValueError("series is too short for this embedding")
    return np.concatenate([x[i * tau: i * tau + n] for i in range(dim)], axis=1)


def rqa(x, dim: int = 3, tau: int | None = None, rr: float = 0.05,
        lmin: int = 2) -> dict:
    """Recurrence quantification, at fixed recurrence rate.

    The threshold is solved per plot so that exactly ``rr`` of pairs count as recurrent.
    That matters: with a fixed absolute threshold, determinism partly measures how tightly
    packed the trajectory is, so two recordings of different amplitude are not comparable.

    Pass several columns to get multidimensional recurrence quantification, which is the
    form used for between-body coupling.

    Leave ``tau`` unset unless you have a reason. A delay of 1 makes consecutive embedding
    vectors share all but one coordinate, so recurrences chain into diagonals that reflect
    the embedding rather than the dynamics: white noise embedded at ``tau=1`` reports a
    determinism of 0.57, against 0.08 at a properly chosen delay.
    """
    x = np.asarray(x, float)
    if x.ndim == 1:
        tau = tau or first_ami_minimum(x, min(100, len(x) // 10))
        emb = embed(x, dim, tau)
    else:
        emb = x
        tau = tau or 1
    d = np.linalg.norm(emb[:, None, :] - emb[None, :, :], axis=2)
    eps = np.quantile(d[np.triu_indices_from(d, 1)], rr)
    R = d <= eps

    n = len(R)
    lengths = []
    for k in range(-(n - lmin), n - lmin + 1):
        if k == 0:
            continue
        diag = np.diagonal(R, k)
        run = 0
        for v in diag:
            if v:
                run += 1
            else:
                if run >= lmin:
                    lengths.append(run)
                run = 0
        if run >= lmin:
            lengths.append(run)
    lengths = np.array(lengths)
    n_rec = R.sum() - n
    det = lengths.sum() / n_rec if n_rec and len(lengths) else 0.0
    p = np.bincount(lengths)[lmin:] if len(lengths) else np.array([])
    p = p[p > 0] / p.sum() if p.sum() else np.array([])
    return {
        "RR": float(n_rec / (n * n - n)),
        "DET": float(det),
        "Lmax": int(lengths.max()) if len(lengths) else 0,
        "Lmean": float(lengths.mean()) if len(lengths) else 0.0,
        "ENTR": float(-np.sum(p * np.log(p))) if len(p) else 0.0,
        "eps": float(eps), "dim": dim, "tau": int(tau),
    }


# --------------------------------------------------------------------------- phase

def plv(a, b, fs: float, band: tuple[float, float] | None = None) -> dict:
    """Phase-locking value between two signals.

    Returns the locking strength from 0 to 1 and the preferred phase difference in radians.
    Band-pass first if the signals are broadband; locking is only meaningful within a band
    where each signal has a well-defined phase.
    """
    from .filters import bandpass

    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if band:
        a, b = bandpass(a, fs, *band), bandpass(b, fs, *band)
    pa = np.angle(_signal.hilbert(a - a.mean()))
    pb = np.angle(_signal.hilbert(b - b.mean()))
    z = np.exp(1j * (pa - pb))
    return {"plv": float(np.abs(z.mean())), "preferred_phase": float(np.angle(z.mean()))}
