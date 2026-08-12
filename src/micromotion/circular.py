"""Statistics for angles. This module is the family's reference for them.

**Where circular statistics live across the four toolboxes.** micromotion owns
them: the axial tests sway needs, the circular-linear correlation, the V-test.
ambiscape keeps a small `circstats` for the time-series end --- `phase_stats`,
`relative_phase` --- and the few primitives those need, rather than taking a
dependency on this package for six short functions. MGT re-exports this module
along with the rest of micromotion.

That division decayed once and would have again. On 2026-08-12 the two
Rayleigh implementations were found to disagree on about a fifth of random
cases: this one uses Wilkie's approximation, ambiscape used Zar's earlier
series expansion, both are published and neither was wrong, and nothing said
they were meant to match. ambiscape now matches this module, and
`ambiscape/tests/test_circstats_agreement.py` asserts it on every run rather
than trusting it. One further difference is deliberate and remains:
`circ_corr` returns a dict here and a float there, so a caller who swaps the
import gets a different shape --- changing either is an API break, and the
test pins the arithmetic so at least the numbers cannot drift on top of it.


Sway direction, the phase of a breath, the time of day a session happened: all are circular,
and ordinary statistics quietly give wrong answers on them. The mean of 350 degrees and 10
degrees is 0, not 180.

Sway direction needs a further distinction. A body swaying forwards and backwards along one
line has no preferred direction, only a preferred axis — 10 degrees and 190 degrees are the
same posture. Statistics that treat those as opposite will report a person with a strong
front-back sway as having no directional preference at all. The axial variants here double
the angles before averaging, which is the standard fix, and they are the ones to use for
sway.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as _stats


def circ_mean(angles, weights=None) -> dict:
    """Mean direction and concentration.

    Returns the mean angle in radians and ``R``, the resultant length, which runs from 0 for
    angles spread uniformly around the circle to 1 for angles all identical. ``R`` is the
    circular answer to "how consistent is this", and there is no separate standard deviation
    worth reporting beside it.
    """
    a = np.asarray(angles, float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"mean": float("nan"), "R": float("nan"), "n": 0}
    w = np.ones_like(a) if weights is None else np.asarray(weights, float)[: len(a)]
    z = np.sum(w * np.exp(1j * a)) / np.sum(w)
    return {"mean": float(np.angle(z)), "R": float(np.abs(z)), "n": len(a)}


def rayleigh(angles) -> dict:
    """Test whether angles are spread uniformly around the circle.

    The null hypothesis is no preferred direction. A small p rejects it. Uses the standard
    approximation, which is accurate for n above about 10.

    Note what this cannot detect: a perfectly bidirectional distribution, with half the
    angles at 0 and half at 180, has a resultant length of zero and passes as uniform. Use
    :func:`rayleigh_axial` when the data are axial, which sway is.
    """
    a = np.asarray(angles, float)
    a = a[np.isfinite(a)]
    n = len(a)
    if n < 3:
        return {"R": float("nan"), "z": float("nan"), "p": float("nan"), "n": n}
    R = np.abs(np.sum(np.exp(1j * a))) / n
    z = n * R**2
    p = np.exp(np.sqrt(1 + 4 * n + 4 * (n**2 - (n * R) ** 2)) - (1 + 2 * n))
    return {"R": float(R), "z": float(z), "p": float(min(1.0, p)), "n": n}


def rayleigh_axial(angles) -> dict:
    """Rayleigh test for axial data, where an angle and its opposite are the same.

    Doubles the angles, tests those, and halves the resulting mean direction back. This is
    the correct test for sway direction, marker orientation, and anything else defined on a
    line rather than a ray.
    """
    a = np.asarray(angles, float)
    a = a[np.isfinite(a)]
    out = rayleigh(2 * a)
    m = circ_mean(2 * a)
    out["mean_axis"] = float(m["mean"] / 2)
    return out


def axial_dispersion(angles) -> float:
    """Circular standard deviation of axial data, in radians.

    Zero for a body swaying along one fixed line, rising towards the isotropic case.
    """
    a = np.asarray(angles, float)
    a = a[np.isfinite(a)]
    if len(a) < 2:
        return float("nan")
    R = np.abs(np.sum(np.exp(2j * a))) / len(a)
    return float(np.sqrt(-2 * np.log(max(R, 1e-12))) / 2)


def circ_corr(a, b) -> dict:
    """Correlation between two circular variables, after Jammalamadaka and Sengupta."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    n = len(a)
    if n < 4:
        return {"r": float("nan"), "p": float("nan"), "n": n}
    sa = np.sin(a - circ_mean(a)["mean"])
    sb = np.sin(b - circ_mean(b)["mean"])
    r = np.sum(sa * sb) / np.sqrt(np.sum(sa**2) * np.sum(sb**2))
    l20, l02 = np.mean(sa**2), np.mean(sb**2)
    l22 = np.mean(sa**2 * sb**2)
    z = np.sqrt(n * l20 * l02 / l22) * r
    return {"r": float(r), "p": float(2 * (1 - _stats.norm.cdf(abs(z)))), "n": n}


def circ_corr_linear(angles, x) -> dict:
    """Correlation between a circular variable and a linear one.

    The question behind "does quantity of motion depend on the time of day", or on the day of
    the year, where the predictor wraps and the outcome does not.
    """
    a = np.asarray(angles, float)
    x = np.asarray(x, float)
    m = np.isfinite(a) & np.isfinite(x)
    a, x = a[m], x[m]
    n = len(a)
    if n < 4:
        return {"r": float("nan"), "p": float("nan"), "n": n}
    rxs = np.corrcoef(x, np.sin(a))[0, 1]
    rxc = np.corrcoef(x, np.cos(a))[0, 1]
    rcs = np.corrcoef(np.sin(a), np.cos(a))[0, 1]
    r2 = (rxc**2 + rxs**2 - 2 * rxc * rxs * rcs) / (1 - rcs**2)
    r2 = float(np.clip(r2, 0, 1))
    p = 1 - _stats.chi2.cdf(n * r2, 2)
    return {"r": float(np.sqrt(r2)), "r2": r2, "p": float(p), "n": n}


def vtest(angles, mu: float) -> dict:
    """Test for a preferred direction at a specified angle.

    More powerful than :func:`rayleigh` when there is a prior expectation of where the
    preference lies — a hypothesis that sway is front-to-back, for instance, rather than a
    search for whichever direction happens to win.
    """
    a = np.asarray(angles, float)
    a = a[np.isfinite(a)]
    n = len(a)
    if n < 3:
        return {"V": float("nan"), "p": float("nan"), "n": n}
    R = np.abs(np.sum(np.exp(1j * a))) / n
    mean = circ_mean(a)["mean"]
    v = R * np.cos(mean - mu)
    u = v * np.sqrt(2 * n)
    return {"V": float(v), "u": float(u),
            "p": float(1 - _stats.norm.cdf(u)), "n": n}
