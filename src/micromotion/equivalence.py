"""Testing that an effect is absent, rather than failing to show that it is present.

Most of this corpus's interesting results are nulls: no environment effect, no seasonal rhythm,
no coupling between performers, no association between sound level and quantity of motion. A
non-significant test does not support any of those claims. It says the data are compatible with
no effect, and equally compatible with an effect too small for this sample to resolve. Stated as
"no effect was found", that is an overclaim, and reviewers say so.

Equivalence testing states the claim the reports actually want to make. Two one-sided tests
(TOST) invert the usual logic: the null is that the effect is at least as large as some bound,
and rejecting it supports the statement that the effect is smaller than that bound. The bound is a smallest
effect size of interest, and choosing it is a scientific judgement, not a statistical one --
which is the point. "No effect" is not a testable claim; "smaller than half a millimetre per
second" is.

Report both together. A result that is neither significant nor equivalent is genuinely
inconclusive, and saying so is more honest than either alternative.

    >>> tost_paired(before, after, bound=0.5)         # doctest: +SKIP
    {'equivalent': True, 'p': 0.004, ...}
"""

from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = ["tost_paired", "tost_independent", "equivalence_correlation", "interpret"]


def _verdict(p_tost: float, p_nhst: float, alpha: float) -> str:
    """The four outcomes, which is one more than people usually allow for."""
    sig, equ = p_nhst < alpha, p_tost < alpha
    if sig and not equ:
        return "effect"
    if equ and not sig:
        return "equivalent"
    if equ and sig:
        return "trivial"          # real, and smaller than the bound: both claims hold
    return "inconclusive"         # neither shown; the sample cannot resolve the question


def tost_paired(a, b, bound: float, alpha: float = 0.05) -> dict:
    """Two one-sided tests on a paired difference, against a bound in the data's own units.

    ``bound`` is the smallest difference worth caring about. Pass it in the units of ``a`` and
    ``b`` -- mm/s for quantity of motion, breaths per minute for a rate -- rather than as a
    standardised effect size, because a reader can argue about millimetres and cannot argue
    about Cohen's d.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    d = a[m] - b[m]
    n = len(d)
    if n < 3:
        raise ValueError(f"need at least 3 paired observations, got {n}")
    if bound <= 0:
        raise ValueError("bound must be positive; it is the smallest effect worth caring about")
    se = d.std(ddof=1) / np.sqrt(n)
    if se == 0:
        se = np.finfo(float).tiny
    df = n - 1
    t_lo = (d.mean() + bound) / se          # H0: difference <= -bound
    t_hi = (d.mean() - bound) / se          # H0: difference >= +bound
    p_tost = max(stats.t.sf(t_lo, df), stats.t.cdf(t_hi, df))
    p_nhst = float(stats.ttest_rel(a[m], b[m]).pvalue)
    half = stats.t.ppf(1 - alpha, df) * se  # the 90% interval TOST is equivalent to at alpha=.05
    return {
        "mean_difference": float(d.mean()),
        "ci_low": float(d.mean() - half),
        "ci_high": float(d.mean() + half),
        "bound": float(bound),
        "p_equivalence": float(p_tost),
        "p_difference": p_nhst,
        "equivalent": bool(p_tost < alpha),
        "verdict": _verdict(p_tost, p_nhst, alpha),
        "n": n,
    }


def tost_independent(a, b, bound: float, alpha: float = 0.05) -> dict:
    """The same for two independent samples, using Welch's standard error."""
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        raise ValueError(f"need at least 3 observations per group, got {len(a)} and {len(b)}")
    if bound <= 0:
        raise ValueError("bound must be positive")
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    if se == 0:
        se = np.finfo(float).tiny
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    diff = a.mean() - b.mean()
    p_tost = max(stats.t.sf((diff + bound) / se, df), stats.t.cdf((diff - bound) / se, df))
    p_nhst = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
    half = stats.t.ppf(1 - alpha, df) * se
    return {
        "mean_difference": float(diff),
        "ci_low": float(diff - half),
        "ci_high": float(diff + half),
        "bound": float(bound),
        "p_equivalence": float(p_tost),
        "p_difference": p_nhst,
        "equivalent": bool(p_tost < alpha),
        "verdict": _verdict(p_tost, p_nhst, alpha),
        "n": (len(a), len(b)),
    }


def equivalence_correlation(r: float, n: int, bound: float, alpha: float = 0.05) -> dict:
    """Is a correlation smaller in magnitude than ``bound``?

    For the corpus's many "no association" results, which are reported as correlations rather
    than as mean differences. Works on Fisher's z, where the sampling distribution is normal
    with a standard error that does not depend on the correlation itself.

    Takes ``r`` and ``n`` rather than the raw series, so it can be applied to a correlation that
    is already published without recomputing it.
    """
    if not -1 < r < 1:
        raise ValueError(f"r must be strictly inside (-1, 1), got {r}")
    if n < 4:
        raise ValueError(f"need n >= 4 for Fisher's z, got {n}")
    if not 0 < bound < 1:
        raise ValueError("bound must be a correlation strictly inside (0, 1)")
    z, zb = np.arctanh(r), np.arctanh(bound)
    se = 1.0 / np.sqrt(n - 3)
    p_tost = max(stats.norm.sf((z + zb) / se), stats.norm.cdf((z - zb) / se))
    p_nhst = float(2 * stats.norm.sf(abs(z) / se))
    crit = stats.norm.ppf(1 - alpha) * se
    return {
        "r": float(r),
        "ci_low": float(np.tanh(z - crit)),
        "ci_high": float(np.tanh(z + crit)),
        "bound": float(bound),
        "p_equivalence": float(p_tost),
        "p_difference": p_nhst,
        "equivalent": bool(p_tost < alpha),
        "verdict": _verdict(p_tost, p_nhst, alpha),
        "n": int(n),
    }


def interpret(result: dict) -> str:
    """One sentence a report can quote, naming the bound rather than hiding it."""
    b = result["bound"]
    v = result["verdict"]
    lo, hi = result["ci_low"], result["ci_high"]
    if v == "equivalent":
        return (f"equivalent to zero within ±{b:g}: the 90% interval [{lo:.3g}, {hi:.3g}] "
                f"lies inside the bound (p = {result['p_equivalence']:.3g})")
    if v == "effect":
        return (f"a real effect larger than ±{b:g} cannot be excluded: interval "
                f"[{lo:.3g}, {hi:.3g}] (p = {result['p_difference']:.3g})")
    if v == "trivial":
        return (f"statistically detectable but smaller than ±{b:g}: interval "
                f"[{lo:.3g}, {hi:.3g}]")
    return (f"inconclusive at a bound of ±{b:g}: the interval [{lo:.3g}, {hi:.3g}] is compatible "
            f"both with no effect and with one worth caring about; this sample cannot decide")
