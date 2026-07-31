"""How many independent things is a descriptor set measuring, and how much of one is the person?

Two reductions that this corpus kept re-implementing per report, with the arithmetic drifting
between copies. Both are here so there is one of each.

They answer questions that are easy to confuse. :func:`effective_dimensionality` asks how many
independent axes a *set of measures* spans -- whether eleven descriptors are eleven findings or
three wearing different names. :func:`intraclass_correlation` asks, of a *single* measure, how
much of its variance is the person rather than the occasion -- whether it is a trait or a state.

**A name collision worth knowing about.** ``group.participation_ratio`` is a different quantity
with a similar name: the fraction of a group whose movement decreased after an event. The
participation ratio *of an eigenvalue spectrum*, which is what effective dimensionality means
here, is deliberately not called that.
"""

from __future__ import annotations

import warnings

import numpy as np


def effective_dimensionality(x, rank: bool = True, by=None) -> dict:
    """How many independent dimensions a set of descriptors spans.

    ``x`` is (n_observations, n_descriptors). Returns the variance share of each component, the
    number of components needed for 80 and 90 per cent, and the **participation ratio**
    ``(sum lambda)^2 / sum(lambda^2)`` -- an effective count that needs no cutoff and is not
    an integer.

    Three choices are baked in because getting them wrong is what made earlier versions of this
    disagree with each other:

    ``rank=True`` correlates the *ranks* rather than the values. Descriptors here are heavy-tailed
    -- burstiness is a ratio of a 99th percentile to a median -- and on raw values a single
    descriptor with a long tail dominates the first component and the answer becomes a statement
    about that descriptor's outliers.

    ``by`` standardises within groups before pooling, and should be the recording session, edition
    or collection. Without it, a between-group difference in level appears as shared variance and
    inflates the first component: descriptors do not become more correlated because two editions
    were recorded at different rates, but they look it.

    Columns that are constant or all-NaN are dropped, and the count is reported, because a
    degenerate column silently adds an eigenvalue of zero and deflates the participation ratio.
    """
    import pandas as pd

    d = pd.DataFrame(x).apply(pd.to_numeric, errors="coerce")
    if by is not None:
        g = pd.Series(list(by), index=d.index)
        d = d.groupby(g).rank() if rank else d
        d = d.groupby(g).transform(lambda s: (s - s.mean()) / s.std())
    elif rank:
        d = d.rank()
        d = (d - d.mean()) / d.std()
    else:
        d = (d - d.mean()) / d.std()

    n_before = d.shape[1]
    d = d.dropna(axis=1, how="all").dropna()
    d = d.loc[:, d.std() > 0]
    dropped = n_before - d.shape[1]
    if d.shape[1] < 2:
        raise ValueError("need at least two non-degenerate descriptors")

    ev = np.linalg.eigvalsh(np.corrcoef(d.to_numpy(), rowvar=False))[::-1]
    ev = np.clip(ev, 0.0, None)
    frac = ev / ev.sum()
    cum = np.cumsum(frac)
    return dict(
        variance_fraction=frac,
        n_for_80=int(np.searchsorted(cum, 0.80) + 1),
        n_for_90=int(np.searchsorted(cum, 0.90) + 1),
        participation_ratio=float(ev.sum() ** 2 / (ev ** 2).sum()),
        n_observations=int(len(d)),
        n_descriptors=int(d.shape[1]),
        n_dropped=int(dropped),
    )


def intraclass_correlation(values, groups, log: bool | None = None) -> dict:
    """The share of a measure's variance that is between groups rather than within them.

    Fitted as a random-intercept mixed model, ``value ~ 1`` with a random intercept per group,
    which is the estimator the corpus uses for "is this a trait or a state": ``groups`` is the
    person and the residual is the occasion.

    ``log=None`` log-transforms when every value is strictly positive, because these are scale
    quantities whose residuals are otherwise skewed; pass ``False`` to force the raw scale.

    Returns the ICC, both variance components, the counts, and ``boundary``. **Check
    ``boundary``.** A random-effect variance can be estimated at exactly zero, which is the
    optimiser hitting the edge of the parameter space rather than a measurement of no group
    effect -- the difference matters when the number of groups is small, and reporting ``0.000``
    from such a fit implies a precision that is not there.
    """
    import pandas as pd
    import statsmodels.formula.api as smf

    d = pd.DataFrame({"y": pd.to_numeric(pd.Series(list(values)), errors="coerce"),
                      "g": pd.Series(list(groups)).astype(str)}).dropna()
    if len(d) < 3 or d.g.nunique() < 2:
        raise ValueError("need at least two groups and three observations")
    used_log = bool(d.y.min() > 0) if log is None else bool(log)
    if used_log:
        d["y"] = np.log(d.y)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.mixedlm("y ~ 1", d, groups=d.g).fit(reml=True)
    between = float(np.asarray(fit.cov_re)[0, 0])
    within = float(fit.scale)
    total = between + within
    return dict(
        icc=float(between / total) if total > 0 else float("nan"),
        var_between=between,
        var_within=within,
        n=int(len(d)),
        n_groups=int(d.g.nunique()),
        log=used_log,
        boundary=bool(between <= 1e-9),
    )
