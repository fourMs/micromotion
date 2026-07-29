"""Did these people move at the same moments?

Everything else in this package describes one recording, or relates a pair. This module asks
the group question: given many people recorded simultaneously, did their movements coincide in
time more than chance allows, and if so, when?

That question needs a null hypothesis with some care in it. Comparing each pair and averaging
loses the timing. Shuffling samples destroys each person's own rhythm along with the alignment.
The scheme used here shifts each person's event train by an independent random offset drawn
from a bounded range: every individual keeps their own event rate and local structure, and only
the alignment between people is destroyed. That is the null that "they moved together" should
be tested against.

Two related things are provided. :func:`event_train` turns a continuous signal into a point
process of events. :func:`coincidence_test` asks whether those events line up across people,
returning both a single score for the whole recording and a p value at each moment, so that the
moments where the group actually converged can be located rather than inferred.

Method after Finn Upham's activity-analysis work, used with permission and reimplemented here;
see `finn42/aa_test_package`. The stilling-response statistic below is from Upham, Hoffding and
Rosas, "The Stilling Response" (Music & Science, 2024).
"""

from __future__ import annotations

import numpy as np


def event_train(x, fs: float, frame_s: float = 1.0, threshold: float | None = None,
                kind: str = "increase") -> np.ndarray:
    """Turn a continuous signal into a binary point process of events.

    An event is a change across a window of ``frame_s`` seconds that exceeds ``threshold``.
    ``kind`` selects ``"increase"``, ``"decrease"`` or ``"change"`` for either direction.

    ``threshold`` defaults to one standard deviation of the framed differences, which makes the
    definition of "an event" relative to how much this person moves rather than absolute. That
    matters when the group wears different sensors on different parts of the body: an absolute
    threshold would count only the noisiest participants as ever doing anything.
    """
    x = np.asarray(x, float)
    half = max(1, int(round(frame_s * fs / 2)))
    idx = np.arange(len(x))
    lo = np.clip(idx - half, 0, len(x) - 1)
    hi = np.clip(idx + half, 0, len(x) - 1)
    d = x[hi] - x[lo]

    if kind == "decrease":
        d = -d
    elif kind == "change":
        d = np.abs(d)
    elif kind != "increase":
        raise ValueError(f"unknown kind {kind!r}")

    if threshold is None:
        threshold = np.nanstd(d)
    return (d >= threshold).astype(float)


def coincidence_test(trains, fs: float, n_surrogates: int = 1000,
                     shift_range_s: float = 30.0, frame_s: float = 1.0, rng=None) -> dict:
    """Test whether events coincide across people more than chance.

    ``trains`` is a sequence of equal-length binary event trains, one per person, already on a
    common timebase.

    Returns the observed coincidence count over time, the surrogate mean, a ``p`` value at every
    sample, a ``surprise`` transform of it, and two summaries of the whole recording.

    Read ``frac_significant`` before ``score``. ``score`` is the mean of -log10(p) over every
    sample, kept because it is the published statistic, but it is insensitive to exactly the
    case these recordings usually present. Measured on twenty simulated people whose event
    trains were independent apart from a few shared moments: three shared moments scored 0.32
    and sixty scored 0.47, while wholly independent trains scored 0.34. The three-moment case
    scored *below* the null case, because a mean over three thousand samples is dominated by the
    99.9 per cent of them where nothing was happening. Over the same runs
    ``frac_significant`` went 0.012, 0.034, 0.100, 0.200 against 0.004 for independence, which
    is the separation you want.

    The surrogates shift each person independently by up to ``shift_range_s`` seconds. Keep that
    range comfortably longer than the timescale you are testing for and shorter than the
    recording, or the null starts to resemble the data.
    """
    rng = rng or np.random.default_rng()
    T = np.asarray(trains, float)
    if T.ndim != 2:
        raise ValueError("trains must be (n_people, n_samples)")
    n_people, n = T.shape
    win = max(1, int(round(frame_s * fs)))

    def framed(a):
        c = np.cumsum(np.insert(a, 0, 0.0))
        half = win // 2
        lo = np.clip(np.arange(n) - half, 0, n)
        hi = np.clip(np.arange(n) - half + win, 0, n)
        return c[hi] - c[lo]

    observed = framed(T.sum(axis=0))

    max_shift = int(round(shift_range_s * fs))
    null = np.empty((n_surrogates, n))
    for s in range(n_surrogates):
        acc = np.zeros(n)
        for p in range(n_people):
            acc += np.roll(T[p], int(rng.integers(-max_shift, max_shift + 1)))
        null[s] = framed(acc)

    # One-sided: how often does chance reach what was observed?
    p = (np.sum(null >= observed[None, :], axis=0) + 1) / (n_surrogates + 1)
    max_surprise = np.log10(n_surrogates + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        surprise = np.log10((1 - p) / p)
    score = float(-np.log10(p + 10.0 ** (-max_surprise)).mean())
    frac = float((p < 0.01).mean())

    return {
        "observed": observed,
        "null_mean": null.mean(axis=0),
        "null_sd": null.std(axis=0),
        "p": p,
        "surprise": np.clip(np.nan_to_num(surprise, neginf=-max_surprise,
                                          posinf=max_surprise), -max_surprise, max_surprise),
        "score": score,
        "frac_significant": frac,
        "n_people": n_people,
        "n_surrogates": n_surrogates,
    }


def participation_ratio(series, event_times, fs: float, pre=(-3.0, -2.0),
                        post=(0.0, 1.0)) -> np.ndarray:
    """The fraction of people whose movement decreased after each event.

    ``series`` is (n_people, n_samples) of quantity of motion on a common timebase;
    ``event_times`` are moments of interest in seconds. For each event, every person's mean in
    the ``pre`` window is compared with their mean in the ``post`` window, and the statistic is
    the proportion who went down.

    Counting signs rather than sizes is the point. Participants in these recordings wear
    different sensors in different places, and their quantity of motion differs by more than an
    order of magnitude, so any statistic that averages magnitudes is dominated by whoever wore
    the noisiest device. A proportion is immune to that, and to missing data.

    Compare the result against :func:`sliding_null`, not against 0.5 — see that function for
    why.
    """
    S = np.asarray(series, float)
    if S.ndim != 2:
        raise ValueError("series must be (n_people, n_samples)")
    out = []
    for t in np.atleast_1d(event_times):
        a = _window_mean(S, fs, t + pre[0], t + pre[1])
        b = _window_mean(S, fs, t + post[0], t + post[1])
        d = b - a
        d = d[np.isfinite(d)]
        n_down, n_up = int((d < 0).sum()), int((d > 0).sum())
        out.append(n_down / (n_down + n_up) if (n_down + n_up) else np.nan)
    return np.asarray(out, float)


def sliding_null(series, fs: float, step_s: float = 0.1, **kw) -> np.ndarray:
    """The same statistic computed at every moment of the recording.

    This is the null distribution :func:`participation_ratio` should be judged against, and it
    is better than a theoretical one for a reason worth stating: people standing still are not
    a coin flip. Movement decays after any excursion, so at a randomly chosen moment rather more
    than half of a group is usually already slowing down. Testing observed events against 0.5
    would find an effect in any recording; testing them against this finds one only where the
    events beat the recording's own baseline.

    Compare the two with a one-sided Kolmogorov-Smirnov test.
    """
    S = np.asarray(series, float)
    n = S.shape[1]
    pre = kw.get("pre", (-3.0, -2.0))
    post = kw.get("post", (0.0, 1.0))
    first = -pre[0]
    last = n / fs - post[1]
    if last <= first:
        return np.asarray([], float)
    times = np.arange(first, last, step_s)
    return participation_ratio(S, times, fs, pre=pre, post=post)


def _window_mean(S, fs, t0, t1):
    i0 = int(round(t0 * fs))
    i1 = int(round(t1 * fs))
    i0, i1 = max(0, i0), min(S.shape[1], i1)
    if i1 <= i0:
        return np.full(S.shape[0], np.nan)
    with np.errstate(invalid="ignore"):
        return np.nanmean(S[:, i0:i1], axis=1)


def sequential_stability(x) -> float:
    """How steady a per-cycle measure is, ignoring slow drift.

    The median absolute difference between consecutive values. Unlike a standard deviation it is
    unmoved by a gradual trend, so it answers "was this person's breathing steady" over a window
    in which their rate is also slowly changing — which is the usual situation.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.median(np.abs(np.diff(x)))) if len(x) > 1 else float("nan")
