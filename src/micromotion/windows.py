"""Comparing conditions over windows of the same length.

A paired contrast between two conditions is only about the conditions if the two were measured
over comparable stretches of time. That is easy to assume and easy to get wrong, because segment
duration usually follows the STIMULUS rather than the design: a running order plays a track for as
long as the track lasts and leaves whatever gap it leaves, so "music" and "silence" end up
different lengths without anyone deciding they should be.

It matters because a quantity averaged over a longer window settles further toward its middle. Two
conditions measured over unequal windows are estimated with unequal smoothing, so the difference
between them is changed by the schedule rather than by the participants. In the corpus this package
was written for, the effect was to SUPPRESS: equalising the windows raised a music-versus-silence
contrast at 0.5-1 Hz from +1.70 to +2.10 per cent and moved two further frequency bands from null
to significant. Elsewhere in the same corpus it worked the other way and produced a false positive
below 0.5 Hz. The direction is not predictable, which is the argument for checking rather than
reasoning about it.

Two functions, in the order they should be used. :func:`balance` asks whether the windows differ by
condition at all, and is the one-line check that would have caught this years earlier than it was
caught. :func:`equalise` truncates every segment to a common length so that they do not.

    >>> import numpy as np
    >>> onset  = np.array([0.,  60., 120., 180.])
    >>> offset = np.array([45., 105., 165., 240.])      # silences 45 s, music 60 s
    >>> cond   = np.array(["s", "s", "s", "m"])
    >>> b = balance(onset, offset, cond)
    >>> b.balanced
    False
    >>> new_offset = equalise(onset, offset)
    >>> (new_offset - onset).tolist()
    [45.0, 45.0, 45.0, 45.0]

CAP PER GROUP, NOT ACROSS THE STUDY. Where segments come from several recordings, editions or
sessions, pass ``by=`` so each group is equalised against its own shortest segment. A single cap
across the whole study equalises the conditions AND shortens every window, and those pull in
opposite directions: applied to six recording sessions whose segments ran from 20 to 180 s, a flat
cap made a real effect look like it had collapsed, purely by discarding the signal of the sessions
that had recorded longest.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Balance", "balance", "equalise"]


@dataclass
class Balance:
    """What :func:`balance` found. Truthy when the windows are comparable.

    ``by_condition`` maps each condition to ``(n, median, min, max)`` in seconds. ``ratio`` is the
    longest condition median over the shortest, so 1.0 is perfect balance. ``balanced`` applies the
    tolerance the caller asked for.
    """

    by_condition: dict = field(default_factory=dict)
    ratio: float = 1.0
    balanced: bool = True
    tol: float = 0.05
    groups: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.balanced

    def __str__(self) -> str:
        if self.groups:
            worst = max(self.groups, key=lambda g: self.groups[g].ratio)
            rows = "\n".join(
                f"    {g!s:<12} ratio {b.ratio:5.2f}   "
                f"{'balanced' if b.balanced else 'CONFOUNDED WITH DURATION'}"
                for g, b in sorted(self.groups.items(), key=lambda kv: str(kv[0]))
            )
            return (f"{rows}\n    worst group is {worst!s} at {self.ratio:.2f}x. "
                    f"Read these and not a pooled figure, which averages them away.")
        rows = "\n".join(
            f"    {c!s:<12} n={n:<5d} median {med:7.1f} s   range {lo:.1f}-{hi:.1f}"
            for c, (n, med, lo, hi) in sorted(self.by_condition.items())
        )
        verdict = (
            "windows are comparable"
            if self.balanced
            else f"WINDOWS DIFFER BY CONDITION: longest median is {self.ratio:.2f}x the shortest. "
            "A contrast computed on these is confounded with duration; call equalise() first."
        )
        return f"{rows}\n    {verdict}"


def _durations(onset_s, offset_s):
    onset = np.asarray(onset_s, float)
    offset = np.asarray(offset_s, float)
    if onset.shape != offset.shape:
        raise ValueError(f"onset and offset differ in shape: {onset.shape} vs {offset.shape}")
    d = offset - onset
    if np.any(d <= 0):
        raise ValueError("every segment must end after it starts")
    return onset, offset, d


def _balance_one(d, cond, tol):
    by = {}
    for c in np.unique(cond):
        sel = d[cond == c]
        by[c.item() if hasattr(c, "item") else c] = (
            int(sel.size), float(np.median(sel)), float(sel.min()), float(sel.max()))
    meds = [m for _n, m, _lo, _hi in by.values()]
    ratio = float(max(meds) / min(meds)) if len(meds) > 1 and min(meds) > 0 else 1.0
    return Balance(by_condition=by, ratio=ratio, balanced=ratio <= 1.0 + tol, tol=tol)


def balance(onset_s, offset_s, condition, by=None, tol: float = 0.05) -> Balance:
    """Do the conditions being compared occupy windows of the same length?

    Returns a :class:`Balance`, which is falsy when they do not. ``tol`` is how far the ratio of
    condition medians may sit from 1.0 before the answer is no; the default of 0.05 allows the few
    per cent that rounding a segment table to whole seconds produces.

    PASS ``by=`` WHENEVER THE SEGMENTS COME FROM MORE THAN ONE RECORDING, SESSION OR EDITION, and
    read the per-group result rather than the pooled one. Pooling averages the imbalances and can
    hide a severe one almost completely. On the corpus this was written for, the pooled ratio over
    six editions is 1.07 — a few per cent, easy to wave away — while one edition inside it sits at
    9.00 and three others between 1.33 and 1.60. The pooled figure is the one that reassures, and
    it is the wrong one. With ``by``, ``ratio`` is the worst group's and ``groups`` holds each.

    Run this before any paired contrast between conditions. It costs one line and answers a
    question that is otherwise invisible: nothing about a table of onsets and offsets announces
    that one condition is systematically longer than another, and no amount of checking the
    analysis code will reveal it, because the fault is in the design of the running order rather
    than in the arithmetic.
    """
    _, _, d = _durations(onset_s, offset_s)
    cond = np.asarray(condition)
    if cond.shape != d.shape:
        raise ValueError(f"condition has shape {cond.shape}, expected {d.shape}")
    if by is None:
        return _balance_one(d, cond, tol)

    groups = np.asarray(by)
    if groups.shape != d.shape:
        raise ValueError(f"by has shape {groups.shape}, expected {d.shape}")
    per = {}
    for g in np.unique(groups):
        sel = groups == g
        per[g.item() if hasattr(g, "item") else g] = _balance_one(d[sel], cond[sel], tol)
    worst = max(per.values(), key=lambda b: b.ratio)
    return Balance(by_condition=worst.by_condition, ratio=worst.ratio,
                   balanced=all(b.balanced for b in per.values()), tol=tol, groups=per)


def equalise(onset_s, offset_s, cap_s: float | None = None, by=None):
    """Truncate every segment to a common length, returning new offsets.

    Each segment keeps its own onset and is cut from the start, so the equalised window is the
    beginning of the segment rather than a slice from its middle. That is deliberate: the start is
    the part every segment has.

    ``cap_s`` sets the length; by default it is the shortest segment present, which keeps as much
    of every segment as can be kept while making them equal. ``by`` groups the segments, so each
    group is capped against its own shortest segment rather than against the study's — see the
    module docstring for why a single cap across groups is the wrong instrument.

    Only offsets are returned. The onsets are unchanged, so the caller's other columns stay aligned
    and nothing needs reordering.
    """
    onset, offset, d = _durations(onset_s, offset_s)

    if by is None:
        cap = float(d.min()) if cap_s is None else float(cap_s)
        if cap <= 0:
            raise ValueError("cap_s must be positive")
        return onset + np.minimum(d, cap)

    groups = np.asarray(by)
    if groups.shape != d.shape:
        raise ValueError(f"by has shape {groups.shape}, expected {d.shape}")
    out = offset.copy()
    for g in np.unique(groups):
        sel = groups == g
        cap = float(d[sel].min()) if cap_s is None else float(cap_s)
        if cap <= 0:
            raise ValueError("cap_s must be positive")
        out[sel] = onset[sel] + np.minimum(d[sel], cap)
    return out
