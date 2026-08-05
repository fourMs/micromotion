"""Checks that fail loudly on the errors this corpus makes silently.

Every check here exists because the failure it catches happened, went unnoticed, and produced a
plausible number that someone used. None of them raised anything at the time. That is the common
thread and the reason for the module: the corpus's characteristic failure is not a crash but a
believable wrong answer, and the only defence is a check that runs on every build.

The intended use is a gate, not a report. Run :func:`validate_series` over everything a
harmonised table is about to be built from, and refuse to build if anything comes back at
``"error"``. A finding at ``"warning"`` is something to record in the manifest beside the number
it affects.

Each check names the incident that motivated it, with its numbers, so that the tolerance is
arguable rather than magic.
"""

from __future__ import annotations

import hashlib
import itertools
import os
from dataclasses import dataclass

import numpy as np

from .resample import measured_rate


@dataclass(frozen=True)
class Finding:
    """One thing wrong with one series or file."""

    check: str
    severity: str
    message: str
    where: str = ""

    def __str__(self) -> str:
        loc = f"{self.where}: " if self.where else ""
        return f"[{self.severity}] {loc}{self.check} — {self.message}"


def _finding(check, severity, message, where):
    return Finding(check=check, severity=severity, message=message, where=where)


def zero_triplets(x, where: str = "", max_fraction: float = 0.0) -> list[Finding]:
    """Rows where every coordinate is exactly zero, which are gaps and not positions.

    Qualisys writes a dropped frame as ``0.000 0.000 0.000``. That is a point on the laboratory
    floor about a metre and a half below a standing head, so a reader that takes it literally
    sees the head leave and return. A median-based measure barely notices; anything that sums or
    integrates does. On one 2021 recording, 93 such frames out of 118698 gave the head a path
    length of 119.8 m where the true figure is 11.0 m.

    Exact zeros in all axes at once do not occur in real optical data, so the default tolerance
    is zero. Raise ``max_fraction`` only for a sensor whose true output can sit at the origin.
    """
    x = np.atleast_2d(np.asarray(x, float).T).T
    if x.shape[1] < 2:
        return []
    gap = (x == 0.0).all(axis=1)
    n = int(gap.sum())
    if n == 0 or n / len(x) <= max_fraction:
        return []
    runs = [len(list(g)) for k, g in itertools.groupby(gap) if k]
    return [_finding(
        "zero_triplets", "error",
        f"{n} of {len(x)} samples ({100 * n / len(x):.2f} %) are exactly zero in every axis, "
        f"longest run {max(runs)}. These are gaps; convert them to NaN before filtering",
        where)]


def marker_average(markers, where: str = "", max_gap_fraction: float = 0.5) -> list[Finding]:
    """Check a set of markers before averaging them into one position.

    Averaging several markers into a single "head" or "trunk" position is routine and looks
    harmless. It is not, if the gaps have not been repaired first, because the usual repair
    happens at the *end* of a pipeline and an average destroys the evidence on the way in: the
    mean of two real coordinates and one zero triplet is a perfectly finite point that no later
    gap check will flag.

    The damage is a clean multiplicative bias. Markers on one rigid segment move together, so
    with ``n`` markers of which ``k`` are dead, the averaged position moves at about
    ``(n - k) / n`` of the true amplitude -- and a speed derived from it is understated by the
    same factor. One marker dead out of three is exactly two thirds, which reads as a third
    less motion.

    This happened. Four recordings in one 86-session collection carried a head marker that was
    never tracked, and their quantity of motion was reported 33.3 per cent low for as long as
    the collection existed, looking like unusually still standing rather than like a fault.

    Pass a mapping of name to (n, 3) array. Repair each marker with
    :func:`micromotion.validate.zero_triplets` and NaN before averaging, then use ``nanmean``.
    

    The same bias appears one level up in :func:`micromotion.qom.group_qom`, which
    averaged over markers that were not visible until 1.0. If you are averaging
    positions here, check how you are averaging speeds there.
    """
    findings: list[Finding] = []
    if not markers:
        return findings
    dead, partial = [], []
    for name, arr in dict(markers).items():
        x = np.atleast_2d(np.asarray(arr, float).T).T
        if x.shape[1] < 2 or not len(x):
            continue
        bad = (x == 0.0).all(axis=1) | ~np.isfinite(x).all(axis=1)
        frac = float(bad.mean())
        if frac >= 1.0:
            dead.append(name)
        elif frac > max_gap_fraction:
            partial.append((name, frac))

    n = len(markers)
    if dead:
        bias = (n - len(dead)) / n if n else 0.0
        findings.append(_finding(
            "marker_average", "error",
            f"{len(dead)} of {n} markers carry no data at all ({', '.join(sorted(dead))}). "
            f"Averaging as-is understates the amplitude by about {100 * (1 - bias):.1f} % "
            f"(factor {bias:.3f}). Repair gaps to NaN per marker and use nanmean",
            where))
    for name, frac in partial:
        findings.append(_finding(
            "marker_average", "warning",
            f"marker {name} is {100 * frac:.1f} % gaps; averaging it in biases the result "
            f"toward the origin over that stretch",
            where))
    return findings


def implausible_position(x, where: str = "", axis: int = 2, min_fraction: float = 0.3,
                         max_fraction: float = 2.5) -> list[Finding]:
    """Samples that put a marker somewhere a body cannot be.

    :func:`zero_triplets` catches a dropped frame written as three exact zeros. It cannot catch the
    near miss: a reconstruction that lands *close* to the laboratory origin without being exactly
    on it. Those samples pass every finiteness and sentinel test, because they are ordinary
    numbers, and they are not rare enough to ignore -- a corpus of 1018 optical person-recordings
    held two, one of which placed a head marker 139 mm *below the floor*.

    The test is physical rather than statistical: a marker on a standing body stays within a band
    around its own median height. Anything below ``min_fraction`` of that median, or above
    ``max_fraction``, is a tracking artefact rather than a posture.

    The damage is uneven, which is why this is worth checking separately. A median-based measure
    barely notices 0.5 per cent of samples. Anything spatial is destroyed by them: on that Sverm
    recording the sway extent read 977 mm where the true figure is about 48.

    Only meaningful for a marker whose median height is a real standing height, so recordings
    whose median falls below ``500`` in the array's own units are skipped.
    """
    x = np.atleast_2d(np.asarray(x, float).T).T
    if x.shape[1] <= axis:
        return []
    z = x[:, axis]
    z = z[np.isfinite(z)]
    if len(z) < 100:
        return []
    med = float(np.median(z))
    if med < 500:                       # not a standing-height marker; no expectation to test
        return []
    bad = (z < med * min_fraction) | (z > med * max_fraction)
    n = int(bad.sum())
    if n == 0:
        return []
    return [_finding(
        "implausible_position", "error",
        f"{n} of {len(z)} samples ({100 * n / len(z):.2f} %) place the marker outside "
        f"{min_fraction:g}-{max_fraction:g} times its median height of {med:.0f}, reaching "
        f"{z.min():.0f}. These are tracking artefacts, not postures; they survive a zero-triplet "
        f"check because they are not exactly zero, and they wreck spatial measures while barely "
        f"moving a median",
        where)]


def finite_fraction(x, where: str = "", min_finite: float = 0.8) -> list[Finding]:
    """Whether enough of a series survived to measure, and whether any of it did.

    A gap that runs off the start or end of a series cannot be interpolated — there is nothing
    on the far side — and a band-pass then spreads the surviving NaN across the whole recording.
    The result is an all-NaN series that is indistinguishable from an absent marker unless
    something looks. One Sverm 2012 recording lost a marker 548 s in and never regained it; the
    deposited value for it had been computed straight across the hole.

    An emptied series is an error. A merely thin one is a warning, because the caller may
    legitimately be measuring the longest clean span instead.
    """
    x = np.asarray(x, float)
    bad = np.isnan(x).any(axis=1) if x.ndim > 1 else np.isnan(x)
    finite = 1.0 - float(bad.mean())
    if finite == 0.0:
        return [_finding("finite_fraction", "error",
                         "the series is entirely NaN. A band-pass across an unbridgeable gap "
                         "does this, and the result looks like an absent sensor", where)]
    if finite < min_finite:
        return [_finding("finite_fraction", "warning",
                         f"only {100 * finite:.1f} % of samples are finite; measure the longest "
                         f"clean span rather than the whole series", where)]
    return []


def longest_finite_span(x) -> tuple[int, int]:
    """Start index and length of the longest run with no missing sample.

    What to measure over when a gap cannot be bridged. Filtering across the gap is not an
    option and dropping the samples silently closes it, which is worse: the series then claims
    a duration it does not have.
    """
    x = np.asarray(x, float)
    bad = np.isnan(x).any(axis=1) if x.ndim > 1 else np.isnan(x)
    best_start, best_len, i = 0, 0, 0
    for is_bad, group in itertools.groupby(bad):
        n = len(list(group))
        if not is_bad and n > best_len:
            best_start, best_len = i, n
        i += n
    return best_start, best_len


def timestamps(t, where: str = "") -> list[Finding]:
    """Whether a timestamp column is usable as a clock.

    Sorting a timestamp column into order is the tempting repair and the wrong one: it destroys
    the evidence that the clock misbehaved while leaving the samples in an order the sensor
    never produced. One balance-board collection carries 123 111 duplicate timestamps and 83
    that step backwards, which is a device fault to be recorded, not a sort key to be fixed.
    """
    t = np.asarray(t, float)
    out = []
    if len(t) < 2:
        return [_finding("timestamps", "error", f"only {len(t)} samples", where)]
    d = np.diff(t)
    back = int((d < 0).sum())
    dup = int((d == 0).sum())
    if back:
        out.append(_finding("timestamps", "error",
                            f"{back} timestamps step backwards; the clock is not monotonic and "
                            f"sorting would hide it", where))
    if dup:
        out.append(_finding("timestamps", "warning",
                            f"{dup} of {len(d)} intervals ({100 * dup / len(d):.2f} %) are zero, "
                            f"so samples share a timestamp", where))
    pos = d[d > 0]
    if len(pos) and pos.max() > 20 * np.median(pos):
        out.append(_finding("timestamps", "warning",
                            f"largest interval {pos.max():.4g} s against a median of "
                            f"{np.median(pos):.4g} s; the series has holes", where))
    return out


def rate_agreement(t, documented_hz: float, where: str = "",
                   tolerance: float = 0.02) -> list[Finding]:
    """Whether the rate written down matches the rate the timestamps imply.

    Measure the rate, do not read it. Documented rates in this corpus are wrong by up to 4.4 per
    cent, one record's was out by a factor of 37, and one championship's accelerometers turn out
    to run at 191.29–207.73 Hz against a nominal 200 — with the true rate a property of the
    individual device rather than the protocol, so three participants sharing a unit share a
    clock and everyone else does not.

    A disagreement is an error rather than a warning because every frequency-domain measure
    downstream scales with it, and nothing further along can detect it.
    """
    fs = measured_rate(t)
    if not np.isfinite(fs) or fs <= 0:
        return [_finding("rate_agreement", "error", "no rate could be measured", where)]
    if documented_hz is None or not np.isfinite(documented_hz) or documented_hz <= 0:
        return []
    rel = abs(fs - documented_hz) / documented_hz
    if rel > tolerance:
        return [_finding("rate_agreement", "error",
                         f"measured {fs:.4g} Hz against a documented {documented_hz:.4g} Hz, "
                         f"a difference of {100 * rel:.1f} %. Use the measured rate", where)]
    return []


def held_samples(x, where: str = "", max_run: int = 50) -> list[Finding]:
    """Long runs of an identical value, which mean a hold rather than a measurement.

    A sensor sampled below the rate it is stored at is written out with each value repeated, and
    the file then claims a rate the data does not carry. The Delsys accelerometers in this corpus
    are stored at 2000 Hz and repeat every value, so their real rate is far lower and any
    spectrum computed at the stored rate is wrong above the true Nyquist.
    """
    x = np.asarray(x, float)
    col = x[:, 0] if x.ndim > 1 else x
    col = col[np.isfinite(col)]
    if len(col) < max_run + 1:
        return []
    longest = max((len(list(g)) for _, g in itertools.groupby(col)), default=0)
    if longest > max_run:
        return [_finding("held_samples", "warning",
                         f"{longest} consecutive identical values; the stored rate is probably "
                         f"higher than the rate actually sampled", where)]
    return []


def frame_count(n: int, where: str = "") -> list[Finding]:
    """A sample count sitting exactly on a 16-bit boundary, which means a silent truncation.

    C3D stores its frame count in sixteen bits, so a conversion through it stops at 65535 and
    says nothing. Seven sessions in this corpus were nearly deposited that way: 327.7 s of a
    360 s recording, complete-looking, with the last thirty-two seconds gone.
    """
    if n in (65535, 65536):
        return [_finding("frame_count", "error",
                         f"exactly {n} samples, which is the 16-bit ceiling. A C3D conversion "
                         f"truncates here silently; re-export from the source", where)]
    return []


def edge_motion(speed, fs: float, where: str = "", edge_s: float = 10.0,
                baseline_s: tuple[float, float] = (60.0, 300.0),
                factor: float = 2.0) -> list[Finding]:
    """Whether a recording opens or closes with movement rather than standstill.

    A deposited standstill recording should contain standstill and nothing else. In practice
    exports are trimmed by hand, or not trimmed at all, and what survives at the edges is people
    walking into position, settling, or being told the recording has ended. That inflates
    anything computed over a short window and is invisible in a whole-recording median.

    ``speed`` is a band-limited speed series, whatever sensor it came from. The comparison is
    against the recording's own settled interior rather than an absolute threshold, because the
    quantity varies by two orders of magnitude across sensors in this corpus.

    Returns one finding per affected end, at ``"warning"``: settling is a fact about the
    recording to be recorded, not necessarily a fault to be fixed.
    """
    speed = np.asarray(speed, float)
    n = len(speed)
    lo, hi = int(baseline_s[0] * fs), int(min(baseline_s[1] * fs, n))
    if n < int(2 * edge_s * fs) + 1 or hi - lo < int(10 * fs):
        return []
    base = float(np.nanmedian(speed[lo:hi]))
    if not np.isfinite(base) or base <= 0:
        return []
    k = int(edge_s * fs)
    out = []
    for end, seg in (("start", speed[:k]), ("end", speed[-k:])):
        v = float(np.nanmedian(seg))
        if np.isfinite(v) and v > factor * base:
            out.append(_finding(
                "edge_motion", "warning",
                f"the first {edge_s:.0f} s move at {v:.3g} against a settled {base:.3g} "
                f"({v / base:.1f}x)" if end == "start" else
                f"the last {edge_s:.0f} s move at {v:.3g} against a settled {base:.3g} "
                f"({v / base:.1f}x)", where))
    return out


def settling_time(speed, fs: float, baseline_s: tuple[float, float] = (60.0, 300.0),
                  factor: float = 1.5, window_s: float = 5.0,
                  max_s: float = 120.0) -> tuple[float, float]:
    """How long each end of a recording takes to reach its settled level, in seconds.

    Returns ``(head, tail)``: the time to trim from the start and from the end so that what
    remains is within ``factor`` of the recording's own settled interior. Zero means that end is
    already settled. Use it to choose a trim rather than guessing one — a fixed twelve seconds
    was not enough for any recording in the collection it was chosen for.

    The search stops at ``max_s`` and returns that value, which should be read as "still moving
    when we stopped looking" rather than as a measurement.
    """
    speed = np.asarray(speed, float)
    n = len(speed)
    lo, hi = int(baseline_s[0] * fs), int(min(baseline_s[1] * fs, n))
    if hi - lo < int(10 * fs):
        return 0.0, 0.0
    base = float(np.nanmedian(speed[lo:hi]))
    if not np.isfinite(base) or base <= 0:
        return 0.0, 0.0
    w = max(1, int(window_s * fs))
    limit = min(int(max_s * fs), n // 2)

    def scan(x):
        t = 0
        while t + w <= limit:
            if np.nanmedian(x[t:t + w]) <= factor * base:
                return t / fs
            t += w
        return limit / fs

    return scan(speed), scan(speed[::-1])


def duplicate_files(paths, where: str = "") -> list[Finding]:
    """Files in a set that are byte-identical.

    A record shipped with the same recording under two names, one of them in no manifest, left
    over from a rename. It inflated the record and would have inflated any count taken by
    listing the directory.
    """
    by_hash: dict[str, list[str]] = {}
    for p in paths:
        try:
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for block in iter(lambda: fh.read(1 << 20), b""):
                    h.update(block)
        except OSError:
            continue
        by_hash.setdefault(h.hexdigest(), []).append(str(p))
    out = []
    for group in by_hash.values():
        if len(group) > 1:
            names = ", ".join(os.path.basename(p) for p in sorted(group))
            out.append(_finding("duplicate_files", "error",
                                f"byte-identical: {names}", where))
    return out


def validate_series(x, t=None, documented_hz: float | None = None, where: str = "",
                    min_finite: float = 0.8, expect_positions: bool = True) -> list[Finding]:
    """Run every applicable check on one series and return what is wrong with it.

    ``x`` is (n_samples, n_axes) or one-dimensional. Pass ``t`` to check the clock, and
    ``documented_hz`` to check it against what the record claims. Set ``expect_positions``
    False for a sensor whose output may legitimately be zero in every axis at once.
    """
    out: list[Finding] = []
    x = np.asarray(x, float)
    out += frame_count(len(x), where)
    if expect_positions:
        out += zero_triplets(x, where)
    out += finite_fraction(x, where, min_finite=min_finite)
    out += held_samples(x, where)
    if t is not None:
        out += timestamps(t, where)
        if documented_hz is not None:
            out += rate_agreement(t, documented_hz, where)
    return out


def errors(findings) -> list[Finding]:
    """Just the findings that should stop a build."""
    return [f for f in findings if f.severity == "error"]


def raise_on_error(findings) -> None:
    """Raise if anything in ``findings`` is an error, listing all of them.

    A build that continues past a finding produces a plausible wrong number, which is the exact
    failure this module exists to prevent.
    """
    bad = errors(findings)
    if bad:
        raise ValueError("validation failed:\n  " + "\n  ".join(str(f) for f in bad))
