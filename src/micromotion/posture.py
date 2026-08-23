"""The shape and extent of standing still.

Quantity of motion says how fast a body part moved. These say where it went: how far it
strayed, what area it covered, whether it swayed along one line or wandered in all
directions.

The pair is more informative than either alone. Two people with the same quantity of motion
can occupy regions differing several-fold, because speed and extent are close to independent
in this data — how much someone moves predicts only about a quarter of how large a region
they occupy.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as _stats

from .descriptors import effective_dimensionality


def sway_geometry(xy) -> dict:
    """Principal axis, anisotropy and dispersion of a two-dimensional trace.

    Give it the horizontal coordinates of a marker, or a centre-of-pressure track.

    ``anisotropy`` runs from 0 to 1: zero when the sway is equally wide in every direction,
    approaching one when it collapses onto a line. There are two other definitions of this
    quantity in circulation in this project — one where 1 means isotropic and one reporting
    only the angle — so the definition is stated rather than assumed. ``axis_deg`` is the
    direction of greatest sway, and is axial: 10 degrees and 190 degrees describe the same
    posture. Test it with :func:`micromotion.circular.rayleigh_axial`, never with the
    ordinary Rayleigh test.

    ``axis_deg`` IS IN THE RECORDING'S OWN FRAME AND IS NOT COMPARABLE ACROSS RECORDINGS
    unless you know that the frames agree. Two things have to hold: the laboratory
    coordinate convention, and which way the body was facing. A sway axis is anatomical --
    it is front-to-back -- so turning the person turns this number without anything about
    their posture changing.

    Both failed in one corpus on one afternoon, which is why this paragraph exists. One
    championship edition is stored with its horizontal axes about 90 degrees from every
    other, at the same concentration, so a comparison of mean axes across editions
    measured the laboratories rather than the bodies. And a three-performer collection
    recorded many sessions with the performers standing in a circle facing each other, so
    within-person axial concentration came out at R = 0.23 to 0.48 with axes spread over
    nearly 180 degrees, against 0.75 to 0.85 in a group all facing one way -- and an
    analysis read that as a weak personal trait when varying facing predicts it exactly.

    ``anisotropy`` and ``dispersion`` are rotation-invariant and carry neither problem.
    So is the CONCENTRATION of a set of axes, which is why a clustering result can survive
    where a mean angle cannot. If you need the angle itself across recordings, establish
    the facing geometry first; if you cannot, use the invariant quantities and say so.
    """
    p = np.asarray(xy, float)
    p = p[np.isfinite(p).all(axis=1)][:, :2]
    if len(p) < 3:
        return {"anisotropy": float("nan"), "axis_deg": float("nan")}
    c = p - p.mean(axis=0)
    _, s, vt = np.linalg.svd(c, full_matrices=False)
    lam = s**2 / max(len(c) - 1, 1)
    major, minor = float(lam[0]), float(lam[1])
    return {
        "anisotropy": float(1 - minor / major) if major > 0 else float("nan"),
        "axis_deg": float(np.degrees(np.arctan2(vt[0, 1], vt[0, 0])) % 180.0),
        "sd_major": float(np.sqrt(major)),
        "sd_minor": float(np.sqrt(minor)),
        "rms_radius": float(np.sqrt(np.mean(np.sum(c**2, axis=1)))),
        "range_major": float(np.ptp(c @ vt[0])),
        "range_minor": float(np.ptp(c @ vt[1])),
    }


def ellipse_area_95(xy) -> float:
    """Area of the 95 per cent confidence ellipse, in the input units squared.

    The standard summary of postural extent in the balance literature, which makes it the
    number to report when comparing against clinical work.
    """
    p = np.asarray(xy, float)
    p = p[np.isfinite(p).all(axis=1)][:, :2]
    if len(p) < 3:
        return float("nan")
    cov = np.cov(p.T)
    return float(np.pi * _stats.chi2.ppf(0.95, 2) * np.sqrt(max(np.linalg.det(cov), 0)))


def heading_persistence(xy, speed_percentile: float = 20.0) -> dict:
    """Does the trace reverse along a line, wander at random, or loop?

    Takes the heading of each step, and averages the cosine of the change in heading from
    one step to the next. It measures smoothness, not shape: +1 is a trace whose direction
    barely changes between samples, 0 is a random walk, and -1 is a trace that reverses at
    every single step.

    A back-and-forth sway along one line reads near +1, not -1, which is worth saying
    because the intuition runs the other way. Such a trace holds its heading for the whole
    of each excursion and reverses only at the turning points, so the reversals are a
    handful of steps among hundreds. Quiet standing in the championship corpus reads about
    0.95. Reaching -1 takes a zigzag that flips direction at the sampling rate, which is a
    signature of noise rather than of movement.

    ``straightness`` is the net displacement over the distance walked to achieve it, so it
    is near 0 for someone who stays put however much they move, and near 1 for someone who
    walks away in a straight line.

    The slowest steps are dropped before averaging, ``speed_percentile`` of them by default.
    A heading is the direction of a step, and the direction of a step that barely happened
    is mostly noise; including them pulls the mean towards the 0 of a random walk.

    .. warning::

       This descriptor is unusually sensitive to how the series was brought to its sampling
       rate, because a change of heading between consecutive samples is a different question
       at every sampling interval. In the standstill corpus a bare polyphase resample read
       0.15 here for the two 120 Hz editions --- which need a 5:12 conversion, where the
       anti-alias filter has least room --- against about 0.95 for the others, and the split
       was very nearly published as a difference between tracking systems. Through
       :func:`~micromotion.to_rate` every edition reads 0.949 to 0.964. Bring series to a
       common rate with that, and compare only series that share one.

    Args:
        xy: Positions, shape (N, 2). Non-finite rows are dropped.
        speed_percentile (float): Percentile of step speeds below which steps are excluded
            from the heading average. Defaults to 20.

    Returns:
        dict: ``persistence`` in [-1, 1] as described above, and ``straightness`` in [0, 1].
            Both are NaN when fewer than three finite samples remain.
    """
    p = np.asarray(xy, float)
    p = p[np.isfinite(p).all(axis=1)]
    if len(p) < 3:
        return {"persistence": float("nan"), "straightness": float("nan")}

    v = np.diff(p, axis=0)
    speed = np.hypot(v[:, 0], v[:, 1])
    path = float(np.sum(speed))
    straightness = float(np.hypot(*(p[-1] - p[0])) / path) if path > 0 else float("nan")

    if len(v) < 2:
        return {"persistence": float("nan"), "straightness": straightness}

    heading = np.arctan2(v[:, 1], v[:, 0])
    turn = np.diff(heading)
    turn = (turn + np.pi) % (2 * np.pi) - np.pi
    fast = speed >= np.percentile(speed, speed_percentile)
    keep = fast[:-1]
    persistence = float(np.mean(np.cos(turn[keep]))) if keep.any() else float("nan")
    return {"persistence": persistence, "straightness": straightness}


def path_length(xy, fs: float | None = None) -> dict:
    """Total distance travelled, and the rate at which it accumulated.

    Unfiltered and undifferentiated, so it is not a quantity of motion and is not comparable
    with one: sensor noise adds to path length monotonically, which means a noisier
    device reports a longer path for an identical movement. Reported because the balance
    literature uses it, and because it correlates with head quantity of motion at 0.61 in
    this corpus, which is worth knowing but is not an equivalence.
    """
    p = np.asarray(xy, float)
    p = p[np.isfinite(p).all(axis=1)]
    if len(p) < 2:
        return {"path": float("nan"), "path_rate": float("nan")}
    d = float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))
    return {"path": d, "path_rate": d / (len(p) / fs) if fs else float("nan")}


def dispersion_radius(xy, quantile: float = 0.95) -> float:
    """Radius containing a given proportion of the samples, about the mean position.

    A robust alternative to the ellipse area when the trace has excursions: a single lean
    that lasts two seconds changes the ellipse considerably and this hardly at all.
    """
    p = np.asarray(xy, float)
    p = p[np.isfinite(p).all(axis=1)][:, :2]
    if len(p) < 3:
        return float("nan")
    c = p - p.mean(axis=0)
    return float(np.quantile(np.linalg.norm(c, axis=1), quantile))


def shared_axis_projection(markers, reference, mask: str = "reference") -> dict:
    """Project several markers onto ONE axis, the reference marker's principal axis.

    This is the difference between asking "does this marker sway along a line" and asking
    "do these segments sway along the SAME line". :func:`micromotion.principal_axis_projection`
    answers the first, per marker, giving each its own axis -- and two markers swaying at right
    angles, each on its own axis, then correlate perfectly while sharing no direction of motion.
    Correlating segments only means something once they are on a common axis, which is what
    this builds.

    The axis is anatomical -- front-to-back for a standing person -- so it is taken from one
    named marker rather than from the pooled cloud, which would be dominated by whichever
    segment moved most. ``axis_deg`` carries the same warning as
    :func:`sway_geometry`'s: it is in the recording's own frame and is not comparable across
    recordings unless the laboratory convention and the direction the body faced are both known.

    Args:
        markers: Mapping of name -> horizontal coordinates, each ``(T, 2)``. Extra columns
            are ignored, so a full ``(T, 3)`` marker may be passed.
        reference: Key naming the marker whose axis every other is projected onto.
        mask (str, optional): Which samples count. ``"reference"`` (the default) keeps the
            frames where the reference marker is finite, and centres and projects every marker
            over exactly those -- the convention of the still-standing coordination analysis,
            kept as the default so its results reproduce. ``"own"`` gives each marker its own
            finite samples, which is more defensible per marker but makes the projections rest
            on different frames.

    Returns:
        dict: ``projection`` (name -> ``(T,)`` array, NaN where masked out), ``axis`` (the
        unit vector), ``axis_deg`` (its direction, axial, 0-180), ``mask`` (the boolean frame
        mask used when ``mask="reference"``, else None), and ``n_finite`` (name -> count of
        finite samples in that marker's projection).

    Raises:
        KeyError: If ``reference`` is not among the markers.
        ValueError: If ``mask`` is neither "reference" nor "own", or the reference marker has
            fewer than three usable frames.
    """
    if mask not in ("reference", "own"):
        raise ValueError(
            f"mask must be 'reference' or 'own', not {mask!r}. 'reference' keeps the frames "
            "where the reference marker is finite; 'own' gives each marker its own.")
    if reference not in markers:
        raise KeyError(
            f"reference marker {reference!r} is not among the markers "
            f"({sorted(markers)!r}). The axis has to come from a named marker, because it is "
            "anatomical rather than a property of the pooled cloud.")

    ref = np.asarray(markers[reference], float)[:, :2]
    ok = np.isfinite(ref).all(axis=1)
    if ok.sum() < 3:
        raise ValueError(
            f"reference marker {reference!r} has {int(ok.sum())} usable frames; "
            "need at least three to define an axis.")

    centred = ref[ok] - ref[ok].mean(axis=0)
    cov = np.cov(centred.T)
    _, vecs = np.linalg.eigh(cov)
    axis = vecs[:, -1]

    projection: dict = {}
    n_finite: dict = {}
    for name, xy in markers.items():
        p = np.asarray(xy, float)[:, :2]
        keep = ok if mask == "reference" else np.isfinite(p).all(axis=1)
        s = np.full(len(p), np.nan)
        if keep.any():
            good = keep & np.isfinite(p).all(axis=1)
            s[good] = (p[good] - p[good].mean(axis=0)) @ axis
        projection[name] = s
        n_finite[name] = int(np.isfinite(s).sum())

    return {
        "projection": projection,
        "axis": axis,
        "axis_deg": float(np.degrees(np.arctan2(axis[1], axis[0])) % 180.0),
        "mask": ok if mask == "reference" else None,
        "n_finite": n_finite,
    }


def segmental_coordination(markers, reference, ratios=None, mask: str = "reference",
                           min_finite: float = 0.8, reduce=None) -> dict:
    """Do a body's segments sway as one rigid link, or as several?

    Projects every marker onto the reference marker's axis with
    :func:`shared_axis_projection`, then reduces the result three ways: how strongly each pair
    of segments agrees, how many independent axes the set spans, and which of a named pair
    sways further. A body rocking at the ankles as a single inverted pendulum gives high
    correlations, an effective degrees-of-freedom near one, and a head-to-hip amplitude ratio
    above one -- the head is further from the ankle, so the same rotation carries it further.

    The effective degrees of freedom is :func:`micromotion.effective_dimensionality` called on
    the projections with ``rank=False``, not a second implementation of the participation
    ratio. Ranking is right for heavy-tailed descriptors and wrong here, where the columns are
    sway signals and their actual covariance is the quantity of interest.

    **The correlations are pairwise-complete and each carries its own n.** ``np.corrcoef``
    returns NaN if a single sample is missing, which drops whole sessions from a pair without
    saying so -- one or two per pair in the corpus this comes from, under a heading that
    reported one N for every row. Read ``n`` beside any correlation.

    No body-part taxonomy is built in: ``markers`` is whatever the caller names, and
    "inverted pendulum" is an interpretation of a ratio above one rather than something this
    computes. Group membership stays with the study.

    Args:
        markers: Mapping of name -> horizontal coordinates, each ``(T, 2)``.
        reference: Key naming the marker whose axis defines the shared direction.
        ratios (optional): Pairs ``(a, b)`` to report an amplitude ratio for, as
            ``std(a) / std(b)`` over their shared finite samples. Defaults to none.
        mask (str, optional): Passed to :func:`shared_axis_projection`. Defaults to
            ``"reference"``.
        reduce (optional): Which markers enter the dimensionality reduction, by name.
            Defaults to all of them. Pass an explicit list when ``markers`` carries a derived
            marker -- a midpoint of two others, say, wanted only for an amplitude ratio --
            because such a marker is a linear combination of segments already present and
            entering it as a ninth segment changes the effective degrees of freedom.
        min_finite (float, optional): Markers finite on a smaller fraction of the *usable*
            frames than this -- the reference-valid frames under ``mask="reference"``, the
            whole recording under ``mask="own"`` --
            are excluded from the effective-dimensionality reduction, which needs a complete
            matrix. They still appear in the pairwise correlations. Defaults to 0.8.

    Returns:
        dict: ``correlation`` and ``n`` (both keyed by the ``(a, b)`` pair, a and b sorted as
        given), ``pc1_fraction`` and ``effective_dof``, ``amplitude_ratio`` (keyed by the
        requested pairs), ``axis_deg``, ``n_finite`` per marker, ``used`` (the markers that
        entered the reduction) and ``n_excluded``.

    Raises:
        KeyError: If ``reference``, or either member of a requested ratio, is not a marker.
        ValueError: If ``mask`` is invalid, or fewer than two markers survive ``min_finite``.
    """
    proj = shared_axis_projection(markers, reference, mask=mask)
    sig = proj["projection"]
    names = list(sig)
    n_frames = len(sig[reference])

    correlation: dict = {}
    n_pairs: dict = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            both = np.isfinite(sig[a]) & np.isfinite(sig[b])
            n_pairs[(a, b)] = int(both.sum())
            if both.sum() < 3:
                correlation[(a, b)] = float("nan")
                continue
            x, y = sig[a][both], sig[b][both]
            if x.std() == 0 or y.std() == 0:
                correlation[(a, b)] = float("nan")
                continue
            correlation[(a, b)] = float(np.corrcoef(x, y)[0, 1])

    # The denominator is the frames that COULD have been used, not the length of the
    # recording: under mask="reference" a marker cannot be finite where the reference is not,
    # so dividing by the full length would penalise every marker for the reference's dropouts.
    usable = int(proj["mask"].sum()) if proj["mask"] is not None else n_frames
    candidates = list(names) if reduce is None else list(reduce)
    missing = [n for n in candidates if n not in sig]
    if missing:
        raise KeyError(f"reduce names markers that were not given: {missing!r}")
    used = [n for n in candidates
            if usable and proj["n_finite"][n] / usable >= min_finite]
    if len(used) < 2:
        raise ValueError(
            f"only {len(used)} marker(s) are finite on at least {min_finite:.0%} of frames, "
            "so there is nothing to reduce. Lower min_finite or check the markers.")

    matrix = np.column_stack([sig[n] for n in used])
    matrix = matrix[np.isfinite(matrix).all(axis=1)]
    dims = effective_dimensionality(matrix, rank=False)

    amplitude_ratio: dict = {}
    for a, b in (ratios or []):
        for key in (a, b):
            if key not in sig:
                raise KeyError(f"amplitude ratio asked for {key!r}, which is not a marker.")
        both = np.isfinite(sig[a]) & np.isfinite(sig[b])
        denom = sig[b][both].std() if both.sum() >= 2 else 0.0
        amplitude_ratio[(a, b)] = (float(sig[a][both].std() / denom) if denom > 0
                                   else float("nan"))

    return {
        "correlation": correlation,
        "n": n_pairs,
        "pc1_fraction": float(dims["variance_fraction"][0]),
        "effective_dof": float(dims["participation_ratio"]),
        "amplitude_ratio": amplitude_ratio,
        "axis_deg": proj["axis_deg"],
        "n_finite": proj["n_finite"],
        "used": used,
        "n_excluded": len(candidates) - len(used),
    }
