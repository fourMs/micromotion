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
