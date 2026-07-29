"""The common structure every reader returns.

The corpus holds sixteen distinct file layouts across fourteen collections, and each one has
already cost an analysis at least once: an axis convention that differs in a single edition,
a rate that three documents disagree about, a gap code that looks like a valid measurement.
A reader's job is to absorb all of that and hand back the same object regardless.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MotionRecord:
    """One recording, in a form the rest of the package can use without special cases.

    Attributes
    ----------
    data
        (n_samples, n_channels) float array. Gaps are NaN, never a sentinel.
    fs
        Sampling rate in Hz, measured from the file where a timebase exists and taken from
        the header only where it does not.
    channels
        Column names, one per column of ``data``.
    kind
        ``"position"`` or ``"acceleration"``: what the numbers are, which decides whether
        quantity of motion differentiates or integrates.
    unit
        ``"mm"``, ``"m"``, ``"g"``, ``"m/s^2"`` or ``"counts"``.
    vertical
        Which axis letter is up. Not always Z: the 2019 championship is Y-up, alone in the
        corpus.
    t
        Timestamps in seconds where the file carries them, otherwise None.
    t0
        Absolute start time where the file carries one, otherwise None.
    source
        Path the record was read from.
    meta
        Anything else the header held, unmodified.
    """

    data: np.ndarray = field(repr=False)
    fs: float
    channels: list[str]
    kind: str = "position"
    unit: str = "mm"
    vertical: str = "Z"
    t: np.ndarray | None = field(default=None, repr=False)
    t0: object | None = None
    source: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.data = np.asarray(self.data, float)
        if self.data.ndim == 1:
            self.data = self.data[:, None]
        if len(self.channels) != self.data.shape[1]:
            raise ValueError(
                f"{len(self.channels)} channel names for {self.data.shape[1]} columns "
                f"in {self.source}"
            )

    @property
    def n_samples(self) -> int:
        return self.data.shape[0]

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.fs

    @property
    def markers(self) -> list[str]:
        """Marker names, for records whose channels are ``<marker> X/Y/Z`` triplets."""
        seen, out = set(), []
        for c in self.channels:
            name = c.rsplit(" ", 1)[0]
            if name not in seen:
                seen.add(name)
                out.append(name)
        return out

    def marker(self, name: str) -> np.ndarray:
        """The (n_samples, 3) block for one marker, by name.

        Reading marker columns by name rather than by position is not a nicety. Six HpSp
        files break the documented 22-marker order, and the README's positional quick-start
        mis-assigned markers in every one of them until it was rewritten.
        """
        idx = [i for i, c in enumerate(self.channels) if c.rsplit(" ", 1)[0] == name]
        if not idx:
            raise KeyError(f"no marker {name!r} in {self.source}; have {self.markers}")
        return self.data[:, idx]

    def missing_fraction(self) -> float:
        """Proportion of the array that is NaN."""
        return float(np.isnan(self.data).mean())

    def qom(self, **kw):
        """Quantity of motion for this record, with kind and unit already filled in.

        Passing a whole record computes across every channel at once, which is rarely what
        you want for a multi-marker file; select a marker first.
        """
        from .qom import qom as _qom

        kw.setdefault("kind", self.kind)
        kw.setdefault("unit", self.unit)
        return _qom(self.data, self.fs, **kw)
