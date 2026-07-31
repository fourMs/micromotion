"""Readers for the layouts this corpus actually uses.

Each returns a :class:`~micromotion.record.MotionRecord`. Gap sentinels become NaN, units
are recorded rather than assumed, and the sampling rate is measured wherever the file
carries a timebase.

:func:`read` dispatches on content, not on the extension, because the extension lies: the
balance-board dumps are headerless and space-delimited whatever they are called, and the
Qualisys family puts three different header shapes behind one name.
"""

from __future__ import annotations

import io as _io
import os
import re

import numpy as np
import pandas as pd

from .record import MotionRecord
from .resample import measured_rate

_METADATA_KEY = re.compile(r"^[A-Z][A-Z0-9_]*$")

QUALISYS_KEYS = (
    "NO_OF_FRAMES",
    "NO_OF_CAMERAS",
    "NO_OF_MARKERS",
    "FREQUENCY",
    "NO_OF_ANALOG",
    "ANALOG_FREQUENCY",
    "DESCRIPTION",
    "TIME_STAMP",
    "DATA_INCLUDED",
    "MARKER_NAMES",
)

Y_UP_COLLECTIONS: tuple[str, ...] = ()
"""Datasets whose vertical axis is Y rather than Z. Empty by default.

A system's frame is a property of how it was configured for a session, not of the system:
the same OptiTrack rig can produce Y-up files for one study and Z-up for another. Where a
dataset is known to be Y-up, name it here rather than compensating in downstream analysis.

Prefer rotating the data at source (``X -> X, Y -> -Z_old, Z -> Y_old`` — a rotation, not an
axis swap). If you do, empty this in the same change: a reader that still claims Y for
rotated files hands every caller a horizontal axis as the vertical one, silently.
"""


def _decode(path: str) -> list[str]:
    return _io.open(path, encoding="latin-1").readlines()


def read_qualisys(path: str, drop_gaps: bool = True) -> MotionRecord:
    """Qualisys or Qualisys-style TSV, in all three header shapes found in the corpus.

    The shapes differ in what follows the ten ``KEY<TAB>value`` metadata lines:

    * a ``<marker> X`` column-name row, then data (2012, 2015, 2017, 2018, 2019, HpSp);
    * nothing, data begins immediately (2022, Bishop 2020);
    * a ``Frame``/``Time`` column-name row, then data (MocapNoiseFloor, Solberg 2016).

    The shape is detected by trying to parse the eleventh line as numbers, so a file that
    was exported differently still reads correctly.
    """
    lines = _decode(path)
    meta, n_header = {}, 0
    for i, ln in enumerate(lines[:16]):
        parts = ln.rstrip("\n").split("\t")
        key = parts[0].strip()
        if key in QUALISYS_KEYS:
            meta[key] = [p for p in parts[1:] if p.strip() != ""]
            n_header = i + 1
        elif _METADATA_KEY.match(key):
            # An unknown metadata line, not the end of the header. QTM adds fields between
            # versions -- exports since 2.0.0 open with FILE_VERSION -- and a parser that
            # stops at the first key it does not recognise reads those files as having no
            # header at all. Recorded but unused; the shape test below still finds the data.
            meta.setdefault(key, [p for p in parts[1:] if p.strip() != ""])
            n_header = i + 1
        else:
            break

    if "MARKER_NAMES" not in meta:
        raise ValueError(f"{path} has no MARKER_NAMES line; not a Qualisys export")
    markers = [m.strip() for m in meta["MARKER_NAMES"]]
    fs_header = float(meta["FREQUENCY"][0])

    def is_numeric(line: str) -> bool:
        try:
            float(line.split("\t")[0])
            return True
        except (ValueError, IndexError):
            return False

    has_frame_time = False
    if is_numeric(lines[n_header]):
        skip = n_header                       # shape 2: no column-name row
    else:
        skip = n_header + 1                   # shape 1 or 3
        first = lines[n_header].split("\t")[0].strip()
        has_frame_time = first.lower() == "frame"

    df = pd.read_csv(path, sep="\t", skiprows=skip, header=None, encoding="latin-1")
    if df.iloc[:, -1].isna().all():
        df = df.iloc[:, :-1]                  # the trailing tab's phantom column
    arr = df.to_numpy(float)

    t = None
    if has_frame_time:
        t = arr[:, 1].copy()
        arr = arr[:, 2:]

    n_expected = 3 * len(markers)
    if arr.shape[1] != n_expected:
        # Trust the columns, not the count. One HpSp file declares 22 markers and carries 7.
        n_actual = arr.shape[1] // 3
        markers = markers[:n_actual] if n_actual <= len(markers) else [
            *markers, *[f"unnamed{i}" for i in range(len(markers), n_actual)]
        ]
        arr = arr[:, : 3 * len(markers)]

    if drop_gaps:
        # A zero triplet is the Qualisys gap code, not a marker at the origin. In HpSp that
        # is 878 948 lines; read as data it would report a marker teleporting to (0,0,0).
        block = arr.reshape(len(arr), -1, 3)
        gap = np.all(block == 0.0, axis=2)
        block[gap] = np.nan
        arr = block.reshape(len(arr), -1)

    channels = [f"{m} {ax}" for m in markers for ax in "XYZ"]
    vertical = "Y" if any(c in path for c in Y_UP_COLLECTIONS) else "Z"
    fs = measured_rate(t) if t is not None and len(t) > 1 else fs_header

    return MotionRecord(
        data=arr,
        fs=fs,
        channels=channels,
        kind="position",
        unit="mm",
        vertical=vertical,
        t=t,
        source=path,
        meta={"header": meta, "nominal_fs": fs_header, "n_markers": len(markers)},
    )


def read_sverm(path: str) -> MotionRecord:
    """Sverm plain-header export: ``Time`` then ``s<n>_head_{x,y,z}``.

    A separate reader because it shares nothing with the Qualisys export but the units. It
    silently returned zero series to a corpus analysis until one was written for it.
    """
    df = pd.read_csv(path, sep="\t")
    if "Time" not in df.columns:
        raise ValueError(f"{path} has no Time column; not a Sverm export")
    t = df["Time"].to_numpy(float)
    cols = [c for c in df.columns if c != "Time"]
    return MotionRecord(
        data=df[cols].to_numpy(float),
        fs=measured_rate(t),
        channels=[c.replace("_", " ") for c in cols],
        kind="position",
        unit="mm",
        t=t,
        source=path,
        meta={"subjects": sorted({c.split("_")[0] for c in cols})},
    )


def read_ax3(path: str) -> MotionRecord:
    """Axivity AX3 export with a ``ts,x,y,z`` header.

    The rate is measured from the timestamps and differs meaningfully between files: it is
    a property of the physical logger, spanning 191.3-207.7 Hz across the 2024 units, and
    it can differ between recording sites in the same direction as the effect a study
    record's headline claim reports.
    """
    df = pd.read_csv(path, sep="\t")
    ts = pd.to_datetime(df["ts"])
    t = (ts - ts.iloc[0]).dt.total_seconds().to_numpy()
    return MotionRecord(
        data=df[["x", "y", "z"]].to_numpy(float),
        fs=measured_rate(t),
        channels=["x", "y", "z"],
        kind="acceleration",
        unit="g",
        t=t,
        t0=ts.iloc[0],
        source=path,
        meta={"device_slot": (re.match(r"[A-Z]+(\d+)", os.path.basename(path)[:-4]) or [None, None])[1]},
    )


def _read_physics_toolbox_raw(path: str) -> pd.DataFrame:
    """A Physics Toolbox Sensor Suite export exactly as the app writes it.

    The app's own CSV is not a plain CSV, and every one of these has cost time:

    - **A blank first line.** The header is line 2.
    - **Semicolon delimiter with a decimal comma.** Read with the default comma delimiter and
      the whole file becomes one string column per row.
    - **Unicode minus U+2212 (``−``), not ASCII hyphen.** Negative numbers silently become
      NaN under ``pd.to_numeric``, so a phone that was tilted one way reads as missing data
      and one tilted the other way reads fine.
    - **Infinity as ``∞``** in the ``Gain`` column.
    - **A trailing empty field** after the last named column, giving a phantom ``Unnamed`` column.
    - **GPS columns** (``Latitude``, ``Longitude``, ``Speed``, sometimes ``Altitude``) are present
      and are identifying. Drop them before depositing anything.

    Column availability varies by handset: the Galaxy A52s writes no ``p`` (pressure) column, so
    concatenating logs by position rather than by name misaligns every channel after ``wz``.
    """
    df = pd.read_csv(path, sep=";", skiprows=1, dtype=str, encoding="utf-8")
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df.columns = [c.strip() for c in df.columns]
    out = {}
    for c in df.columns:
        s = (df[c].astype(str)
             .str.replace("−", "-", regex=False)      # Unicode minus
             .str.replace("∞", "inf", regex=False)    # infinity, in Gain
             .str.replace(",", ".", regex=False))          # decimal comma
        out[c] = pd.to_numeric(s, errors="coerce")
    return pd.DataFrame(out)


def read_phone(path: str, trim_clap_s: float = 0.0, *,
               trim_start_s: float | None = None,
               trim_end_s: float | None = None) -> MotionRecord:
    """Physics Toolbox phone log, raw app export or cleaned tab-separated form.

    The variant is detected from the first two lines, so both the app's own semicolon/
    decimal-comma CSV and the cleaned TSV used by the deposited pipeline read correctly.
    See ``_read_physics_toolbox_raw`` for what the raw format does that plain CSV readers
    get wrong.

    ``ax``/``ay``/``az`` are linear acceleration in m/s^2, gravity removed by sensor fusion.
    They are not in g, whatever an older version of the data dictionary said; reading them as
    g inflates every quantity of motion by 9.80665. Only
    ``gF*`` is in g, and it is total g-force *including* gravity — its magnitude sits at 1.0
    on a phone at rest, so substituting it for ``a*`` inflates band-limited motion by a factor
    of thousands rather than failing.

    **The rate is neither constant nor the nominal one.** Physics Toolbox delivers whatever the
    Android sensor stack gives it, so a log requested at 100 Hz arrives between roughly 100 and
    170 Hz with millisecond-scale jitter, and differs between handsets recording the same event.
    ``fs`` here is the measured mean rate over the span. **Long dropouts are common** — logging
    stops when the app is backgrounded or the screen sleeps, and resumes silently, leaving gaps
    of tens of seconds inside a file that otherwise looks continuous. Check ``meta["gaps"]``
    before treating a file as one recording; resample onto a uniform grid before filtering.

    ``trim_clap_s`` drops that many seconds from **each** end. ``trim_start_s`` and
    ``trim_end_s`` override it per end, so an opening clap can be removed without discarding
    good data at the close — pass ``trim_start_s=35, trim_end_s=0``.

    **Trim before you plot or transform.** A recording that opens with a synchronisation clap
    carries a transient that is a timing marker, not movement, and it can be two orders of
    magnitude above the standstill it precedes — in one recording 10.29 m/s² against a body
    maximum of 0.155. Left in, it sets the y-axis of any plot, dominates any spectrum, and gives
    a peak detector a transient that is not a breath.

    The settling that follows is slower and also worth removing. Measure it rather than guessing:
    both posture and heart rate can still be moving well beyond ten seconds.
    """
    with open(path, encoding="utf-8", errors="replace") as fh:
        head = [fh.readline() for _ in range(2)]
    raw = ";" in head[1] and "\t" not in head[1]
    df = _read_physics_toolbox_raw(path) if raw else pd.read_csv(path, sep="\t")
    t = df["time"].to_numpy(float)
    head_s = trim_clap_s if trim_start_s is None else trim_start_s
    tail_s = trim_clap_s if trim_end_s is None else trim_end_s
    m = np.ones(len(t), bool)
    if head_s or tail_s:
        m = (t >= t[0] + head_s) & (t <= t[-1] - tail_s)
        if not m.any():
            raise ValueError(f"trimming {head_s} s from the start and {tail_s} s from the end "
                             f"leaves nothing of a {t[-1] - t[0]:.1f} s recording")
    cols = [c for c in ("ax", "ay", "az") if c in df.columns]
    data = df[cols].to_numpy(float)[m]

    # Missing samples are written as exact zeros, not NaN. Every file opens with a few.
    zero_rows = np.all(data == 0.0, axis=1)
    data[zero_rows] = np.nan

    # Dropouts are silent and common: report them rather than letting a caller average across
    # a 40 s hole as though it were one continuous recording.
    tm = t[m]
    dt = np.diff(tm)
    gap_idx = np.where(dt > 1.0)[0]
    gaps = [(float(tm[i]), float(tm[i + 1])) for i in gap_idx]
    bounds = [0, *(gap_idx + 1), len(tm)]
    segments = [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if b - a >= 2]
    longest = max((tm[b - 1] - tm[a] for a, b in segments), default=0.0)

    # A span-averaged rate is meaningless once there are dropouts: one 108 s hole in a file
    # sampled at 120 Hz returns 3.8 Hz, which would then be handed to a 0.2-10 Hz filter as
    # though it were the truth. Report the rate of the longest continuous run instead, and keep
    # the span average in meta for anyone who wants it.
    fs_span = measured_rate(t)
    if segments:
        a, b = max(segments, key=lambda ab: tm[ab[1] - 1] - tm[ab[0]])
        fs = measured_rate(tm[a:b])
    else:
        fs = fs_span

    return MotionRecord(
        data=data,
        fs=fs,
        channels=cols,
        kind="acceleration",
        unit="m/s^2",
        t=tm,
        source=path,
        meta={"trimmed_s": trim_clap_s, "trimmed_start_s": head_s, "trimmed_end_s": tail_s, "n_zero_rows": int(zero_rows.sum()),
              "raw_export": raw, "fs_span_average": float(fs_span),
              "gaps": gaps,
              "segments": segments,
              "longest_continuous_s": float(longest),
              "extra": {c: df[c].to_numpy(float)[m] for c in df.columns
                        if c in ("wx", "wy", "wz", "gFx", "gFy", "gFz")}},
    )


def read_equivital(path: str) -> MotionRecord:
    """Equivital physiology CSV: accelerometer, ECG, respiration or RR.

    Four of the five files per participant are delimited by comma-and-space and the fifth by
    a bare comma, so the separator is handled rather than assumed. The accelerometer is in
    raw counts, calibrated to g by the median vector magnitude, since a person standing
    still averages one g.

    The rate is measured from the timestamps by span over count. Taking the median interval
    instead returns exactly 250 Hz for a recording that runs at 256, because the timestamps
    are rounded to whole milliseconds; that error was live in the deposited quantity of
    motion.
    """
    df = pd.read_csv(path, skipinitialspace=True)
    tcol = df.columns[0]
    ts = pd.to_datetime(df[tcol], format="mixed", utc=True)
    t = (ts - ts.iloc[0]).dt.total_seconds().to_numpy()
    cols = list(df.columns[1:])

    # copy(): pandas returns a read-only view under copy-on-write, and the rail masking
    # below writes in place. Without this, reading a respiration file raises.
    data = df[cols].to_numpy(float).copy()
    unit, kind = "counts", "acceleration"
    if any("Accelerometer" in c for c in cols):
        mag = np.median(np.linalg.norm(data, axis=1))
        data = data / mag                      # counts -> g
        unit = "g"
    elif "Breathing" in cols:
        kind, unit = "respiration", "adc"
        # 0 and 1023 are the rails of a 10-bit converter, not waveform.
        data[(data <= 0) | (data >= 1023)] = np.nan

    return MotionRecord(
        data=data,
        fs=measured_rate(t),
        channels=cols,
        kind=kind,
        unit=unit,
        t=t,
        t0=ts.iloc[0],
        source=path,
    )


def read_balance_board(path: str) -> MotionRecord:
    """Wii balance board dump: headerless, space-delimited, irregularly sampled.

    Eight columns: timestamp in milliseconds, four corner load cells, total load, and
    centre of pressure in x and y, all normalised to 0-1.

    Samples with no load carry a centre of pressure of exactly (0.5, 0.5), the midpoint of
    the board, which is a plausible-looking value for nobody standing on it. Those are
    returned as NaN. They are 13.8 per cent of the HpSp balance data.
    """
    arr = np.loadtxt(path)
    if arr.ndim != 2 or arr.shape[1] < 8:
        raise ValueError(f"{path} is not an 8-column balance dump")
    t = arr[:, 0] / 1000.0
    data = arr[:, 6:8].copy()
    load = arr[:, 5]
    data[(load <= 0) | ((data[:, 0] == 0.5) & (data[:, 1] == 0.5))] = np.nan
    return MotionRecord(
        data=data,
        fs=measured_rate(t) if len(t) > 1 else float("nan"),
        channels=["cop_x", "cop_y"],
        kind="position",
        unit="normalised",
        t=t,
        source=path,
        meta={"load": load, "cells": arr[:, 1:5], "irregular": True},
    )


BOARD_MM = (433.0, 238.0)
"""Wii Balance Board sensing area, width by depth, in millimetres.

Multiplying the normalised centre of pressure by these gives millimetres, which is what
makes balance data comparable with the optical collections.
"""


def read_fnirs(path: str) -> MotionRecord:
    """Artinis Brite wide export: 96 haemoglobin channels plus two IMUs.

    Only the accelerometer and gyroscope are returned as ``data``; the haemoglobin channels
    are kept in ``meta`` because they are not motion. The accelerometer is in milli-g.

    Column 0 is a sample index, not time, so the rate comes from the header. The absolute
    start can be decoded from the ``TimeStampHi``/``Lo`` pair if it is needed.
    """
    df = pd.read_csv(path, sep="\t")
    acc = [c for c in df.columns if c.startswith("ACC_")]
    gyr = [c for c in df.columns if c.startswith("GYR_")]
    if not acc:
        raise ValueError(f"{path} has no ACC_ columns; not an fNIRS export")
    dev = acc[0].split("_")[-2]
    acc = [c for c in acc if dev in c][:3]
    gyr = [c for c in gyr if dev in c][:3]

    legend = os.path.join(os.path.dirname(path), "channel_legend.txt")
    fs = 75.0
    if os.path.exists(legend):
        m = re.search(r"sample rate:\s*([\d.]+)", _io.open(legend, encoding="latin-1").read())
        if m:
            fs = float(m.group(1))

    return MotionRecord(
        data=df[acc].to_numpy(float) / 1000.0,     # milli-g -> g
        fs=fs,
        channels=acc,
        kind="acceleration",
        unit="g",
        source=path,
        meta={
            "device": dev,
            "gyro_deg_s": df[gyr].to_numpy(float) if gyr else None,
            "haemoglobin": [c for c in df.columns if c.endswith(("O2Hb", "HHb"))],
        },
    )


_READERS = {
    "qualisys": read_qualisys,
    "sverm": read_sverm,
    "ax3": read_ax3,
    "phone": read_phone,
    "equivital": read_equivital,
    "balance": read_balance_board,
    "fnirs": read_fnirs,
}


def sniff(path: str) -> str:
    """Identify a file's layout from its first lines."""
    head = "".join(_decode(path)[:3])
    first = head.split("\n")[0]
    if first.startswith("NO_OF_FRAMES"):
        return "qualisys"
    if first.startswith("Time\t") and "_head_" in first:
        return "sverm"
    if first.startswith("ts\tx\ty\tz"):
        return "ax3"
    if first.startswith("time\t") and "\tax\t" in first:
        return "phone"
    if first.startswith("sample_index\t"):
        return "fnirs"
    if first.startswith("DateTime"):
        return "equivital"
    if re.fullmatch(r"[\d.\s eE+-]+", first) and len(first.split()) == 8:
        return "balance"
    raise ValueError(f"cannot identify the layout of {path}")


def read(path: str, **kw) -> MotionRecord:
    """Read any corpus motion file, dispatching on content."""
    return _READERS[sniff(path)](path, **kw)
