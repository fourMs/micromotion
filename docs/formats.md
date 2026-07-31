# Reading files

```python
rec = mm.read(path)          # dispatches on content
mm.sniff(path)               # just tell me what this is
```

`read` inspects the file's first lines rather than its extension. In the corpus this was
built for, the extension lies: the balance-board dumps are named `.tsv` and are
space-delimited and headerless.

## What you get back

Every reader returns a [`MotionRecord`](api.md#micromotion.record.MotionRecord) with the same
fields whatever the source:

| Field | Meaning |
|---|---|
| `data` | `(n_samples, n_channels)` float, gaps as `NaN` |
| `fs` | sampling rate, **measured** where the file has a timebase |
| `channels` | column names |
| `kind` | `"position"` or `"acceleration"` — decides differentiate or integrate |
| `unit` | `"mm"`, `"m"`, `"g"`, `"m/s^2"`, `"counts"` |
| `vertical` | which axis is up; not always `Z` |
| `t`, `t0` | timestamps and absolute start, where they exist |
| `meta` | whatever else the header carried |

## Supported layouts

| Reader | Format |
|---|---|
| `read_qualisys` | Qualisys and Qualisys-style TSV, all three header shapes |
| `read_sverm` | plain-header optical export, `Time` plus per-subject columns |
| `read_ax3` | Axivity AX3, `ts,x,y,z` |
| `read_phone` | Physics Toolbox phone log, raw app export or cleaned TSV |
| `read_equivital` | Equivital physiology CSV |
| `read_balance_board` | Wii balance board, headerless and irregular |
| `read_fnirs` | Artinis Brite wide export, IMU channels only |

The Qualisys reader detects its own variant by trying to parse the eleventh line as numbers,
so exports differing in whether they carry a column-name row, or `Frame`/`Time` columns, all
read correctly.

## Coordinate frames: which axis is up, and how to rotate

`MotionRecord.vertical` exists because **the vertical axis is not always Z**. Getting this wrong
does not raise anything — it silently swaps a vertical measurement for a horizontal one, and every
magnitude still looks plausible.

### What the systems do

| System | Default frame | Units in export |
|---|---|---|
| Qualisys (QTM) | **Z-up**, right-handed | millimetres |
| OptiTrack (Motive) | **Y-up**, right-handed — but configurable, and configured differently between sessions | metres |
| C3D generally | whatever the source wrote; read `POINT:UNITS` | declared in the file |

**Do not infer the frame from the system.** The same OptiTrack rig, configured differently at two
sessions, will give you Y-up files from one and Z-up from the other. The only reliable checks are
the file's own metadata (`POINT:UNITS` for C3D) and the data itself — a standing head marker sits at
roughly 1.5–1.9 in whatever unit is in force, on whichever axis is vertical.

```python
import numpy as np
med = [np.nanmedian(a[:, k::3]) for k in range(3)]   # a is (frames, 3*markers)
vertical = int(np.argmax(med))                        # the axis holding head height
```

### Rotating Y-up to Z-up

    X_zup = X_yup
    Y_zup = -Z_yup
    Z_zup =  Y_yup

**Use this rather than swapping Y and Z.** A swap is a *reflection*: it mirrors the horizontal plane,
so every sway direction reverses sign and every rotational statistic inverts, while all magnitudes —
speed, range, quantity of motion — stay exactly as they were and give no hint that anything moved.
The mapping above is a rotation, which preserves handedness: X × (−Z) = −(X × Z) = Y, the new Z.

Gaps survive it. Missing samples are the zero triplet and negating zero leaves zero.

### The check that a rotation was correct

Quantity of motion is the magnitude of a 3-D speed, so it is **invariant under rotation** and must
come out unchanged to floating-point precision. If it moves at all, the transform is wrong. The
vertical median should land on the new axis at the value it had on the old one.

That test is cheap and worth running on any converted corpus: it distinguishes a rotation from a
reflection, from an axis swap, and from a unit error, all of which otherwise produce data that looks
entirely reasonable.

### Fix the data, then hunt for the compensations

If you rotate a dataset at source, **every script that special-cased its old frame becomes wrong**.
A known wrinkle tends to be worked around in many places, and those workarounds are invisible until
the wrinkle is gone — at which point each one silently inverts the correction it was making. Grep
for the compensation before changing the data, and empty it in the same commit.

## Physics Toolbox Sensor Suite

A widely used phone sensor-logging app. `read_phone` accepts
both the app's own export and the cleaned tab-separated form, detecting which from the first two
lines, so nothing needs converting by hand any more.

### What the app's CSV actually is

Not a plain CSV. Each of these has cost time at least once:

| Quirk | What goes wrong without it |
|---|---|
| Blank first line; header on line 2 | header read as data |
| Semicolon delimiter | one string column per row |
| Decimal comma (`0,0523`) | every value parses as text |
| **Unicode minus U+2212 (`−`)** | negatives become NaN — a tilt one way reads as missing, the other way reads fine |
| `∞` in the `Gain` column | column stays text and poisons arithmetic |
| Trailing `;` on each row | a phantom `Unnamed` column |
| `Latitude`, `Longitude`, `Speed` present | identifying location data ships with the motion |

The Unicode minus is the dangerous one, because it fails *asymmetrically*: the sign of the value
decides whether it survives, so a file can look nine-tenths intact and be systematically biased.

### Columns, and which acceleration to use

`ax`/`ay`/`az` are **linear acceleration in m/s²**, gravity removed by sensor fusion. `gFx`/`gFy`/`gFz`
are **total g-force in g, including gravity** — magnitude 1.0 on a phone at rest.

Use `a*`. Substituting `gF*` does not fail, it inflates band-limited motion roughly 4000-fold,
because a slowly rotating gravity vector has ample 0.2–5 Hz content that no band-pass will remove.
Reading `a*` as if it were g inflates every quantity of motion by 9.80665 — a clean constant
factor, so rankings and correlations survive it and nothing looks wrong.

Column availability varies by handset: the Galaxy A52s writes no `p` (pressure) column, so joining
logs by column position rather than by name misaligns every channel after `wz`.

### The sampling rate is not what you asked for

Physics Toolbox passes on whatever the Android sensor stack delivers. A log requested at 100 Hz
arrives somewhere between about 100 and 170 Hz, with millisecond-scale jitter, and **two handsets
recording the same event will differ** — in one three-phone session the measured rates were 105,
123 and 169 Hz. Always resample onto a uniform grid before filtering.

### Dropouts are silent, and they are the real hazard

Logging stops when the app is backgrounded or the screen sleeps, then resumes without any marker.
A file that looks like one continuous recording can contain a hole of tens of seconds. In the same
three-phone session every one of the six files had at least one gap, the worst being **108 s inside
a 111 s file** — leaving 2.6 s of actual data.

**The timestamps are the app's own awake-time, not wall-clock.** When Android suspends the app the
clock stops with it. A ten-minute recording whose screen slept produced a file spanning 160 s — the
missing 440 s is not present as a gap, it is absent from the timeline. Two phones dozed at different
moments therefore have time bases that drift apart non-linearly, so **no cross-correlation can
recover a fixed offset between two such files, because there is no fixed offset to find.** Align
multi-device recordings on a physical event that appears in the signal itself — a sharp tap on a
rigid stack holding all the phones — not on their clocks, and not on an acoustic clap, which leaves
no accelerometer transient at all.

`read_phone` therefore reports:

| `meta` key | Meaning |
|---|---|
| `gaps` | `(start, end)` in seconds for every break over 1 s |
| `segments` | index ranges of the continuous runs |
| `longest_continuous_s` | the longest unbroken stretch — usually the only analysable part |
| `fs_span_average` | rate over the whole span, for reference |

**`fs` is measured over the longest continuous run, not the whole span.** Averaging across a hole
gives a number that is not a sampling rate at all: the 111 s file above yields 3.8 Hz by span
average against a true 120 Hz, and handing that to a 0.2–5 Hz filter produces confident nonsense
rather than an error.

Check `longest_continuous_s` before trusting a file's duration. A recording is only as long as its
longest segment.

## Sentinels become NaN

Each of these is a plausible-looking value that is not a measurement, and each has cost an
analysis at least once:

| Source | Sentinel |
|---|---|
| optical | zero coordinate triplet — a "marker at the origin" |
| phone | exact-zero rows at the start of a recording |
| respiration belt | samples pinned at 0 or 1023, the converter's rails |
| balance board | centre of pressure at exactly (0.5, 0.5) under no load |

## Reading markers

```python
rec.markers                  # names
rec.marker("HF")             # (n_samples, 3) for one marker
```

Always by name. Six files in one collection break the documented 22-marker order, and a
positional read mis-assigns every marker in all of them.

## Gaps

```python
mm.gap_report(rec.data, rec.fs)
clean = mm.interpolate_gaps(rec.data, max_gap=200)
```

`interpolate_gaps` bridges short runs and leaves long ones as `NaN`. Bridging a dropped frame
is reconstruction; bridging a 469-second hole is invention. A single missing fraction hides
the distinction that matters — one per cent scattered evenly is a usable recording, one per
cent in a single block is two recordings.
