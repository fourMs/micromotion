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
| `read_phone` | Physics Toolbox phone log |
| `read_equivital` | Equivital physiology CSV |
| `read_balance_board` | Wii balance board, headerless and irregular |
| `read_fnirs` | Artinis Brite wide export, IMU channels only |

The Qualisys reader detects its own variant by trying to parse the eleventh line as numbers,
so exports differing in whether they carry a column-name row, or `Frame`/`Time` columns, all
read correctly.

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
