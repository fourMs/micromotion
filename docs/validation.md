# Validating data

Every check in `micromotion.validate` exists because the failure it catches happened, went
unnoticed, and produced a believable wrong number. None of them raised anything at the time.
That is the point: the characteristic failure of real motion data is not a crash but a plausible
answer, and the only defence is a check that runs every time.

## The gate

```python
from micromotion import validate

findings = validate.validate_series(xyz, t=timestamps, documented_hz=200.0, where="A0001:P01")
validate.raise_on_error(findings)      # refuse to build on anything at "error"
```

`validate_series` runs whichever checks apply and returns a list of `Finding` objects, each with
a `check`, a `severity` of `"error"` or `"warning"`, a `message` and a `where`. Errors should stop
a build. Warnings are facts to record in a manifest beside the number they affect.

Set `expect_positions=False` for a sensor whose output may legitimately be zero on every axis at
once.

## The checks

### `zero_triplets` — gaps read as positions

Qualisys writes a dropped frame as `0.000 0.000 0.000`. That is a point on the laboratory floor
about a metre and a half below a standing head, so a reader that takes it literally sees the head
leave and return. A median barely notices; a path length does. On one recording, 93 such frames
out of 118 698 gave the head a travel of 119.8 m where the true figure is 11.0 m.

Exact zeros on all axes do not occur in real optical data, so the default tolerance is zero.

### `finite_fraction` and `longest_finite_span` — a series emptied by its own filter

A gap running off the start or end of a series cannot be interpolated — there is nothing on the
far side — and a band-pass then spreads the surviving NaN across the whole recording. The result
is indistinguishable from an absent sensor unless something looks.

```python
start, n = validate.longest_finite_span(xyz)   # what to measure over instead
```

### `timestamps` — a clock that is not a clock

Sorting a timestamp column into order is the tempting repair and the wrong one: it destroys the
evidence that the clock misbehaved while leaving samples in an order the sensor never produced.
One balance-board collection carries 123 111 duplicate timestamps and 83 that step backwards.

### `rate_agreement` — the documented rate against the measured one

Measure the rate, do not read it. Documented rates in the source corpus were wrong by up to
4.4 per cent, one by a factor of 37, and one edition's accelerometers run at 191.29–207.73 Hz
against a nominal 200 — with the true rate a property of the individual device, so three
participants sharing a unit share a clock and everyone else does not.

### `frame_count` — the 16-bit ceiling

C3D stores its frame count in sixteen bits, so a conversion through it stops at 65 535 and says
nothing. Seven sessions were nearly deposited that way: 327.7 s of a 360 s recording,
complete-looking, with the last thirty-two seconds gone.

### `held_samples` — a stored rate above the sampled rate

A sensor sampled below the rate it is stored at is written out with each value repeated, and the
file then claims a rate the data does not carry. Any spectrum computed at the stored rate is
wrong above the true Nyquist.

### `duplicate_files` — the same recording twice

Found by content, not by name. One record shipped the same recording under two names, one of them
in no manifest, left over from a rename.

## Is it standstill throughout?

Deposited standstill data should contain standstill and nothing else. In practice exports are
trimmed by hand, or not at all, and what survives at the edges is people walking into position,
settling, or reacting to being told the recording has ended.

```python
speed = mm.speed_from_position(xyz, fs)
validate.edge_motion(speed, fs)                # which end, and by how much
validate.settling_time(speed, fs)              # (head, tail) seconds to trim
```

Both compare against the recording's own settled interior rather than an absolute threshold,
because the quantity spans two orders of magnitude across sensors. Run over the source corpus,
784 of 1464 recordings were clean; two collections carried nearly all of the rest, and one
recording ended at 1608 mm/s against a settled 3.06 — someone walking.

`settling_time` answers the follow-on question rather than leaving it to a guess. One collection
trims a fixed twelve seconds for a synchronisation clap; the measurement says twenty-five at the
median and thirty-five at the ninetieth percentile.

!!! warning "A ratio test needs an absolute floor"

    On a near-motionless reference marker a ratio is meaningless. Two flags in the source corpus
    are 2× on a settled level of 0.1 mm/s, which is noise. If you act on `edge_motion`
    automatically, require an absolute speed as well as a ratio.

## What it found on its first run

Over 172 files and 1178 marker series in nine deposited records: fourteen series carrying
zero-triplet gaps, one of them 41 per cent of the recording, in a collection that had not been
repaired. Whether to repair such a record at source or on read is a judgement — see
[the two bands](conventions.md) for the general principle that archival data stays as deposited
and derived layers repair on read.
