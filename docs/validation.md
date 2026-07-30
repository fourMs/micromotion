# Validating data

Motion data fails quietly. A dropped optical frame written as a coordinate triplet of zeros, a
timestamp column that steps backwards, a documented sampling rate that is not the real one — none
of these raises an error, and all of them produce a plausible number. The characteristic failure
is not a crash but a believable wrong answer, so the only defence is a check that runs every time.

`micromotion.validate` is that set of checks. Each one exists because the failure it catches has
occurred in real deposited datasets and gone unnoticed; the figures quoted below are measurements
from those cases, given so that you can judge whether a tolerance is reasonable rather than take
it on trust.

## Using it as a gate

```python
from micromotion import validate

findings = validate.validate_series(xyz, t=timestamps, documented_hz=200.0, where="A0001:P01")
validate.raise_on_error(findings)      # refuse to build on anything at "error"
```

`validate_series` runs whichever checks apply and returns a list of `Finding` objects, each with a
`check`, a `severity` of `"error"` or `"warning"`, a `message` and a `where`. Treat errors as
build-stopping and warnings as facts to record in a manifest beside the number they affect.

Set `expect_positions=False` for a sensor whose output may legitimately be zero on every axis at
once.

## The checks

### `zero_triplets` — gaps read as positions

Several optical systems, including Qualisys, write a dropped frame as `0.000 0.000 0.000`. In
laboratory coordinates that is a point on the floor, often more than a metre from where the marker
actually is, so a reader that takes it literally sees the marker leave and return.

A median barely notices. Anything that sums or integrates does: in one recording, 93 such frames
out of 118 698 gave a head marker a path length of 119.8 m where the true figure was 11.0 m — a
factor of eleven, from 0.08 % of the samples.

Exact zeros on all axes at once do not occur in real optical data, so the default tolerance is
zero.

### `finite_fraction` and `longest_finite_span` — a series emptied by its own filter

A gap running off the start or end of a series cannot be interpolated: there is nothing on the far
side to interpolate from. A band-pass then spreads the surviving `NaN` across the whole recording,
and the result is indistinguishable from an absent sensor unless something checks.

```python
start, n = validate.longest_finite_span(xyz)   # what to measure over instead
```

Measuring over the longest clean span, and recording how long it was, is usually better than
either dropping the recording or silently closing the gap — closing it makes the series claim a
duration it does not have.

### `timestamps` — a clock that is not a clock

Sorting a timestamp column into order is the tempting repair and the wrong one. It destroys the
evidence that the clock misbehaved while leaving samples in an order the sensor never produced.
One balance-board dataset carries 123 111 duplicate timestamps and 83 that step backwards; those
are a device fault to be recorded, not a sort key.

### `rate_agreement` — the documented rate against the measured one

Measure the rate, do not read it. Across one multi-device corpus, documented rates were wrong by
up to 4.4 per cent, one was out by a factor of 37, and one set of wearable accelerometers ran at
191.29–207.73 Hz against a nominal 200 — with the true rate a property of the *individual unit*,
so participants sharing a device shared a clock and everyone else did not.

Every frequency-domain measure scales with this, and nothing further downstream can detect it,
which is why it is an error rather than a warning. See [sampling rates](rates.md) for why
`measured_rate` counts samples over the elapsed span instead of inverting the median interval.

### `frame_count` — the 16-bit ceiling

C3D stores its frame count in a 16-bit field, so a conversion through C3D stops at 65 535 frames
and says nothing. At 200 Hz that is 327.7 s of a 360 s recording: complete-looking, with the last
thirty-two seconds gone. Treat a frame count of exactly 65 535 or 65 536 as a bug rather than a
coincidence.

### `held_samples` — a stored rate above the sampled rate

A sensor sampled below the rate it is written out at appears with each value repeated. The file
then claims a rate the data does not carry, and any spectrum computed at the stored rate is wrong
above the true Nyquist frequency. Some phone loggers write a row whenever *any* sensor updates,
which produces exactly this.

### `duplicate_files` — the same recording twice

Found by content rather than by name, because the usual cause is a rename that left the original
behind under a name no manifest mentions.

## Is it standstill throughout?

A recording of stationary behaviour should contain stationary behaviour and nothing else. In
practice recordings are trimmed by hand, or not at all, and what survives at the edges is
participants walking into position, settling, stepping onto a force plate, or reacting to being
told the recording has ended.

```python
speed = mm.speed_from_position(xyz, fs)
validate.edge_motion(speed, fs)          # which end, and by how much
validate.settling_time(speed, fs)        # (head, tail) seconds to trim
```

Both compare against the recording's own settled interior rather than an absolute threshold,
because the quantity spans two orders of magnitude across sensor types.

This is worth checking even when you believe the trimming was done. Across 1465 recordings in one
corpus, 784 were clean and the rest were not: two collections accounted for nearly all of it, and
one recording ended at 1608 mm/s against a settled 3.06 — someone walking away, inside a dataset
described as standstill.

`settling_time` answers the follow-on question rather than leaving it to a guess. One collection
trimmed a fixed twelve seconds to remove a synchronisation clap; measured, the settling ran to
twenty-five seconds at the median and thirty-five at the ninetieth percentile, and correcting it
changed a published null result into a significant effect.

!!! warning "A ratio test needs an absolute floor"

    On a near-motionless reference marker a ratio is meaningless — twice a settled level of
    0.1 mm/s is still nothing. If you act on `edge_motion` automatically, require an absolute speed
    as well as a ratio.

## Repair at source, or on read?

When a check fires on archival data you have two options, and the right one depends on what
depends on the file.

**Repair on read** if the recording backs a publication, or if deposited code reproduces published
figures from it. Rewriting such a file breaks the reproduction, and a gap encoding that is
*documented* is not a defect even when it is a hazard — it may be the convention that the
published analysis code expects.

**Repair at source** if the encoding is undocumented, nothing computes from the file yet, or the
stored values are wrong rather than merely awkward.

Either way, record what was found. A dataset that documents its own hazards is more useful than
one that has quietly removed them, because the second gives a reuser no way to tell whether a
number they compute matches the published one.
