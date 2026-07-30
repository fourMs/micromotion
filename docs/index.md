# micromotion

Analysis of human micromotion in motion time series: optical marker data, body-worn
accelerometers, and force-plate centre of pressure.

The measure the package exists for is **quantity of motion** — the average speed of a body
part, band-limited to 0.2–10 Hz, in millimetres per second. It applies equally to all three
sensor families because the shared abstraction is the frequency band, not the instrument.

## Why it exists

Across the research repository this was extracted from there were 159 analysis scripts.
Fifty-eight defined their own band-pass filter and thirty-seven computed quantity of motion.
The project's central measure existed in dozens of copies that did not all agree, and the
disagreements were invisible: two scripts returned 3.77 and 4.20 mm/s for the same series
purely because their filter chains differed, and neither said so.

That is the argument for a library. Not tidiness — reproducibility of the central quantity.

## What it will not let you do

Most of the design is about refusing comparisons that look reasonable and are not.

- Asking for the legacy optical band on accelerometer data **raises**, because gravity is a
  DC term and an accelerometer cannot be integrated without a lower edge.
- Upsampling **raises**, because it invents structure that scale-sensitive methods then treat
  as real.
- Gap sentinels become `NaN` on read, because a marker "at the origin" and a centre of
  pressure at the exact middle of a board are plausible-looking values that are not
  measurements.
- Partial and filter-contaminated bins are **flagged**, not silently included.

## Where to start

| If you want to | Read |
|---|---|
| compute a quantity of motion | [Getting started](quickstart.md) |
| know which filter convention to use | [The two bands](conventions.md) |
| compare across datasets | [Sampling rates](rates.md) |
| load a file | [Reading files](formats.md) |
| check data before trusting it | [Validating data](validation.md) |
| combine with MGT or ambiscape | [Working with other packages](interop.md) |
| look up a function | [API reference](api.md) |
