"""Generate the figures used in the README and the documentation.

Every figure here is real output of the package, computed on synthetic signals whose
answer is known, in the same way the test suite builds its inputs. Run this script from
the repository root after changing anything the figures depend on::

    python docs/img/make_figures.py

Nothing in the package imports this file, and no figure is checked by a test. The point
is that a reader can rerun it and get the same picture.
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import micromotion as mm  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent

INK = "#1c1c1c"
MUTED = "#6b6b6b"
GRID = "#dcdcda"
BLUE = "#1f6fb2"
ORANGE = "#c85a1e"
GREY = "#b0b0ae"

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "savefig.dpi": 110,
    }
)


# The generator these figures are drawn from lives in the package now, as
# `micromotion.examples`, so that the documentation's own examples can be run by a
# reader with no data of their own. It was defined here first and copied nowhere; this
# import is what keeps it that way, since a figure drawn from a private copy of the
# signal would drift away from the one the quickstart shows.
standstill = mm.examples.standstill


def figure_qom() -> None:
    """Quantity of motion on the synthetic recording, and the same series in bins."""
    fs = 100.0
    xyz = standstill(fs=fs)
    r = mm.qom(xyz, fs, kind="position", unit="mm")
    bins = r.binned(5.0)
    t = np.arange(r.n_samples) / fs

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.2, 4.2), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    ax1.plot(t, r.speed, color=BLUE, linewidth=0.4, alpha=0.7)
    ax1.axhline(r.median_mm_s, color=ORANGE, linewidth=1.6, linestyle="--")
    ax1.axhline(r.mean_mm_s, color=MUTED, linewidth=1.2, linestyle=":")
    ax1.set_ylim(0, 12)
    ax1.annotate(
        f"median {r.median_mm_s:.2f} mm/s",
        xy=(0.5, 11.4), ha="left", va="top", color=ORANGE, fontsize=8,
    )
    ax1.annotate(
        f"mean {r.mean_mm_s:.2f} mm/s",
        xy=(0.5, 9.9), ha="left", va="top", color=MUTED, fontsize=8,
    )
    ax1.set_ylabel("speed (mm/s)")
    ax1.set_title("Band-limited speed of a synthetic head marker, 0.2–5 Hz", loc="left")
    ax1.grid(axis="y")
    ax1.set_axisbelow(True)

    ok = bins.edge == "ok"
    ax2.bar(
        bins.time_s[ok], bins.qom_mm_s[ok], width=4.2, align="edge",
        color=BLUE, label="ok",
    )
    ax2.bar(
        bins.time_s[~ok], bins.qom_mm_s[~ok], width=4.2, align="edge",
        color=GREY, label="filter transient",
    )
    ax2.set_ylabel("5 s bins (mm/s)")
    ax2.set_xlabel("time (s)")
    ax2.legend(loc="upper left", ncol=2)
    ax2.grid(axis="y")
    ax2.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(HERE / "qom-standstill.png", bbox_inches="tight")
    plt.close(fig)
    print(f"qom-standstill.png  mean {r.mean_mm_s:.3f}  median {r.median_mm_s:.3f} mm/s")


def figure_rolloff() -> None:
    """How much of a pure tone survives each band, against the analytic answer."""
    fs, amp, dur_s = 200.0, 3.0, 240.0
    t = np.arange(0, dur_s, 1 / fs)
    tones = [0.3, 0.5, 1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0]

    rows = {"micromotion": [], "wideband": []}
    for band in rows:
        for f in tones:
            x = amp * np.sin(2 * np.pi * f * t)
            r = mm.qom(x, fs, kind="position", unit="mm", band=band)
            trim = r.edge_samples
            got = float(np.mean(np.abs(r.speed[trim:-trim])))
            rows[band].append(got / (amp * 2 * np.pi * f * 2 / np.pi))

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.axhline(1.0, color=GRID, linewidth=1.0)
    ax.plot(tones, rows["micromotion"], color=BLUE, linewidth=2.0, marker="o",
            markersize=4, label="BAND, 0.2–5 Hz")
    ax.plot(tones, rows["wideband"], color=ORANGE, linewidth=2.0, marker="s",
            markersize=4, label="WIDEBAND, 0.2–10 Hz")
    ax.set_xlabel("tone frequency (Hz)")
    ax.set_ylabel("recovered / analytic speed")
    ax.set_ylim(0, 1.15)
    ax.set_title("A band edge is not a content boundary", loc="left")
    ax.legend(loc="lower left")
    ax.grid(axis="y")
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(HERE / "band-rolloff.png", bbox_inches="tight")
    plt.close(fig)
    for band, vals in rows.items():
        pairs = ", ".join(f"{f:g}:{v:.3f}" for f, v in zip(tones, vals))
        print(f"band-rolloff.png  {band}  {pairs}")


def figure_zero_triplets() -> None:
    """What 0.1 per cent of gap frames does to a path length, and to a median."""
    fs = 100.0
    xyz = standstill(fs=fs)
    holed = xyz.copy()
    rng = np.random.default_rng(11)
    # Dropped frames arrive in short runs rather than one at a time, which is what an
    # occluded marker looks like: nine bursts of four frames, 0.1 per cent of the file.
    for start in rng.choice(len(holed) - 10, size=9, replace=False):
        holed[start:start + 4] = 0.0

    clean_path = mm.path_length(xyz)["path"]
    holed_path = mm.path_length(holed)["path"]
    clean_med = mm.qom(xyz, fs, kind="position", unit="mm").median_mm_s
    holed_med = mm.qom(holed, fs, kind="position", unit="mm").median_mm_s

    fig, ax = plt.subplots(figsize=(6.0, 2.9))
    labels = ["path length\n(metres)", "median quantity of motion\n(mm/s)"]
    clean = [clean_path / 1000.0, clean_med]
    holed_v = [holed_path / 1000.0, holed_med]
    y = np.arange(len(labels))
    ax.barh(y + 0.19, clean, height=0.32, color=BLUE, label="gaps as NaN")
    ax.barh(y - 0.19, holed_v, height=0.32, color=ORANGE, label="gaps left as zero triplets")
    for k in range(len(labels)):
        ax.annotate(f"{clean[k]:.2f}", (clean[k], y[k] + 0.19), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8, color=MUTED)
        ax.annotate(f"{holed_v[k]:.2f}", (holed_v[k], y[k] - 0.19), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8, color=MUTED)
    ax.set_yticks(y, labels)
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xlim(1.5, 90)
    ax.set_xticks([2, 5, 10, 20, 50])
    ax.set_xticklabels(["2", "5", "10", "20", "50"])
    ax.set_xlabel("value (log scale)")
    ax.set_title("0.1 per cent of frames written as (0, 0, 0)", loc="left")
    ax.legend(loc="upper right")
    ax.grid(axis="x")
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(HERE / "zero-triplets.png", bbox_inches="tight")
    plt.close(fig)
    print(
        f"zero-triplets.png  path {clean_path / 1000:.2f} -> {holed_path / 1000:.2f} m; "
        f"median {clean_med:.2f} -> {holed_med:.2f} mm/s"
    )


def figure_one_measure() -> None:
    """The package's headline claim, drawn: one number from three devices.

    The same synthetic body motion is presented three ways -- as optical position at
    100 Hz, as the acceleration a body-worn sensor would report, and as position
    sampled at 50 Hz -- and reduced by the same band. If the shared abstraction really
    is the frequency band rather than the device, the three medians agree.

    The gyroscope-free acceleration route is the honest one to draw: it is derived from
    the same trajectory, so any disagreement here is the pipeline's and not the body's.
    """
    fs = 100.0
    xyz = standstill(fs=fs)

    optical = mm.qom(xyz, fs, kind="position", unit="mm")

    worn = mm.qom(mm.examples.worn_acceleration(fs=fs), fs,
                  kind="acceleration", unit="m/s^2")

    fs_low = 50.0
    slow = mm.qom(xyz[::2], fs_low, kind="position", unit="mm")

    names = ["optical position\n100 Hz", "worn accelerometer\n100 Hz",
             "optical position\n50 Hz"]
    vals = [optical.median_mm_s, worn.median_mm_s, slow.median_mm_s]

    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    bars = ax.bar(names, vals, width=0.55, color=[BLUE, ORANGE, BLUE])
    bars[2].set_alpha(0.55)
    for name, v in zip(names, vals):
        ax.annotate(f"{v:.2f}", (name, v), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=8, color=MUTED)
    ax.axhline(vals[0], color=GRID, linewidth=1.0, zorder=0)
    ax.set_ylabel("median quantity of motion (mm/s)")
    ax.set_ylim(0, max(vals) * 1.35)
    ax.set_title("One body, three devices, one number", loc="left")
    ax.grid(axis="y")
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(HERE / "one-measure.png", bbox_inches="tight")
    plt.close(fig)
    spread = max(vals) / min(vals)
    print("one-measure.png  " + "  ".join(f"{n.splitlines()[0]} {v:.3f}"
                                          for n, v in zip(names, vals))
          + f"  spread {spread:.3f}x")


if __name__ == "__main__":
    figure_qom()
    figure_rolloff()
    figure_zero_triplets()
    figure_one_measure()
