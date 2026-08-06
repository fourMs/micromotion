"""Reader behaviour that has cost this corpus real errors."""

import numpy as np
import pandas as pd
import pytest

import micromotion as mm

def test_read_phone_asymmetric_trim(tmp_path):
    """trim_start_s / trim_end_s override trim_clap_s per end.

    The symmetric-only version forced a choice between keeping an opening synchronisation clap
    and discarding good data at the close. On StillStanding365 day 221 the clap reaches
    10.29 m/s2 against a body maximum of 0.155, so leaving it in is not an option and losing
    35 s of standstill to remove it was the only alternative.
    """
    fs, dur = 100.0, 100.0
    t = np.arange(0, dur, 1 / fs)
    a = np.full((len(t), 3), 0.01)
    a[int(2 * fs)] = 10.0                      # the clap, at t = 2 s
    df = pd.DataFrame({"time": t, "ax": a[:, 0], "ay": a[:, 1], "az": a[:, 2]})
    p = tmp_path / "clap.tsv"
    df.to_csv(p, sep="\t", index=False)

    # channel="fused": this fixture carries only ax/ay/az, and since 0.15.0 the default is the
    # accelerometer. The test is about asymmetric trimming, not about which channel is read.
    full = mm.read_phone(str(p), channel="fused")
    assert np.nanmax(full.data) > 9                      # clap present

    sym = mm.read_phone(str(p), channel="fused", trim_clap_s=35)
    asym = mm.read_phone(str(p), channel="fused", trim_start_s=35, trim_end_s=0)
    assert np.nanmax(sym.data) < 1 and np.nanmax(asym.data) < 1     # clap gone from both
    assert len(asym.data) > len(sym.data)                            # but the tail is kept
    assert asym.meta["trimmed_start_s"] == 35 and asym.meta["trimmed_end_s"] == 0

    with pytest.raises(ValueError, match="leaves nothing"):
        mm.read_phone(str(p), channel="fused", trim_start_s=60, trim_end_s=60)


def test_no_edition_is_y_up_any_more():
    """Standstill2019 was rotated to Z-up at source, so the compensation must be off.

    A reader that still claimed Y for 2019 would hand every caller a horizontal axis as the
    vertical one, silently — the mirror image of the bug the rotation fixed.
    """
    from micromotion.io import Y_UP_COLLECTIONS
    assert Y_UP_COLLECTIONS == ()


def test_version_matches_pyproject():
    """__version__ and pyproject.toml must agree.

    They did not at 0.9.0: the release bumped pyproject and left the hardcoded string in
    __init__.py at 0.8.3, so an installed copy reported the previous version. A figure or a
    report that records "produced with micromotion X" would then record the wrong X.
    """
    import pathlib
    import re
    import micromotion as mm

    root = pathlib.Path(mm.__file__).resolve().parents[2]
    pyproject = root / "pyproject.toml"
    if not pyproject.exists():          # installed without the source tree
        return
    text = pyproject.read_text()
    try:
        import tomllib                  # stdlib from 3.11; this package supports 3.10
        declared = tomllib.loads(text)["project"]["version"]
    except ModuleNotFoundError:
        # A regex rather than a skip. The point of this test is to catch the two files drifting
        # apart, and skipping it on the oldest supported Python is exactly where a release would
        # slip through unchecked.
        m = re.search(r'(?m)^version\s*=\s*"([^"]+)"', text)
        assert m, "no version field in pyproject.toml"
        declared = m.group(1)
    assert mm.__version__ == declared, f"{mm.__version__} != {declared}"

    # CITATION.cff is the third place the version is written, and it drifted to 0.7.0 while the
    # other two reached 0.12.3 -- five minor versions, unnoticed, because only the first two were
    # tested. It is the file GitHub's "Cite this repository" and Zenodo read, so a stale version
    # here does not misreport a figure, it misattributes the software. Regex rather than a YAML
    # parser: pyyaml is not a dependency and this is one scalar on one line.
    citation = root / "CITATION.cff"
    if citation.exists():
        m = re.search(r'(?m)^version:\s*"?([^"\s]+)"?\s*$', citation.read_text())
        assert m, "no version field in CITATION.cff"
        assert m.group(1) == declared, f"CITATION.cff {m.group(1)} != {declared}"


def test_equivital_labels_each_signal_type(tmp_path):
    """ECG and inter-beat intervals must not be labelled as acceleration.

    The reader defaulted anything it did not recognise to kind="acceleration", unit="counts", so
    a file of heartbeat intervals in milliseconds described itself as motion. `qom` happened to
    refuse on the unit, but that is the unit guard doing the kind label's job -- anything
    inspecting `rec.kind` to decide what a record was would have been told wrong.
    """
    import micromotion as mm

    cases = {
        "accelerometer.csv": ("DateTime, Vert Accelerometer, Lat Accelerometer, Long Accelerometer",
                              "2025-10-09 13:45:00.003+00:00, -881, -172, 27", "acceleration"),
        "ecg.csv": ("DateTime, Lead 1, Lead 2",
                    "2025-10-09 13:45:00.003+00:00, 0.681318, 0.54945", "ecg"),
        "respiration.csv": ("DateTime, Breathing",
                            "2025-10-09 13:45:00.003+00:00, 2048", "respiration"),
        "rr.csv": ("DateTime, Interbeat Interval (MS)",
                   "2025-10-09 13:45:00.058+00:00, 691", "interbeat_interval"),
    }
    for name, (header, row, expected_kind) in cases.items():
        p = tmp_path / name
        rest = row.split(",", 1)[1]
        rows = [f"2025-10-09 13:45:0{i}.003+00:00,{rest}" for i in range(1, 4)]
        p.write_text("\n".join([header, *rows]) + "\n")
        rec = mm.read_equivital(str(p))
        assert rec.kind == expected_kind, f"{name}: {rec.kind} != {expected_kind}"
        assert rec.unit != "counts" or expected_kind == "acceleration"


def test_physics_toolbox_header_on_line_1_or_line_2(tmp_path):
    """The blank first line is a property of the app build, not a constant.

    The Pro build writes the header on line 1 and the older build writes a blank line first.
    Assuming the blank line consumes the header, promotes the first data row to column names and
    loses a sample -- so both layouts must parse identically.
    """
    body = ("time;gFx;gFy;gFz;Gain\n"
            "0,10;0,1178;−0,5612;0,8295;−∞\n"
            "0,20;0,1180;−0,5610;0,8290;−∞\n")
    with_blank = tmp_path / "blank.csv"
    with_blank.write_text("\n" + body, encoding="utf-8")
    without = tmp_path / "noblank.csv"
    without.write_text(body, encoding="utf-8")

    a = mm.io._read_physics_toolbox_raw(str(with_blank))
    b = mm.io._read_physics_toolbox_raw(str(without))
    for d in (a, b):
        assert list(d.columns) == ["time", "gFx", "gFy", "gFz", "Gain"]
        assert len(d) == 2
        assert d["gFy"].iloc[0] == pytest.approx(-0.5612)   # Unicode minus survived
    assert a.equals(b)


def _phone_tsv(tmp_path):
    """A file whose two accelerations differ, and whose channels advance at different rates."""
    import numpy as np
    import pandas as pd
    n = 600
    t = np.arange(n) / 100.0
    # gF* updates every 2nd row (50 Hz), ax/ay/az every 5th (20 Hz): a zero-order hold, which is
    # what Physics Toolbox writes when several sensors share one file.
    g = np.repeat(1.0 + 0.01 * np.sin(2 * np.pi * 1.0 * t[::2]), 2)[:n]
    a = np.repeat(0.05 * np.sin(2 * np.pi * 1.0 * t[::5]), 5)[:n]
    df = pd.DataFrame({"time": t,
                       "gFx": g, "gFy": np.zeros(n), "gFz": np.zeros(n),
                       "ax": a, "ay": np.zeros(n), "az": np.zeros(n),
                       "wx": np.zeros(n), "wy": np.zeros(n), "wz": np.zeros(n)})
    p = tmp_path / "phone.tsv"
    df.to_csv(p, sep="\t", index=False)
    return str(p)


def test_read_phone_defaults_to_the_accelerometer(tmp_path):
    """The default changed in 0.15.0 and the whole point is that it is gF*, in m/s^2."""
    import numpy as np
    p = _phone_tsv(tmp_path)
    rec = mm.read_phone(p)
    assert rec.meta["channel"] == "accel"
    assert rec.channels == ["gFx", "gFy", "gFz"]
    assert rec.unit == "m/s^2"
    # gF* is in g in the file and must come back converted, so its magnitude sits near 9.8.
    assert 9.0 < float(np.nanmean(np.abs(rec.data[:, 0]))) < 10.5


def test_read_phone_fused_is_opt_in_and_unconverted(tmp_path):
    import numpy as np
    p = _phone_tsv(tmp_path)
    rec = mm.read_phone(p, channel="fused")
    assert rec.channels == ["ax", "ay", "az"]
    # ax is already m/s^2 and must NOT be multiplied by g a second time.
    assert float(np.nanmax(np.abs(rec.data[:, 0]))) < 0.2


def test_read_phone_reports_each_channels_own_rate(tmp_path):
    """The row rate is 100 Hz and belongs to no sensor in the file."""
    p = _phone_tsv(tmp_path)
    rates = mm.read_phone(p).meta["channel_rates"]
    assert 45 < rates["accel"] < 55, rates
    assert 15 < rates["fused"] < 25, rates


def test_read_phone_rejects_an_unknown_channel(tmp_path):
    import pytest
    with pytest.raises(ValueError, match="accel"):
        mm.read_phone(_phone_tsv(tmp_path), channel="raw")


def test_channel_rate_survives_bursts_and_repeats():
    """A burst-written channel: the median interval lies, the change count does not."""
    import numpy as np
    # 10 updates a second, each written as two rows 1 ms apart
    t = np.sort(np.concatenate([np.arange(0, 10, 0.1), np.arange(0, 10, 0.1) + 0.001]))
    x = np.repeat(np.arange(100, dtype=float), 2)
    assert 9.0 < mm.channel_rate(t, x) < 11.0
    assert 1.0 / np.median(np.diff(t)) > 100        # the estimator this replaces


def test_channel_resolution_catches_a_signal_inside_one_step():
    """The Delsys case: a quantisation step larger than the amplitude being measured.

    Those accelerometers step by 0.0395 m/s2 while the head acceleration being measured has a
    median of 0.033, so the whole signal sits inside one step and every correlation came back at
    about 0.03 -- indistinguishable from a real null. The test asserts the two halves that matter:
    the step is recovered, and `ratio` falls below 1 exactly when the signal cannot be resolved.
    """
    rng = np.random.default_rng(0)
    step, need = 0.0395, 0.033
    coarse = np.round(rng.normal(0, 1, 20_000) / step) * step
    r = mm.channel_resolution(coarse, need=need)
    assert abs(r["step"] - step) < 1e-9
    assert r["ratio"] < 1.0                      # the signal is inside one step
    assert r["levels"] < 500                     # and the channel holds few distinct values

    fine = rng.normal(0, 1, 20_000)
    f = mm.channel_resolution(fine, need=need)
    assert f["step"] < step / 100
    assert f["ratio"] > 10.0
    assert f["levels"] == 20_000

    # It must FAIL to report trouble when there is none: a fine channel with the same span and the
    # same need must not be flagged. Without this the assertions above pass on any input at all.
    assert f["ratio"] > r["ratio"] * 100


def test_channel_resolution_handles_a_block_and_a_constant():
    rng = np.random.default_rng(1)
    block = np.column_stack([np.round(rng.normal(0, 1, 5_000) / 0.05) * 0.05,
                             rng.normal(0, 1, 5_000)])
    out = mm.channel_resolution(block)
    assert set(out) == {"col0", "col1"}
    assert out["col0"]["step"] > out["col1"]["step"] * 100
    assert mm.channel_resolution(np.zeros(10))["levels"] == 1
