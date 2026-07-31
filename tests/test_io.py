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

    full = mm.read_phone(str(p))
    assert np.nanmax(full.data) > 9                      # clap present

    sym = mm.read_phone(str(p), trim_clap_s=35)
    asym = mm.read_phone(str(p), trim_start_s=35, trim_end_s=0)
    assert np.nanmax(sym.data) < 1 and np.nanmax(asym.data) < 1     # clap gone from both
    assert len(asym.data) > len(sym.data)                            # but the tail is kept
    assert asym.meta["trimmed_start_s"] == 35 and asym.meta["trimmed_end_s"] == 0

    with pytest.raises(ValueError, match="leaves nothing"):
        mm.read_phone(str(p), trim_start_s=60, trim_end_s=60)


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
    import tomllib
    import micromotion as mm

    root = pathlib.Path(mm.__file__).resolve().parents[2]
    pyproject = root / "pyproject.toml"
    if not pyproject.exists():          # installed without the source tree
        return
    declared = tomllib.loads(pyproject.read_text())["project"]["version"]
    assert mm.__version__ == declared, f"{mm.__version__} != {declared}"


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
