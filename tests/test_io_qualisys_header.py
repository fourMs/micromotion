"""Qualisys header shapes, including fields added by later QTM versions.

The parser used to stop at the first key it did not recognise. QTM 2.0.0 exports open with a
`FILE_VERSION` line, so every file from it read as having no header at all and raised "not a
Qualisys export" -- on files that are unambiguously Qualisys exports. QTM will add more fields;
these pin the behaviour that unknown metadata keys are skipped rather than treated as the end
of the header.
"""

import numpy as np
import pytest

import micromotion as mm

BODY = "\n".join("\t".join(f"{v:.3f}" for v in (i, i * 0.005, 1.0 + i, 2.0, 3.0, 4.0, 5.0, 6.0))
                 for i in range(20))

CLASSIC = (
    "NO_OF_FRAMES\t20\nNO_OF_CAMERAS\t12\nNO_OF_MARKERS\t2\nFREQUENCY\t200\n"
    "NO_OF_ANALOG\t0\nANALOG_FREQUENCY\t0\nDESCRIPTION\t--\nTIME_STAMP\t2018-04-24\n"
    "DATA_INCLUDED\t3D\nMARKER_NAMES\tHF\tHL\n"
    "HF X\tHF Y\tHF Z\tHL X\tHL Y\tHL Z\n"
)

MODERN = "FILE_VERSION\t2.0.0\n" + CLASSIC


def _write(tmp_path, header, body=BODY, name="take.tsv"):
    p = tmp_path / name
    # two leading columns (frame, time) then the marker triplets
    rows = []
    for i in range(20):
        vals = [i, i * 0.005, 1.0 + i, 2.0, 3.0, 4.0, 5.0, 6.0]
        rows.append("\t".join(f"{v:.3f}" for v in vals))
    p.write_text(header + "\n".join(rows) + "\n")
    return str(p)


def test_reads_the_classic_header(tmp_path):
    r = mm.read_qualisys(_write(tmp_path, CLASSIC))
    assert r.fs == 200.0
    assert r.markers == ["HF", "HL"]


def test_reads_a_header_with_file_version(tmp_path):
    """QTM 2.0.0 prepends FILE_VERSION; this used to raise 'not a Qualisys export'."""
    r = mm.read_qualisys(_write(tmp_path, MODERN))
    assert r.fs == 200.0
    assert r.markers == ["HF", "HL"]


def test_both_header_shapes_give_identical_data(tmp_path):
    a = mm.read_qualisys(_write(tmp_path, CLASSIC, name="a.tsv"))
    b = mm.read_qualisys(_write(tmp_path, MODERN, name="b.tsv"))
    assert np.array_equal(np.asarray(a.data), np.asarray(b.data))


def test_tolerates_several_unknown_metadata_keys(tmp_path):
    header = ("FILE_VERSION\t2.0.0\nSOME_FUTURE_KEY\tvalue\nANOTHER_ONE\t7\n") + CLASSIC
    r = mm.read_qualisys(_write(tmp_path, header))
    assert r.markers == ["HF", "HL"]


def test_a_column_name_row_still_ends_the_header(tmp_path):
    """The discriminator must not swallow the column-name row: 'HF X' is not a metadata key."""
    r = mm.read_qualisys(_write(tmp_path, MODERN))
    assert np.asarray(r.data).shape[0] == 20


def test_still_rejects_a_file_that_is_not_qualisys(tmp_path):
    p = tmp_path / "nope.tsv"
    p.write_text("a\tb\tc\n1\t2\t3\n")
    with pytest.raises(ValueError, match="not a Qualisys export"):
        mm.read_qualisys(str(p))
