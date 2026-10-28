"""Binary .raw reader checked on synthetic files in both simulator dialects."""

from __future__ import annotations

import numpy as np
import pytest

from dpt.rawread import RawFileError, parse_raw


def _header(names: list[str], n_points: int, flags: str) -> str:
    lines = [
        "Title: * synthetic",
        "Date: today",
        "Plotname: Transient Analysis",
        f"Flags: {flags}",
        f"No. Variables: {len(names)}",
        f"No. Points: {n_points}",
        "Variables:",
    ]
    lines += [f"\t{i}\t{n}\t{'time' if i == 0 else 'voltage'}" for i, n in enumerate(names)]
    lines.append("Binary:")
    return "\n".join(lines) + "\n"


def _ltspice_blob(t: np.ndarray, cols: list[np.ndarray], double: bool) -> bytes:
    names = ["time"] + [f"V(n{i})" for i in range(len(cols))]
    flags = "real forward double" if double else "real forward"
    head = _header(names, len(t), flags).encode("utf-16-le")
    other = "<f8" if double else "<f4"
    dtype = np.dtype([("t", "<f8")] + [(f"c{i}", other) for i in range(len(cols))])
    rec = np.zeros(len(t), dtype=dtype)
    rec["t"] = t
    for i, c in enumerate(cols):
        rec[f"c{i}"] = c
    return head + rec.tobytes()


def _ngspice_blob(t: np.ndarray, cols: list[np.ndarray]) -> bytes:
    names = ["time"] + [f"v(n{i})" for i in range(len(cols))]
    head = _header(names, len(t), "real").encode("ascii")
    data = np.column_stack([t, *cols]).astype("<f8")
    return head + data.tobytes()


def test_ltspice_float32_payload_and_negative_time() -> None:
    t = np.linspace(0, 1e-6, 50)
    t_stored = t.copy()
    t_stored[10] = -t_stored[10]  # LTspice step-marker convention
    a, b = np.sin(t * 1e7), np.cos(t * 1e7) * 800
    raw = parse_raw(_ltspice_blob(t_stored, [a, b], double=False))
    assert raw.variables == ["time", "V(n0)", "V(n1)"]
    np.testing.assert_allclose(raw.time, t)
    np.testing.assert_allclose(raw["v(n1)"], b, rtol=1e-6)  # case-insensitive lookup
    assert raw["V(n0)"].dtype == np.float64


def test_ltspice_double_flag() -> None:
    t = np.linspace(0, 1e-6, 20)
    a = 1.0 + 1e-12 * np.arange(20)  # needs float64 to survive
    raw = parse_raw(_ltspice_blob(t, [a], double=True))
    np.testing.assert_array_equal(raw["V(n0)"], a)


def test_ngspice_ascii_header() -> None:
    t = np.linspace(0, 1e-6, 30)
    a = t * 3.0
    raw = parse_raw(_ngspice_blob(t, [a]))
    np.testing.assert_array_equal(raw["v(n0)"], a)
    assert "v(n0)" in raw and "v(nx)" not in raw


def test_truncated_file_keeps_complete_points() -> None:
    t = np.linspace(0, 1e-6, 10)
    blob = _ngspice_blob(t, [t])
    raw = parse_raw(blob[:-5])
    assert len(raw.time) == 9


def test_missing_binary_marker() -> None:
    with pytest.raises(RawFileError):
        parse_raw(b"Title: x\nNo. Variables: 1\n")
