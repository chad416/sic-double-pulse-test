"""Reader for SPICE binary ``.raw`` files (LTspice and ngspice dialects).

Only the subset needed here is supported: a single real-valued transient plot.

Format notes
------------
* LTspice writes the ASCII header as UTF-16LE; ngspice writes plain ASCII.
* The header ends with a ``Binary:`` line, followed by the data block, stored
  point by point (all variables of point 0, then point 1, ...).
* ngspice stores every value as float64.
* LTspice stores the time axis as float64 and all other variables as float32,
  unless the ``Flags`` line contains ``double``, in which case all are float64.
  With waveform compression enabled LTspice may store negative time values as
  markers; the absolute value is the real time. We always take ``abs(time)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


class RawFileError(ValueError):
    """Raised when a ``.raw`` file cannot be parsed."""


@dataclass
class RawFile:
    """Parsed transient result: variable name -> sample array."""

    title: str
    plotname: str
    flags: list[str]
    variables: list[str]
    data: dict[str, FloatArray] = field(repr=False)

    @property
    def time(self) -> FloatArray:
        return self.data[self.variables[0]]

    def __getitem__(self, name: str) -> FloatArray:
        key = _find_key(self.data, name)
        return self.data[key]

    def __contains__(self, name: str) -> bool:
        try:
            _find_key(self.data, name)
        except KeyError:
            return False
        return True


def _find_key(data: dict[str, FloatArray], name: str) -> str:
    """Case-insensitive lookup (SPICE node names are case-insensitive)."""
    if name in data:
        return name
    lowered = name.lower()
    for key in data:
        if key.lower() == lowered:
            return key
    raise KeyError(f"variable {name!r} not in raw file (have: {sorted(data)})")


def _split_header(blob: bytes) -> tuple[str, int]:
    """Return the decoded header text and the byte offset of the data block."""
    utf16 = len(blob) > 1 and blob[1] == 0
    if utf16:
        marker = "Binary:\n".encode("utf-16-le")
        idx = blob.find(marker)
        if idx < 0:
            raise RawFileError("no 'Binary:' marker found (ASCII raw files are not supported)")
        return blob[:idx].decode("utf-16-le"), idx + len(marker)
    marker_b = b"Binary:\n"
    idx = blob.find(marker_b)
    if idx < 0:
        raise RawFileError("no 'Binary:' marker found (ASCII raw files are not supported)")
    return blob[:idx].decode("ascii", errors="replace"), idx + len(marker_b)


def parse_raw(blob: bytes) -> RawFile:
    """Parse the bytes of a binary ``.raw`` file."""
    header, offset = _split_header(blob)
    lines = header.replace("\r\n", "\n").split("\n")

    meta: dict[str, str] = {}
    variables: list[str] = []
    in_vars = False
    for line in lines:
        if in_vars:
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit():
                variables.append(parts[1])
                continue
            in_vars = False
        if line.startswith("Variables:"):
            in_vars = True
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip().lower()] = value.strip()

    try:
        n_vars = int(meta["no. variables"])
        n_points = int(meta["no. points"])
    except (KeyError, ValueError) as exc:
        raise RawFileError("header lacks 'No. Variables' / 'No. Points'") from exc
    if len(variables) != n_vars:
        raise RawFileError(f"expected {n_vars} variables, parsed {len(variables)}")

    flags = meta.get("flags", "").lower().split()
    if "complex" in flags:
        raise RawFileError("complex (AC) plots are not supported")

    is_ltspice = blob[1:2] == b"\x00"
    all_double = (not is_ltspice) or ("double" in flags)
    if all_double:
        dtype = np.dtype([(f"v{i}", "<f8") for i in range(n_vars)])
    else:
        dtype = np.dtype([("v0", "<f8")] + [(f"v{i}", "<f4") for i in range(1, n_vars)])

    payload = blob[offset:]
    expected = dtype.itemsize * n_points
    if len(payload) < expected:
        # A simulation that aborted can leave a truncated file; keep complete points only.
        n_points = len(payload) // dtype.itemsize
        if n_points == 0:
            raise RawFileError("raw file has no complete data points")
    records = np.frombuffer(payload, dtype=dtype, count=n_points)

    data: dict[str, FloatArray] = {}
    for i, name in enumerate(variables):
        col = records[f"v{i}"].astype(np.float64)
        if i == 0:
            col = np.abs(col)
        data[name] = col
    return RawFile(
        title=meta.get("title", ""),
        plotname=meta.get("plotname", ""),
        flags=flags,
        variables=variables,
        data=data,
    )


def read_raw(path: str | Path) -> RawFile:
    """Read and parse a binary ``.raw`` file from disk."""
    return parse_raw(Path(path).read_bytes())
