"""Obtain, verify and unpack the onsemi LTspice model libraries.

The onsemi model files are not redistributed in this repository: the LTspice
libraries are vendor-encrypted and carry the notice "Permission is granted to
use this file for simulations but not to reverse engineer its contents", which
does not grant redistribution. Each user downloads them from onsemi.com.

onsemi's download server rejects most non-browser clients, so this script first
tries a plain download and otherwise tells you exactly which file to save where.

Usage::

    python -m dpt.fetch_models          # verify + unpack (download if possible)
    python -m dpt.fetch_models --check  # verify only, non-zero exit if missing
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from dpt.ltspice import MODEL_ROOT


@dataclass(frozen=True)
class ModelZip:
    stem: str
    url: str
    product_page: str
    sha256: str  # of the file downloaded on 2026-10-28 (model version 2.3)


ZIPS = (
    ModelZip(
        stem="onsemi_sicmosfet_1200m3s_ltspice",
        url="https://www.onsemi.com/download/models/zip/onsemi_sicmosfet_1200m3s_ltspice.zip",
        product_page="https://www.onsemi.com/products/discrete-power-modules/silicon-carbide-sic/silicon-carbide-sic-mosfets/nth4l022n120m3s",
        sha256="c45a3f596c3addf72f4b29ae4be1c2a2992e704941806a6f49b9b20fd77d1196",
    ),
    ModelZip(
        stem="onsemi_sicmosfet_650m3s_ltspice",
        url="https://www.onsemi.com/download/models/zip/onsemi_sicmosfet_650m3s_ltspice.zip",
        product_page="https://www.onsemi.com/products/discrete-power-modules/silicon-carbide-sic/silicon-carbide-sic-mosfets/nth4l032n065m3s",
        sha256="e5ffd4043ff3aa5c845420e6078eb6eae0c005ef0c7ad5407cc1bdb9af518d09",
    ),
)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _try_download(z: ModelZip, dest: Path) -> bool:
    try:
        with urllib.request.urlopen(z.url, timeout=60) as resp:
            data = resp.read()
    except (urllib.error.URLError, TimeoutError):
        return False
    if not data.startswith(b"PK"):
        return False
    dest.write_bytes(data)
    return True


def ensure(z: ModelZip, root: Path, download: bool) -> bool:
    """Make sure ``z`` is present, verified and unpacked. Returns success."""
    root.mkdir(parents=True, exist_ok=True)
    zpath = root / f"{z.stem}.zip"
    if not zpath.is_file() and download and _try_download(z, zpath):
        print(f"downloaded {zpath.name}")
    if not zpath.is_file():
        print(
            f"MISSING {zpath.name}\n"
            f"  Open {z.product_page}\n"
            f"  -> Technical Documentation -> Simulation Models -> '{z.stem}'\n"
            f"  and save the ZIP as {zpath}"
        )
        return False
    digest = sha256_of(zpath)
    if digest != z.sha256:
        print(
            f"WARNING {zpath.name}: SHA256 {digest[:12]}... differs from the version used "
            f"for the committed results ({z.sha256[:12]}...). onsemi may have updated the "
            "model; results can differ slightly from the README."
        )
    out = root / z.stem
    if not out.is_dir():
        with zipfile.ZipFile(zpath) as zf:
            for member in zf.namelist():
                target = (out / member).resolve()
                if not str(target).startswith(str(out.resolve())):
                    raise RuntimeError(f"unsafe path in {zpath.name}: {member}")
            zf.extractall(out)
        print(f"unpacked {zpath.name} -> {out}")
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="verify only, never download")
    args = ap.parse_args(argv)
    ok = all([ensure(z, MODEL_ROOT, download=not args.check) for z in ZIPS])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
