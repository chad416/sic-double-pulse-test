"""Run LTspice in batch mode and load the result as DPT waveforms.

Results are cached in ``build/sim/<config-key>/`` so a sweep can be interrupted
and resumed, and plots can be regenerated without re-simulating.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dpt.extract import Waveforms
from dpt.netlist import DPTConfig, render
from dpt.rawread import RawFile, read_raw

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = PROJECT_ROOT / "third_party" / "onsemi"
CACHE_ROOT = PROJECT_ROOT / "build" / "sim"


class SimulationError(RuntimeError):
    """LTspice did not produce a complete transient result."""


def find_ltspice() -> Path:
    """Locate the LTspice executable (env var ``LTSPICE_EXE`` wins)."""
    env = os.environ.get("LTSPICE_EXE")
    candidates = [Path(env)] if env else []
    local = os.environ.get("LOCALAPPDATA")
    if local:
        candidates.append(Path(local) / "Programs" / "ADI" / "LTspice" / "LTspice.exe")
    candidates += [
        Path(r"C:\Program Files\ADI\LTspice\LTspice.exe"),
        Path(r"C:\Program Files\LTC\LTspiceXVII\XVIIx64.exe"),
    ]
    which = shutil.which("LTspice") or shutil.which("LTspice.exe")
    if which:
        candidates.append(Path(which))
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        "LTspice not found. Install LTspice (analog.com/ltspice) or set LTSPICE_EXE."
    )


@dataclass
class SimResult:
    cfg: DPTConfig
    raw: RawFile
    workdir: Path

    def waveforms(self) -> Waveforms:
        r = self.raw
        return Waveforms(
            t=r.time,
            vds=r["V(dut_d)"] - r["V(dut_s)"],
            id=r["I(Vid)"],
            vgs=r["V(dut_g)"] - r["V(dut_ks)"],
            vds_hs=r["V(bus_p)"] - r["V(sw)"],
            i_hs=r["I(Vihs)"],
            vgs_hs=r["V(hs_g)"] - r["V(hs_ks)"],
        )

    def load_current_at(self, t: float) -> float:
        return float(np.interp(t, self.raw.time, self.raw["I(Lload)"]))


def _complete(raw_path: Path, log_path: Path, t_stop: float) -> bool:
    if not raw_path.is_file() or not log_path.is_file():
        return False
    log = log_path.read_text(encoding="utf-8", errors="replace")
    if "Simulation Failed" in log or "Total elapsed time" not in log:
        return False
    try:
        raw = read_raw(raw_path)
    except Exception:
        return False
    return bool(raw.time[-1] >= 0.999 * t_stop)


def simulate(
    cfg: DPTConfig,
    cache_root: Path = CACHE_ROOT,
    model_root: Path = MODEL_ROOT,
    timeout_s: float = 3600.0,
    force: bool = False,
) -> SimResult:
    """Simulate ``cfg`` (or reuse a cached result) and return the parsed output."""
    workdir = cache_root / cfg.key()
    net = workdir / "dpt.net"
    raw_path, log_path = workdir / "dpt.raw", workdir / "dpt.log"
    if force or not _complete(raw_path, log_path, cfg.t_stop):
        workdir.mkdir(parents=True, exist_ok=True)
        net.write_text(render(cfg, model_root), encoding="ascii")
        exe = find_ltspice()
        try:
            subprocess.run(
                [str(exe), "-b", net.name],
                cwd=workdir,
                timeout=timeout_s,
                check=False,
                capture_output=True,
            )
        except subprocess.TimeoutExpired as exc:
            raise SimulationError(f"LTspice timed out after {timeout_s:.0f} s ({net})") from exc
        if not _complete(raw_path, log_path, cfg.t_stop):
            tail = log_path.read_text(errors="replace")[-600:] if log_path.is_file() else ""
            raise SimulationError(f"LTspice run incomplete for {net}:\n{tail}")
    return SimResult(cfg=cfg, raw=read_raw(raw_path), workdir=workdir)
