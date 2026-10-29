"""Parameter sweeps: run many DPT configurations and tabulate the metrics.

Each configuration is simulated once and cached (see ``dpt.ltspice``); this
module turns a list of configurations into rows of extracted metrics.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from dpt.extract import (
    ExtractionError,
    as_dict,
    crossing_time,
    extract_turn_off,
    extract_turn_on,
)
from dpt.ltspice import SimResult, SimulationError, simulate_robust
from dpt.netlist import DPTConfig

# Configuration columns written next to the metrics in every CSV row.
CONFIG_COLUMNS = (
    "part",
    "vbus",
    "i_load",
    "temp",
    "rg_on",
    "rg_off",
    "l_loop",
    "l_cs",
    "kelvin",
    "key",
)


@dataclass
class RunRecord:
    cfg: DPTConfig
    metrics: dict[str, float]
    error: str = ""

    def row(self) -> dict[str, object]:
        c = self.cfg
        out: dict[str, object] = {
            "part": c.device.part,
            "vbus": c.vbus,
            "i_load": c.i_load,
            "temp": c.temp,
            "rg_on": c.rg_on,
            "rg_off": c.rg_off,
            "l_loop": c.l_loop,
            "l_cs": c.l_cs,
            "kelvin": int(c.kelvin),
            "key": c.key(),
        }
        out.update(self.metrics)
        out["error"] = self.error
        return out


def metrics_from(sim: SimResult) -> dict[str, float]:
    """Extract all turn-off / turn-on metrics of one finished simulation."""
    c = sim.cfg
    w = sim.waveforms()
    # Switched currents: at turn-off the load current when vds starts to rise (the
    # device keeps conducting through its turn-off delay); at turn-on the free-wheeling
    # current at the gate command.
    t_v10 = crossing_time(w.t, w.vds, 0.1 * c.vbus, "rising", c.t_turn_off)
    i_off = sim.load_current_at(t_v10)
    i_on = sim.load_current_at(c.t_turn_on)
    off = extract_turn_off(w, c.t_turn_off, c.vbus, i_off, c.vgs_on, c.vgs_off)
    on = extract_turn_on(w, c.t_turn_on, c.vbus, i_on, c.vgs_on, c.vgs_off)
    m: dict[str, float] = {"i_off": i_off, "i_on": i_on}
    m.update({f"off_{k}": v for k, v in as_dict(off).items()})
    m.update({f"on_{k}": v for k, v in as_dict(on).items()})
    m["e_ts"] = m["off_e_off"] + m["on_e_on"]
    m["vgs_hs_margin"] = c.device.vth_min - m["on_vgs_hs_max"]
    # A gate swing outside the absolute ratings means the stage oscillates: the loss
    # numbers of such a run describe an unusable design, not a loss to optimise.
    vg_hi = max(m["on_vgs_max"], m["off_vgs_max"])
    vg_lo = min(m["on_vgs_min"], m["off_vgs_min"])
    m["gate_ok"] = float(vg_hi <= c.device.vgs_abs_max and vg_lo >= c.device.vgs_abs_min)
    return m


def run_one(cfg: DPTConfig) -> RunRecord:
    try:
        sim = simulate_robust(cfg)
        m = metrics_from(sim)
        m["cshunt"] = sim.cfg.cshunt
        return RunRecord(cfg, m)
    except (SimulationError, ExtractionError) as exc:
        return RunRecord(cfg, {}, error=f"{type(exc).__name__}: {str(exc).splitlines()[0]}")


def run_many(cfgs: Sequence[DPTConfig], workers: int = 2) -> list[RunRecord]:
    """Simulate (or load from cache) every configuration, preserving order.

    LTspice is itself multi-threaded, so a small worker count is usually fastest.
    """
    unique = list(dict.fromkeys(cfgs))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        done = dict(zip(unique, ex.map(run_one, unique), strict=True))
    return [done[c] for c in cfgs]


def write_csv(records: Iterable[RunRecord], path: Path) -> None:
    rows = [r.row() for r in records]
    keys: list[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n")
        wr.writeheader()
        for row in rows:
            wr.writerow({k: _fmt(row.get(k, "")) for k in keys})


def _fmt(v: object) -> object:
    if isinstance(v, float):
        return "" if math.isnan(v) else f"{v:.6g}"
    return v


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))
