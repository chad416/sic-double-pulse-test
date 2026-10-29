"""Quick sanity checks of each vendor model before it is used in the DPT study.

1. Static on-resistance at the datasheet test point (25 C and 175 C).
2. Resistive-load turn-on: delay from the gate edge to vDS = 50 %. With an ideal
   driver and Rg ~ 5 Ohm a healthy model switches in tens of ns; a delay of
   hundreds of ns means the model does not behave correctly in this simulator.

Only models that pass both checks are used for switching-loss results.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from dpt.devices import NTH4L022N120M3S, NTH4L032N065M3S
from dpt.extract import ExtractionError, crossing_time
from dpt.ltspice import CACHE_ROOT, MODEL_ROOT, find_ltspice
from dpt.netlist import Device
from dpt.rawread import read_raw

TURN_ON_DELAY_LIMIT = 100e-9  # s, generous upper bound for a healthy 4-pin SiC model


@dataclass(frozen=True)
class CheckPoint:
    device: Device
    id_test: float  # A, datasheet RDS(on) test current
    rds_25_ds: float  # Ohm, typical at 25 C
    rds_175_ds: float  # Ohm, typical at 175 C
    vbus: float  # V, resistive-load test voltage
    rg: float  # Ohm


CHECKS = (
    CheckPoint(NTH4L022N120M3S, 40.0, 22e-3, 44e-3, 800.0, 4.5),
    CheckPoint(NTH4L032N065M3S, 15.0, 32e-3, 49e-3, 400.0, 4.7),
)


def _pins(dev: Device, d: str, g: str, s: str, ks: str) -> str:
    nodes = {"d": d, "g": g, "s": s, "ks": ks}
    return " ".join(nodes[r] for r in dev.pin_order)


def _lib(dev: Device) -> str:
    return (MODEL_ROOT / dev.zip_stem / dev.library).resolve().as_posix()


def _run(name: str, text: str) -> Path:
    work = CACHE_ROOT.parent / "model_checks"
    work.mkdir(parents=True, exist_ok=True)
    net = work / f"{name}.net"
    net.write_text(text, encoding="ascii")
    subprocess.run([str(find_ltspice()), "-b", net.name], cwd=work, timeout=300, check=False)
    return work / f"{name}.raw"


def rds_on(cp: CheckPoint, temp: float) -> float:
    """On-resistance from a DC operating point at the datasheet current."""
    dev = cp.device
    text = "\n".join(
        [
            f"* RDS(on) check {dev.part} at {temp:g} C",
            f'.lib "{_lib(dev)}"',
            f"Id 0 d {cp.id_test:g}",
            f"X1 {_pins(dev, 'd', 'g', '0', 'ks')} {dev.subckt}",
            "Vg g ks 18",
            "Rks ks 0 1m",
            f".temp {temp:g}",
            ".op",
            ".options solver=alt",
            ".end",
        ]
    )
    raw = read_raw(_run(f"rds_{dev.part}_{temp:g}", text))  # .op result is in <name>.raw
    return float(raw["V(d)"][0]) / cp.id_test


def resistive_turn_on_delay(cp: CheckPoint) -> float:
    """Gate-edge-to-50 %-vDS delay into a resistive load drawing the test current."""
    dev = cp.device
    r_load = cp.vbus / cp.id_test
    text = "\n".join(
        [
            f"* resistive turn-on check {dev.part}",
            f'.lib "{_lib(dev)}"',
            f"Vbus vin 0 PWL(0 0 1u {cp.vbus:g}) Rser=10m",
            "Cbus vin 0 10u",
            "Ls vin b 10n",
            f"Rl b d {r_load:g}",
            f"X1 {_pins(dev, 'd', 'g', '0', 'ks')} {dev.subckt}",
            "Vdrv drv ks PWL(0 -3 1.5u -3 1.505u 18 3u 18)",
            f"Rg drv g {cp.rg:g}",
            ".tran 0 3u",
            ".options plotwinsize=0 solver=alt abstol=1e-7 vntol=1e-4 cshunt=1e-21",
            ".save V(d)",
            ".end",
        ]
    )
    raw = read_raw(_run(f"ton_{dev.part}", text))
    try:
        t50 = crossing_time(raw.time, raw["V(d)"], 0.5 * cp.vbus, "falling", 1.5e-6)
    except ExtractionError:
        return float("inf")
    return t50 - 1.5e-6


def run_checks() -> list[dict[str, object]]:
    """Both checks for every part; ``passed`` is the turn-on delay criterion."""
    rows: list[dict[str, object]] = []
    for cp in CHECKS:
        r25, r175 = rds_on(cp, 25.0), rds_on(cp, 175.0)
        delay = resistive_turn_on_delay(cp)
        rows.append(
            {
                "part": cp.device.part,
                "rds_on_25_sim_mohm": r25 * 1e3,
                "rds_on_25_ds_mohm": cp.rds_25_ds * 1e3,
                "rds_on_175_sim_mohm": r175 * 1e3,
                "rds_on_175_ds_mohm": cp.rds_175_ds * 1e3,
                "turn_on_delay_ns": delay * 1e9,
                "passed": bool(delay < TURN_ON_DELAY_LIMIT),
            }
        )
    return rows


if __name__ == "__main__":
    for row in run_checks():
        print(
            "  ".join(
                f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()
            )
        )
