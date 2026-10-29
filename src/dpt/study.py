"""The study: every simulation behind the README, defined in one place.

Nominal operating point (1200 V part): 800 V, 40 A, Rg = 4.5 Ohm, -3/+18 V,
25 C, 30 nH external power loop, Kelvin-source drive, body-diode free-wheeling.
This matches the datasheet switching-loss condition; 30 nH is a typical value
for a discrete TO-247 double-pulse board (the datasheet does not state its own).
"""

from __future__ import annotations

from dataclasses import dataclass

from dpt.devices import DATASHEET_POINTS, NTH4L022N120M3S, DatasheetPoint
from dpt.netlist import DPTConfig

NH = 1e-9

NOMINAL = DPTConfig(device=NTH4L022N120M3S)

# Design limits used by the analysis (stated here so every threshold is reviewable).
V_PEAK_LIMIT_FRACTION = 0.80  # keep every drain-source peak below 80 % of V(BR)DSS
DVDT_LIMIT = 50e9  # V/s, a common EMI / motor-insulation guideline for SiC drives

LOOP_BRACKET = (10 * NH, 20 * NH, 30 * NH, 50 * NH)
CSHUNT_STUDY = (1e-15, 1e-16, 1e-17, 1e-19, 1e-21, 1e-22)


@dataclass(frozen=True)
class Sweep:
    name: str
    param: str  # x-axis variable name as written in the CSV
    label: str  # axis label
    values: tuple[float, ...]
    configs: tuple[DPTConfig, ...]


def validation_configs(
    parts: set[str] | None = None,
) -> list[tuple[DatasheetPoint, float, DPTConfig]]:
    """Datasheet test conditions x loop-inductance bracket, for the given parts."""
    out = []
    for p in DATASHEET_POINTS:
        if parts is not None and p.device.part not in parts:
            continue
        for l_loop in LOOP_BRACKET:
            cfg = DPTConfig(
                device=p.device,
                vbus=p.vbus,
                i_load=p.i_load,
                rg_on=p.rg,
                rg_off=p.rg,
                temp=p.temp,
                l_loop=l_loop,
            )
            out.append((p, l_loop, cfg))
    return out


def convergence_configs() -> list[DPTConfig]:
    return [NOMINAL.with_(cshunt=c) for c in CSHUNT_STUDY]


def _sweep(name: str, param: str, label: str, values: tuple[float, ...], base: DPTConfig) -> Sweep:
    return Sweep(name, param, label, values, tuple(base.with_(**{param: v}) for v in values))


def sweeps() -> list[Sweep]:
    rg = (1.0, 2.2, 4.5, 10.0, 15.0, 22.0)
    out = [
        Sweep(
            "rg",
            "rg_on",
            "Gate resistance Rg,on = Rg,off (Ohm)",
            rg,
            tuple(NOMINAL.with_(rg_on=r, rg_off=r) for r in rg),
        ),
        _sweep("rg_on", "rg_on", "Turn-on gate resistance Rg,on (Ohm)", rg, NOMINAL),
        _sweep("rg_off", "rg_off", "Turn-off gate resistance Rg,off (Ohm)", rg, NOMINAL),
        _sweep(
            "l_loop",
            "l_loop",
            "External power-loop inductance (nH)",
            (5 * NH, 10 * NH, 20 * NH, 30 * NH, 50 * NH, 80 * NH),
            NOMINAL,
        ),
        _sweep("vbus", "vbus", "DC-link voltage (V)", (400.0, 600.0, 800.0), NOMINAL),
        _sweep("i_load", "i_load", "Switched current (A)", (10.0, 20.0, 40.0, 60.0), NOMINAL),
        _sweep("temp", "temp", "Temperature (C)", (25.0, 100.0, 150.0, 175.0), NOMINAL),
        _sweep(
            "l_cs",
            "l_cs",
            "Extra common-source inductance, non-Kelvin drive (nH)",
            (0.0, 2 * NH, 5 * NH, 10 * NH),
            NOMINAL.with_(kelvin=False),
        ),
    ]
    return out
