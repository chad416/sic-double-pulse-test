"""The study: every simulation behind the README, defined in one place.

Nominal operating point (1200 V part): 800 V, 40 A, Rg = 4.5 Ohm, -3/+18 V,
25 C, 30 nH external power loop, Kelvin-source drive, body-diode free-wheeling.
This matches the datasheet switching-loss condition; 30 nH is a typical value
for a discrete TO-247 double-pulse board (the datasheet does not state its own).
"""

from __future__ import annotations

from dataclasses import dataclass

from dpt.devices import DATASHEET_POINTS, NTH4L022N120M3S, DatasheetPoint
from dpt.extract import Thresholds
from dpt.netlist import DPTConfig

NH = 1e-9

NOMINAL = DPTConfig(device=NTH4L022N120M3S)

# Design limits used by the analysis (stated here so every threshold is reviewable).
V_PEAK_LIMIT_FRACTION = 0.80  # keep every drain-source peak below 80 % of V(BR)DSS
DVDT_LIMIT = 50e9  # V/s, a common EMI / motor-insulation guideline for SiC drives

LOOP_BRACKET = (10 * NH, 20 * NH, 30 * NH, 50 * NH)
CSHUNT_STUDY = (1e-15, 1e-16, 1e-17, 1e-19, 1e-21, 1e-22)
RG_VALUES = (1.0, 2.2, 4.5, 6.8, 10.0, 15.0, 22.0)

# Error budget at the nominal point: how far each uncertain input moves Eon / Eoff.
# Re-simulated variants (things the datasheet bench has and the ideal set-up lacks):
DRIVER_R_EXTRA = 1.0  # Ohm, typical gate-driver output resistance, added to Rg,on/Rg,off
GATE_LOOP_L = 10 * NH  # H, driver-to-gate loop inductance of a compact layout
# Re-extracted variants on the nominal waveforms (no new simulation):
WINDOW_VARIANTS: tuple[tuple[str, Thresholds], ...] = (
    ("Eon window ends at 10 % of Vbus instead of 2 %", Thresholds(vds_on_end=0.10)),
    ("Eoff window ends at 10 % of iD instead of 2 %", Thresholds(id_off_end=0.10)),
)
DESKEW_VARIANTS = (1e-9, -1e-9)  # s, DUT current trace late / early by 1 ns


@dataclass(frozen=True)
class Sweep:
    """A named one-parameter sweep."""

    name: str  # results/sweep_<name>.csv
    param: str  # the swept DPTConfig field
    configs: tuple[DPTConfig, ...]


def validation_configs(
    parts: set[str] | None = None,
) -> list[tuple[DatasheetPoint, float, DPTConfig]]:
    """Datasheet test conditions x loop-inductance bracket, for the given parts."""
    out: list[tuple[DatasheetPoint, float, DPTConfig]] = []
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
    """Nominal point at every cshunt value of the convergence-aid study."""
    return [NOMINAL.with_(cshunt=c) for c in CSHUNT_STUDY]


def budget_configs() -> list[tuple[str, DPTConfig]]:
    """Re-simulated error-budget variants of the nominal point."""
    r = NOMINAL.rg_on + DRIVER_R_EXTRA
    return [
        (
            f"Driver output resistance +{DRIVER_R_EXTRA:g} Ω on both edges",
            NOMINAL.with_(rg_on=r, rg_off=r),
        ),
        (f"Gate-loop inductance {GATE_LOOP_L / NH:g} nH", NOMINAL.with_(l_gate=GATE_LOOP_L)),
    ]


def _sweep(name: str, param: str, values: tuple[float, ...], base: DPTConfig = NOMINAL) -> Sweep:
    return Sweep(name, param, tuple(base.with_(**{param: v}) for v in values))


def sweeps() -> list[Sweep]:
    """Every one-parameter sweep around the nominal point."""
    rg = RG_VALUES
    return [
        Sweep("rg", "rg_on", tuple(NOMINAL.with_(rg_on=r, rg_off=r) for r in rg)),
        _sweep("rg_on", "rg_on", rg),
        _sweep("rg_off", "rg_off", rg),
        _sweep("l_loop", "l_loop", (5 * NH, 10 * NH, 20 * NH, 30 * NH, 50 * NH, 80 * NH)),
        _sweep("vbus", "vbus", (400.0, 600.0, 800.0)),
        _sweep("i_load", "i_load", (10.0, 20.0, 40.0, 60.0)),
        _sweep("temp", "temp", (25.0, 100.0, 150.0, 175.0)),
        _sweep("vgs_off", "vgs_off", (0.0, -3.0, -5.0)),
        _sweep("l_cs", "l_cs", (0.0, 2 * NH, 5 * NH, 10 * NH), NOMINAL.with_(kelvin=False)),
    ]
