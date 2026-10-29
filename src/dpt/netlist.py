"""Double-pulse test netlist generation (LTspice dialect).

Circuit (low-side DUT, high-side freewheeling device of the same part number):

::

    Vbus --+-- Lloop/2 --+----------+------------+
           |             |          |            |
         Cbus          Vihs       Lload        (bus_p)
           |             |          |
           |           XHS (gate held at Vgs_off through Rg_off)
           |             |          |
           |             +---- sw --+
           |                   |
           |                  Vid (drain current sense)
           |                   |
           |                  XLS (DUT)  <-- gate driver, Rg_on / Rg_off
           |                   |
           +---- Lloop/2 ------+ (power source pin)

* ``Lloop`` is the *external* commutation-loop inductance (PCB + capacitor ESL);
  the package inductances are inside the onsemi sub-circuit.
* Kelvin mode: the driver returns to the Kelvin-source pin (SS).
  Non-Kelvin mode: the driver returns to the power-source side of an extra
  common-source inductance ``Lcs`` (PCB trace shared by gate and power loops),
  and the Kelvin pin is left unconnected.
* Separate turn-on / turn-off gate resistors are steered by ideal diodes.

The netlist is plain text so it can be inspected, diffed and re-run by hand.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class Device:
    """An onsemi part and the model library it lives in."""

    part: str  # orderable part number, e.g. NTH4L022N120M3S
    subckt: str  # sub-circuit name inside the library (4-pin, isothermal)
    library: str  # library file name inside third_party/onsemi/<zip stem>/
    zip_stem: str  # name of the onsemi ZIP (without .zip)
    vds_rating: float  # V
    rg_int: float  # internal gate resistance from the datasheet, Ohm
    vth_min: float  # minimum gate threshold voltage at 25 C, V (false turn-on margin)
    # Sub-circuit pin order using the roles d, g, s (power source), ks (Kelvin source).
    # The onsemi libraries are not consistent: the 1200 V M3S symbol orders D G S SS,
    # the 650 V M3S symbol orders D G S1 S2 with S1 = driver (Kelvin) source.
    pin_order: tuple[str, str, str, str] = ("d", "g", "s", "ks")


@dataclass(frozen=True)
class DPTConfig:
    """All knobs of one double-pulse simulation."""

    device: Device
    vbus: float = 800.0  # V
    i_load: float = 40.0  # A, target current at the first turn-off
    temp: float = 25.0  # C, ambient = junction (isothermal model)
    rg_on: float = 4.5  # Ohm, external
    rg_off: float = 4.5  # Ohm, external
    vgs_on: float = 18.0  # V
    vgs_off: float = -3.0  # V
    t_edge_drv: float = 5e-9  # s, driver output 0-100 % edge of the measured edges
    t_edge_first: float = 50e-9  # s, edge of the first (unmeasured, zero-current) turn-on
    l_loop: float = 30e-9  # H, external power-loop inductance
    l_cs: float = 0.0  # H, extra common-source inductance (non-Kelvin mode only)
    kelvin: bool = True
    # Freewheeling device: "body_diode" = same MOSFET held off at vgs_off (datasheet
    # condition); "sbd" = generic recovery-free SiC Schottky diode (NOT an onsemi model,
    # used only to separate DUT turn-on from body-diode recovery effects).
    freewheel: str = "body_diode"
    l_load: float = 20e-6  # H
    c_bus: float = 10e-6  # F
    t_off: float = 1e-6  # s, freewheeling interval between the two pulses
    t_after: float = 0.5e-6  # s, simulated time after the second turn-on
    t_ramp: float = 2e-6  # s, DC-link ramp at start (eases the initial operating point)
    t_settle: float = 1e-6  # s, quiet time before the first pulse
    max_step: float = 0.0  # s, transient maximum time step (0 = simulator default)
    # Solver settings. "solver=alt" is onsemi's recommendation (AND90315). cshunt adds a
    # capacitor from every node to ground; it is needed for convergence but values
    # >= 1e-20 F visibly slow the modelled turn-on (see DESIGN.md, "Convergence").
    options: tuple[str, ...] = ("solver=alt", "abstol=1e-7", "vntol=1e-4")
    cshunt: float = 1e-21  # F

    # ---- derived timing -------------------------------------------------------
    @property
    def t_first_on(self) -> float:
        return self.t_ramp + self.t_settle

    @property
    def t_pulse1(self) -> float:
        """First-pulse width that charges the load inductor to ``i_load``."""
        return self.l_load * self.i_load / self.vbus

    @property
    def t_turn_off(self) -> float:
        """Gate command edge of the measured turn-off (end of pulse 1)."""
        return self.t_first_on + self.t_edge_first + self.t_pulse1

    @property
    def t_turn_on(self) -> float:
        """Gate command edge of the measured turn-on (start of pulse 2)."""
        return self.t_turn_off + self.t_edge_drv + self.t_off

    @property
    def t_stop(self) -> float:
        return self.t_turn_on + self.t_after

    def with_(self, **changes: object) -> DPTConfig:
        return replace(self, **changes)  # type: ignore[arg-type]

    def key(self) -> str:
        """Stable short hash of the configuration, used as a cache key."""
        blob = json.dumps(asdict(self), sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


SAVE_VECTORS = (
    "V(dut_d)",
    "V(dut_s)",
    "V(dut_g)",
    "V(dut_ks)",
    "V(bus_p)",
    "V(sw)",
    "V(hs_g)",
    "V(hs_ks)",
    "I(Vid)",
    "I(Vihs)",
    "I(Lload)",
)


def _pins(dev: Device, prefix: str, d: str) -> str:
    """Instance pin list for ``dev``; ``d`` is the drain node name."""
    nodes = {"d": d, "g": f"{prefix}_g", "s": f"{prefix}_s", "ks": f"{prefix}_ks"}
    return " ".join(nodes[role] for role in dev.pin_order)


# Generic 1200 V SiC Schottky diode: junction capacitance only, no stored charge.
SBD_MODEL = ".model DSBD D(Is=1e-13 N=1.5 Rs=25m Cjo=2.5n Vj=1.2 M=0.5 Tt=0 BV=1300 Ibv=1u)"


def _freewheel(cfg: DPTConfig) -> list[str]:
    if cfg.freewheel == "body_diode":
        return [
            f"XHS {_pins(cfg.device, 'hs', 'hs_d')} {cfg.device.subckt}",
            "Vhss hs_s sw 0",
            f"Vhsoff hs_drv hs_ks {cfg.vgs_off:g}",
            f"Rghs hs_drv hs_g {cfg.rg_off:g}",
        ]
    if cfg.freewheel == "sbd":
        # hs_g / hs_ks kept as dummy nodes so the saved vector list is identical.
        return ["Dfw sw hs_d DSBD", SBD_MODEL, "Rdum1 hs_g 0 1G", "Rdum2 hs_ks 0 1G"]
    raise ValueError(f"unknown freewheel device {cfg.freewheel!r}")


def render(cfg: DPTConfig, model_root: Path) -> str:
    """Return the LTspice netlist text for ``cfg``."""
    dev = cfg.device
    lib = (model_root / dev.zip_stem / dev.library).resolve().as_posix()
    # Gate command as PWL: a soft first edge (zero-current turn-on, not measured, eases
    # convergence), then the measured turn-off and turn-on edges with t_edge_drv.
    off, on = cfg.vgs_off, cfg.vgs_on
    pwl = [
        (0.0, off),
        (cfg.t_first_on, off),
        (cfg.t_first_on + cfg.t_edge_first, on),
        (cfg.t_turn_off, on),
        (cfg.t_turn_off + cfg.t_edge_drv, off),
        (cfg.t_turn_on, off),
        (cfg.t_turn_on + cfg.t_edge_drv, on),
    ]
    pwl_text = " ".join(f"{t:.6g} {v:g}" for t, v in pwl)
    if cfg.kelvin:
        ls_driver_ref = "dut_ks"
        source_path = "Lsrc dut_s 0 {Lloop/2} Rser=0.1m"
    else:
        ls_driver_ref = "cs"
        source_path = "\n".join(
            [
                "Lcs dut_s cs {Lcs} Rser=0.1m",
                "Lsrc cs 0 {Lloop/2} Rser=0.1m",
            ]
        )
        if cfg.l_cs <= 0:
            source_path = "Lsrc dut_s 0 {Lloop/2} Rser=0.1m"
            ls_driver_ref = "dut_s"
    if cfg.rg_on == cfg.rg_off:
        driver = f"Rg drv dut_g {cfg.rg_on:g}"
    else:
        driver = "\n".join(
            [
                f"Rgon drv gon {cfg.rg_on:g}",
                "Don gon dut_g DGATE",
                "Doff dut_g goff DGATE",
                f"Rgoff goff drv {cfg.rg_off:g}",
                ".model DGATE D(Ron=1m Roff=1G Vfwd=0 epsilon=20m)",
            ]
        )
    lines = [
        f"* Double-pulse test: {dev.part}  (generated by dpt.netlist, key {cfg.key()})",
        f'.lib "{lib}"',
        f".param Vbus={cfg.vbus:g} Lloop={cfg.l_loop:g} Lcs={cfg.l_cs:g}",
        f".temp {cfg.temp:g}",
        "* --- DC link: source ramps up so the initial operating point is trivial",
        f"Vbus vin 0 PWL(0 0 {cfg.t_ramp:g} {{Vbus}}) Rser=10m",
        f"Cbus vin 0 {cfg.c_bus:g} Rser=2m",
        "Lbus vin bus_p {Lloop/2} Rser=0.1m",
        "* --- high side: freewheeling device",
        "Vihs bus_p hs_d 0",
        *_freewheel(cfg),
        f"Lload bus_p sw {cfg.l_load:g} Rser=10m",
        "* --- low side: device under test",
        "Vid sw dut_d 0",
        f"XLS {_pins(dev, 'dut', 'dut_d')} {dev.subckt}",
        source_path,
        "* --- gate driver (ideal source, double pulse as PWL)",
        f"Vdrv drv {ls_driver_ref} PWL({pwl_text})",
        driver,
        f".tran 0 {cfg.t_stop:g}" + (f" 0 {cfg.max_step:g}" if cfg.max_step > 0 else ""),
        ".options plotwinsize=0 " + " ".join(cfg.options) + f" cshunt={cfg.cshunt:g}",
        ".save " + " ".join(SAVE_VECTORS),
        ".end",
    ]
    return "\n".join(lines) + "\n"
