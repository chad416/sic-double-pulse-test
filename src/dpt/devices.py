"""Devices under test and their datasheet switching-loss reference points.

Every number here is copied from the onsemi datasheet named in ``source``; the
validation step compares simulated losses against them.
"""

from __future__ import annotations

from dataclasses import dataclass

from dpt.netlist import Device

NTH4L022N120M3S = Device(
    part="NTH4L022N120M3S",
    subckt="NTH4L022N120M3S_4P",
    library="ONSEMI_SiCMOSFET_1200M3S_ltspice.txt",
    zip_stem="onsemi_sicmosfet_1200m3s_ltspice",
    vds_rating=1200.0,
    rg_int=1.5,
    vth_min=2.04,
    vgs_abs_max=22.0,
    vgs_abs_min=-10.0,
)

NTH4L032N065M3S = Device(
    part="NTH4L032N065M3S",
    subckt="NTH4L032N065M3S_4P",
    library="ONSEMI_SICMOSFET_650M3S_ltspice.txt",
    zip_stem="onsemi_sicmosfet_650m3s_ltspice",
    vds_rating=650.0,
    rg_int=5.0,
    vth_min=2.0,
    pin_order=("d", "g", "ks", "s"),
)

DEVICES = {d.part: d for d in (NTH4L022N120M3S, NTH4L032N065M3S)}


@dataclass(frozen=True)
class DatasheetPoint:
    """One row of a datasheet "Switching characteristics" table."""

    device: Device
    vbus: float  # V
    i_load: float  # A
    rg: float  # Ohm, external
    temp: float  # C
    e_on: float  # J
    e_off: float  # J
    td_on: float | None  # s
    tr: float | None  # s
    td_off: float | None  # s
    tf: float | None  # s
    source: str


DATASHEET_POINTS = (
    DatasheetPoint(
        device=NTH4L022N120M3S,
        vbus=800.0,
        i_load=40.0,
        rg=4.5,
        temp=25.0,
        e_on=490e-6,
        e_off=221e-6,
        td_on=18e-9,
        tr=24e-9,
        td_off=48e-9,
        tf=13e-9,
        source="NTH4L022N120M3S/D Rev. 9, Table 3 (VGS = -3/18 V, inductive load, body diode)",
    ),
    DatasheetPoint(
        device=NTH4L032N065M3S,
        vbus=400.0,
        i_load=15.0,
        rg=4.7,
        temp=25.0,
        e_on=33e-6,
        e_off=16e-6,
        td_on=8.8e-9,
        tr=12e-9,
        td_off=31e-9,
        tf=9e-9,
        source="NTH4L032N065M3S/D Rev. 6, switching characteristics, TJ = 25 C",
    ),
    DatasheetPoint(
        device=NTH4L032N065M3S,
        vbus=400.0,
        i_load=15.0,
        rg=4.7,
        temp=175.0,
        e_on=31e-6,
        e_off=25e-6,
        td_on=7.8e-9,
        tr=12e-9,
        td_off=37e-9,
        tf=11e-9,
        source="NTH4L032N065M3S/D Rev. 6, switching characteristics, TJ = 175 C",
    ),
)
