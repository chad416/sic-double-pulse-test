"""Netlist generation: topology, timing and pin mapping (no simulator needed)."""

from __future__ import annotations

from pathlib import Path

import pytest

from dpt.devices import NTH4L022N120M3S, NTH4L032N065M3S
from dpt.netlist import DPTConfig, render

ROOT = Path("models")


def _line(text: str, prefix: str) -> str:
    matches = [ln for ln in text.splitlines() if ln.startswith(prefix)]
    assert len(matches) == 1, f"expected one line starting with {prefix!r}"
    return matches[0]


def test_pin_order_follows_device_definition() -> None:
    n1200 = render(DPTConfig(device=NTH4L022N120M3S), ROOT)
    n650 = render(DPTConfig(device=NTH4L032N065M3S, vbus=400, i_load=15), ROOT)
    # 1200 V symbol: D G S SS ; 650 V symbol: D G S1(Kelvin) S2(power)
    assert _line(n1200, "XLS").split()[1:5] == ["dut_d", "dut_g", "dut_s", "dut_ks"]
    assert _line(n650, "XLS").split()[1:5] == ["dut_d", "dut_g", "dut_ks", "dut_s"]
    assert _line(n650, "XLS").split()[-1] == NTH4L032N065M3S.subckt


def test_first_pulse_width_charges_inductor_to_target() -> None:
    cfg = DPTConfig(device=NTH4L022N120M3S, vbus=800, i_load=40, l_load=20e-6)
    assert cfg.t_pulse1 == pytest.approx(1e-6)
    assert cfg.t_turn_on - cfg.t_turn_off == pytest.approx(cfg.t_off + cfg.t_edge_drv)
    pulse = _line(render(cfg, ROOT), "Vdrv").split("PULSE(")[1].rstrip(")").split()
    v1, v2, tdel, tr, _tf, ton, period, ncyc = (float(x) for x in pulse)
    assert (v1, v2) == (cfg.vgs_off, cfg.vgs_on)
    assert tdel + tr + ton == pytest.approx(cfg.t_turn_off)
    assert tdel + period == pytest.approx(cfg.t_turn_on)
    assert ncyc == 2


def test_kelvin_and_common_source_modes() -> None:
    base = DPTConfig(device=NTH4L022N120M3S)
    kelvin = render(base, ROOT)
    assert "Vdrv drv dut_ks" in kelvin
    non_kelvin = render(base.with_(kelvin=False, l_cs=5e-9), ROOT)
    assert "Vdrv drv cs" in non_kelvin
    assert _line(non_kelvin, "Lcs").split()[1:3] == ["dut_s", "cs"]


def test_split_gate_resistors_use_steering_diodes() -> None:
    text = render(DPTConfig(device=NTH4L022N120M3S, rg_on=2.0, rg_off=10.0), ROOT)
    assert "Rgon drv gon 2" in text and "Rgoff goff drv 10" in text
    assert ".model DGATE D(" in text


def test_cache_key_changes_with_parameters() -> None:
    a = DPTConfig(device=NTH4L022N120M3S)
    assert a.key() == DPTConfig(device=NTH4L022N120M3S).key()
    assert a.key() != a.with_(rg_on=10.0).key()
