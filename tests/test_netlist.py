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
    assert cfg.t_pulse1 == pytest.approx(1e-6 - cfg.t_delay_comp)
    assert cfg.t_turn_on - cfg.t_turn_off == pytest.approx(cfg.t_off + cfg.t_edge_drv)
    pwl = _line(render(cfg, ROOT), "Vdrv").split("PWL(")[1].rstrip(")").split()
    pts = [(float(pwl[k]), float(pwl[k + 1])) for k in range(0, len(pwl), 2)]
    times = [p[0] for p in pts]
    assert times == sorted(times)
    # gate is on exactly over pulse 1, then off for t_off, then on again
    assert any(t == pytest.approx(cfg.t_turn_off) and v == cfg.vgs_on for t, v in pts)
    assert pts[-1][0] == pytest.approx(cfg.t_turn_on + cfg.t_edge_drv)
    assert pts[-1][1] == cfg.vgs_on
    assert pts[-2][0] == pytest.approx(cfg.t_turn_on)
    assert pts[-2][1] == cfg.vgs_off
    assert pts[2][0] - pts[1][0] == pytest.approx(cfg.t_edge_first)


def test_kelvin_and_common_source_modes() -> None:
    base = DPTConfig(device=NTH4L022N120M3S)
    kelvin = render(base, ROOT)
    assert "Vdrv drv0 dut_ks" in kelvin
    non_kelvin = render(base.with_(kelvin=False, l_cs=5e-9), ROOT)
    assert "Vdrv drv0 cs" in non_kelvin
    assert _line(non_kelvin, "Lcs").split()[1:3] == ["dut_s", "cs"]


def test_split_gate_resistors_use_steering_diodes() -> None:
    text = render(DPTConfig(device=NTH4L022N120M3S, rg_on=2.0, rg_off=10.0), ROOT)
    assert "Rgon drv gon 2" in text and "Rgoff goff drv 10" in text
    assert ".model DGATE D(" in text


def test_cache_key_changes_with_parameters() -> None:
    a = DPTConfig(device=NTH4L022N120M3S)
    assert a.key() == DPTConfig(device=NTH4L022N120M3S).key()
    assert a.key() != a.with_(rg_on=10.0).key()
