"""Extraction functions checked against synthetic waveforms with known answers."""

from __future__ import annotations

import numpy as np
import pytest

from dpt.extract import (
    ExtractionError,
    Thresholds,
    Waveforms,
    crossing_time,
    extract_turn_off,
    extract_turn_on,
    integrate,
    ringing,
)

V, IL = 800.0, 40.0
VGS_ON, VGS_OFF = 18.0, -3.0
NS = 1e-9


def _nonuniform_time(t_end: float, n: int = 6000, seed: int = 1) -> np.ndarray:
    """Irregular time grid, as a SPICE simulator would produce."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0.0, t_end, n))
    return np.concatenate(([0.0], t, [t_end]))


def _ramp(t: np.ndarray, t0: float, t1: float, y0: float, y1: float) -> np.ndarray:
    return np.interp(t, [t0, t1], [y0, y1])


def _turn_off_waveforms(t: np.ndarray, ring_amp: float = 0.0) -> Waveforms:
    vgs = _ramp(t, 100 * NS, 120 * NS, VGS_ON, VGS_OFF)
    vds = _ramp(t, 130 * NS, 150 * NS, 0.0, V)
    id_ = _ramp(t, 150 * NS, 160 * NS, IL, 0.0)
    tau = np.clip(t - 150 * NS, 0.0, None)
    ring = ring_amp * np.exp(-tau / (40 * NS)) * np.sin(2 * np.pi * 50e6 * tau)
    vds = vds + np.where(t > 150 * NS, ring, 0.0)
    return Waveforms(
        t=t,
        vds=vds,
        id=id_,
        vgs=vgs,
        vds_hs=V - vds,
        i_hs=id_ - IL,
        vgs_hs=np.full_like(t, VGS_OFF),
    )


def test_crossing_time_interpolates() -> None:
    t = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 10.0, 0.0, 10.0])
    assert crossing_time(t, y, 5.0, "rising") == pytest.approx(0.5)
    assert crossing_time(t, y, 5.0, "falling") == pytest.approx(1.5)
    assert crossing_time(t, y, 5.0, "rising", t_start=1.0) == pytest.approx(2.5)
    with pytest.raises(ExtractionError):
        crossing_time(t, y, 50.0, "rising")


def test_integrate_partial_window_of_constant() -> None:
    t = np.linspace(0.0, 1.0, 11)
    y = np.full_like(t, 3.0)
    assert integrate(t, y, 0.123, 0.789) == pytest.approx(3.0 * (0.789 - 0.123))


def test_ringing_frequency_and_damping() -> None:
    f0, zeta0 = 37e6, 0.05
    wn = 2 * np.pi * f0 / np.sqrt(1 - zeta0**2)
    t = np.linspace(0.0, 400 * NS, 20000)
    y = 50 * np.exp(-zeta0 * wn * t) * np.sin(2 * np.pi * f0 * t)
    f, zeta = ringing(t, y, 0.0, 400 * NS)
    assert f == pytest.approx(f0, rel=0.01)
    assert zeta == pytest.approx(zeta0, rel=0.15)


def test_turn_off_energy_slopes_and_times() -> None:
    t = _nonuniform_time(400 * NS)
    w = _turn_off_waveforms(t)
    r = extract_turn_off(w, 100 * NS, V, IL, VGS_ON, VGS_OFF)
    # Window: 90 % vgs at 102 ns -> 2 % id at 159.8 ns.
    assert r.t_start == pytest.approx(102 * NS, rel=1e-3)
    assert r.t_end == pytest.approx(159.8 * NS, rel=1e-3)
    # vds ramp at full current (V*IL*20ns/2) + current fall to 2 % at full voltage.
    expected = V * IL * (10 * NS) + V * IL * (9.8 - 9.8**2 / 20) * NS
    assert r.e_off == pytest.approx(expected, rel=5e-3)
    assert r.dv_dt == pytest.approx(V / (20 * NS), rel=5e-3)
    assert r.di_dt == pytest.approx(-IL / (10 * NS), rel=5e-3)
    assert r.td_off == pytest.approx((151 - 102) * NS, rel=5e-3)
    assert r.tf == pytest.approx(8 * NS, rel=5e-3)
    assert r.td_off_v == pytest.approx((132 - 102) * NS, rel=5e-3)
    assert r.tf_v == pytest.approx(16 * NS, rel=5e-3)


def test_turn_off_overshoot_and_ringing() -> None:
    t = np.linspace(0.0, 600 * NS, 60001)
    w = _turn_off_waveforms(t, ring_amp=120.0)
    r = extract_turn_off(w, 100 * NS, V, IL, VGS_ON, VGS_OFF)
    assert r.vds_peak == pytest.approx(float(np.max(w.vds)))
    assert r.overshoot == pytest.approx(float(np.max(w.vds)) - V)
    assert 60.0 < r.overshoot < 120.0
    assert r.f_ring == pytest.approx(50e6, rel=0.02)


def _turn_on_waveforms(t: np.ndarray) -> Waveforms:
    vgs = _ramp(t, 100 * NS, 120 * NS, VGS_OFF, VGS_ON)
    i_rise = _ramp(t, 130 * NS, 140 * NS, 0.0, IL)
    # Reverse-recovery current of the freewheeling diode: triangle, 10 A peak at 145 ns.
    i_rr = np.interp(t, [140 * NS, 145 * NS, 155 * NS], [0.0, 10.0, 0.0])
    vds = _ramp(t, 140 * NS, 160 * NS, V, 0.0)
    id_ = i_rise + i_rr
    bounce = 2.0 * np.interp(t, [140 * NS, 150 * NS, 160 * NS], [0.0, 1.0, 0.0])
    return Waveforms(
        t=t,
        vds=vds,
        id=id_,
        vgs=vgs,
        vds_hs=V - vds,
        i_hs=id_ - IL,
        vgs_hs=VGS_OFF + bounce,
    )


def _dense_reference(f, t0: float, t1: float) -> float:
    tt = np.linspace(t0, t1, 200001)
    return float(np.trapezoid(f(tt), tt))


def test_turn_on_energy_recovery_and_bounce() -> None:
    t = _nonuniform_time(400 * NS, n=8000, seed=7)
    w = _turn_on_waveforms(t)
    r = extract_turn_on(w, 100 * NS, V, IL, VGS_ON, VGS_OFF)
    # Window: 10 % vgs at 102 ns -> vds at 2 % of V at 159.6 ns.
    assert r.t_start == pytest.approx(102 * NS, rel=1e-3)
    assert r.t_end == pytest.approx(159.6 * NS, rel=1e-3)

    def p_on(tt: np.ndarray) -> np.ndarray:
        wd = _turn_on_waveforms(tt)
        return wd.vds * wd.id

    assert r.e_on == pytest.approx(_dense_reference(p_on, 102 * NS, 159.6 * NS), rel=5e-3)

    # Err window: i_hs crosses zero at 140 ns, decays to 2 % of 10 A at 154.8 ns.
    def p_rr(tt: np.ndarray) -> np.ndarray:
        wd = _turn_on_waveforms(tt)
        return wd.vds_hs * wd.i_hs

    assert r.i_rrm == pytest.approx(10.0, rel=1e-2)
    assert r.e_rr == pytest.approx(_dense_reference(p_rr, 140 * NS, 154.8 * NS), rel=1e-2)
    assert r.q_rr == pytest.approx(0.5 * 10.0 * 15 * NS * (1 - 0.02**2), rel=1e-2)
    assert r.vgs_hs_max == pytest.approx(VGS_OFF + 2.0, abs=0.02)
    assert r.vds_hs_peak == pytest.approx(V, rel=1e-3)
    assert r.di_dt == pytest.approx(IL / (10 * NS), rel=5e-3)
    assert r.dv_dt == pytest.approx(-V / (20 * NS), rel=5e-3)
    assert r.td_on == pytest.approx((131 - 102) * NS, rel=5e-3)
    assert r.tr == pytest.approx(8 * NS, rel=5e-3)
    assert r.td_on_v == pytest.approx((142 - 102) * NS, rel=5e-3)
    assert r.tr_v == pytest.approx(16 * NS, rel=5e-3)


def test_window_thresholds_are_respected() -> None:
    t = _nonuniform_time(400 * NS)
    w = _turn_off_waveforms(t)
    loose = extract_turn_off(w, 100 * NS, V, IL, VGS_ON, VGS_OFF, Thresholds(id_off_end=0.10))
    tight = extract_turn_off(w, 100 * NS, V, IL, VGS_ON, VGS_OFF)
    assert loose.t_end < tight.t_end
    assert loose.e_off < tight.e_off
