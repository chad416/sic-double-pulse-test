"""Reproduce every result of the project with one command.

    python -m dpt.run_all            # simulate (cached), analyse, plot, write reports
    python -m dpt.run_all --workers 2

Outputs
-------
results/*.csv           one row per simulation (configuration + all extracted metrics)
results/summary.json    every number quoted in README.md and docs/results.md
docs/figures/*.png      every figure
docs/results.md         full results tables (generated)
README.md               the blocks between the generated-block markers are rewritten
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from dpt import plots, report
from dpt.analysis import (
    best_under_constraints,
    combined_limit,
    deviation_pct,
    interp_at,
    max_x_below,
)
from dpt.devices import DATASHEET_POINTS
from dpt.extract import extract_turn_off, extract_turn_on
from dpt.fetch_models import main as fetch_main
from dpt.ltspice import CSHUNT_LADDER, PROJECT_ROOT, simulate, simulate_with_retry
from dpt.model_checks import run_checks
from dpt.plots import Panel
from dpt.study import (
    DESKEW_VARIANTS,
    DVDT_LIMIT,
    NH,
    NOMINAL,
    V_PEAK_LIMIT_FRACTION,
    WINDOW_VARIANTS,
    Sweep,
    budget_configs,
    convergence_configs,
    sweeps,
    validation_configs,
)
from dpt.sweeps import RunRecord, metrics_from, run_many, write_csv

RESULTS = PROJECT_ROOT / "results"
FIGURES = PROJECT_ROOT / "docs" / "figures"
README = PROJECT_ROOT / "README.md"
RESULTS_MD = PROJECT_ROOT / "docs" / "results.md"

UJ, NS, VNS, ANS = 1e6, 1e9, 1e-9, 1e-9
Row = dict[str, Any]
Sweeps = dict[str, tuple[Sweep, list[RunRecord]]]
NOMINAL_TITLE = "800 V, 40 A, Rg = 4.5 Ω, 30 nH loop, 25 °C"


def _col(records: list[RunRecord], key: str, scale: float = 1.0) -> list[float]:
    return [r.metrics.get(key, float("nan")) * scale for r in records]


def _ok(records: list[RunRecord]) -> list[RunRecord]:
    return [r for r in records if not r.error]


def _write_rows(rows: list[Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


# --------------------------------------------------------------------------- stages
def run_model_checks() -> list[Row]:
    rows: list[Row] = list(run_checks())
    _write_rows(rows, RESULTS / "model_checks.csv")
    return rows


def run_validation(workers: int, parts: set[str]) -> list[Row]:
    vcfg = validation_configs(parts)
    recs = run_many([c for _, _, c in vcfg], workers=workers)
    rows: list[Row] = []
    for (p, l_loop, _), rec in zip(vcfg, recs, strict=True):
        m = rec.metrics
        e_on = m.get("on_e_on", float("nan")) * UJ
        e_off = m.get("off_e_off", float("nan")) * UJ
        row: Row = {
            "part": p.device.part,
            "temp": p.temp,
            "l_loop_nH": round(l_loop / NH, 6),
            "e_on_sim_uJ": e_on,
            "e_on_ds_uJ": p.e_on * UJ,
            "e_on_dev_pct": deviation_pct(e_on, p.e_on * UJ),
            "e_off_sim_uJ": e_off,
            "e_off_ds_uJ": p.e_off * UJ,
            "e_off_dev_pct": deviation_pct(e_off, p.e_off * UJ),
            "e_tot_sim_uJ": e_on + e_off,
            "e_tot_ds_uJ": (p.e_on + p.e_off) * UJ,
            "e_tot_dev_pct": deviation_pct(e_on + e_off, (p.e_on + p.e_off) * UJ),
            "error": rec.error,
        }
        for k_sim in ("on_td_on", "on_tr", "off_td_off", "off_tf"):
            row[f"{k_sim}_sim_ns"] = m.get(k_sim, float("nan")) * NS
            row[f"{k_sim}_v_sim_ns"] = m.get(f"{k_sim}_v", float("nan")) * NS
        for k_ds in ("td_on", "tr", "td_off", "tf"):
            ref = getattr(p, k_ds)
            row[f"{k_ds}_ds_ns"] = ref * NS if ref is not None else float("nan")
        rows.append(row)
    _write_rows(rows, RESULTS / "validation.csv")
    write_csv(recs, RESULTS / "validation_runs.csv")
    return rows


def run_convergence() -> list[Row]:
    rows: list[Row] = []
    for cfg in convergence_configs():
        row: Row = {"cshunt_F": cfg.cshunt}
        try:
            m = metrics_from(simulate(cfg, timeout_s=300))
            row.update(
                e_on_uJ=m["on_e_on"] * UJ,
                e_off_uJ=m["off_e_off"] * UJ,
                tr_ns=m["on_tr"] * NS,
                di_dt_on_A_per_ns=m["on_di_dt"] * ANS,
                error="",
            )
        except Exception as exc:  # a non-converging setting is a result, not a crash
            row.update(error=type(exc).__name__)
        rows.append(row)
    _write_rows(rows, RESULTS / "convergence.csv")
    return rows


def run_sweeps(workers: int) -> Sweeps:
    out: Sweeps = {}
    for sw in sweeps():
        recs = run_many(list(sw.configs), workers=workers)
        write_csv(recs, RESULTS / f"sweep_{sw.name}.csv")
        out[sw.name] = (sw, recs)
    return out


def run_error_budget(val_rows: list[Row], conv_rows: list[Row]) -> list[Row]:
    """How far each uncertain input moves Eon / Eoff at the nominal point.

    Sources: re-simulated variants (driver resistance, gate-loop inductance),
    re-extraction of the nominal waveforms (integration windows, 1 ns current-probe
    deskew), and runs that already exist (loop-inductance bracket, cshunt plateau).
    """
    sim = simulate_with_retry(NOMINAL)
    ref = metrics_from(sim)
    e_on0, e_off0 = ref["on_e_on"] * UJ, ref["off_e_off"] * UJ

    def row(source: str, method: str, e_on: float, e_off: float) -> Row:
        return {
            "source": source,
            "method": method,
            "e_on_uJ": e_on,
            "e_on_change_pct": deviation_pct(e_on, e_on0),
            "e_off_uJ": e_off,
            "e_off_change_pct": deviation_pct(e_off, e_off0),
            "e_tot_change_pct": deviation_pct(e_on + e_off, e_on0 + e_off0),
        }

    rows = [row("Nominal set-up (reference)", "simulated", e_on0, e_off0)]
    for name, cfg in budget_configs():
        m = metrics_from(simulate_with_retry(cfg))
        rows.append(row(name, "simulated", m["on_e_on"] * UJ, m["off_e_off"] * UJ))
    part = NOMINAL.device.part
    for v in val_rows:
        if v["part"] == part and v["l_loop_nH"] != round(NOMINAL.l_loop / NH, 6) and not v["error"]:
            rows.append(
                row(
                    f"External loop inductance {v['l_loop_nH']:g} nH",
                    "simulated (validation bracket)",
                    v["e_on_sim_uJ"],
                    v["e_off_sim_uJ"],
                )
            )
    for dt in DESKEW_VARIANTS:
        m = metrics_from(sim, id_delay=dt)
        word = "late" if dt > 0 else "early"
        rows.append(
            row(
                f"Current probe {abs(dt) * NS:g} ns {word} (deskew error)",
                "re-extracted",
                m["on_e_on"] * UJ,
                m["off_e_off"] * UJ,
            )
        )
    for name, th in WINDOW_VARIANTS:
        m = metrics_from(sim, th=th)
        rows.append(row(name, "re-extracted", m["on_e_on"] * UJ, m["off_e_off"] * UJ))
    for c in conv_rows:
        if c["cshunt_F"] in CSHUNT_LADDER and c["cshunt_F"] != NOMINAL.cshunt and not c["error"]:
            rows.append(
                row(
                    f"cshunt {c['cshunt_F']:.0e} F instead of {NOMINAL.cshunt:.0e} F",
                    "simulated (convergence study)",
                    c["e_on_uJ"],
                    c["e_off_uJ"],
                )
            )
    _write_rows(rows, RESULTS / "error_budget.csv")
    return rows


# --------------------------------------------------------------------------- figures + summary
def _nominal(summary: Row) -> dict[str, float]:
    sim = simulate_with_retry(NOMINAL)
    w = sim.waveforms()
    m = metrics_from(sim)
    c = NOMINAL
    off = extract_turn_off(w, c.t_turn_off, c.vbus, m["i_off"], c.vgs_on, c.vgs_off)
    on = extract_turn_on(w, c.t_turn_on, c.vbus, m["i_on"], c.vgs_on, c.vgs_off)
    plots.switching_waveforms(
        w,
        c.t_turn_off,
        c.t_turn_on,
        off,
        on,
        f"{c.device.part} double-pulse test: {NOMINAL_TITLE}",
        FIGURES / "waveforms_nominal.png",
    )
    summary["nominal"] = dict(m)
    return m


def _convergence(conv_rows: list[Row], summary: Row) -> None:
    good = [r for r in conv_rows if not r["error"]]
    cs = [float(r["cshunt_F"]) for r in good]

    def ys(key: str) -> list[float]:
        return [float(r[key]) for r in good]

    x = "cshunt (F)"
    plots.panel_grid(
        [
            [
                Panel(cs, x, "Eon (µJ)", [("Eon", ys("e_on_uJ"))], xlog=True),
                Panel(cs, x, "Current rise time tr (ns)", [("tr", ys("tr_ns"))], xlog=True),
                Panel(cs, x, "Eoff (µJ)", [("Eoff", ys("e_off_uJ"))], xlog=True),
            ]
        ],
        "Convergence aid: results vs LTspice cshunt (nominal point)",
        FIGURES / "convergence_cshunt.png",
    )
    ref = next(r for r in good if r["cshunt_F"] == NOMINAL.cshunt)
    worst = max(good, key=lambda r: float(r["e_on_uJ"]))
    plateau = [r for r in good if r["cshunt_F"] in CSHUNT_LADDER]
    summary["convergence"] = conv_rows
    summary["convergence_summary"] = {
        "worst_cshunt_F": worst["cshunt_F"],
        "worst_e_on_ratio": worst["e_on_uJ"] / ref["e_on_uJ"],
        "ladder_F": list(CSHUNT_LADDER),
        "ladder_max_e_on_change_pct": max(
            abs(deviation_pct(r["e_on_uJ"], ref["e_on_uJ"])) for r in plateau
        ),
        "ladder_max_e_off_change_pct": max(
            abs(deviation_pct(r["e_off_uJ"], ref["e_off_uJ"])) for r in plateau
        ),
    }


def _validation(val_rows: list[Row], summary: Row) -> None:
    rows = []
    for p in DATASHEET_POINTS:
        sel = [
            r
            for r in val_rows
            if r["part"] == p.device.part and r["temp"] == p.temp and not r["error"]
        ]
        if not sel:
            continue
        ls = [float(r["l_loop_nH"]) for r in sel]
        x = "External loop inductance (nH)"
        ds = "datasheet typ."
        rows.append(
            [
                Panel(
                    ls,
                    x,
                    "Eon (µJ)",
                    [("simulated", [r["e_on_sim_uJ"] for r in sel])],
                    (p.e_on * UJ, ds),
                ),
                Panel(
                    ls,
                    x,
                    "Eoff (µJ)",
                    [("simulated", [r["e_off_sim_uJ"] for r in sel])],
                    (p.e_off * UJ, ds),
                ),
                Panel(
                    ls,
                    x,
                    "Eon + Eoff (µJ)",
                    [("simulated", [r["e_tot_sim_uJ"] for r in sel])],
                    ((p.e_on + p.e_off) * UJ, ds),
                ),
            ]
        )
    p0 = DATASHEET_POINTS[0]
    plots.panel_grid(
        rows,
        f"{p0.device.part} vs datasheet: {p0.vbus:.0f} V, {p0.i_load:.0f} A, "
        f"{p0.rg:g} Ω, {p0.temp:.0f} °C (datasheet loop L not published)",
        FIGURES / "validation.png",
    )
    summary["validation"] = val_rows


def _rg_tradeoff(sw: Sweeps, summary: Row) -> None:
    dev = NOMINAL.device
    v_lim = V_PEAK_LIMIT_FRACTION * dev.vds_rating
    lim_v = (v_lim, f"{V_PEAK_LIMIT_FRACTION:.0%} of V(BR)DSS")
    lim_dv = (DVDT_LIMIT * VNS, "dv/dt limit")
    lim_th = (dev.vth_min, "VGS(th) min")
    ok_on, ok_off = _ok(sw["rg_on"][1]), _ok(sw["rg_off"][1])
    x_on = [r.cfg.rg_on for r in ok_on]
    x_off = [r.cfg.rg_off for r in ok_off]
    # Peak dv/dt (1 ns window) is used for the EMI limit: the turn-on vDS fall has a slow
    # L*di/dt part and a fast diode-recovery collapse, so a 20-80 % average misleads.
    dv_on = _col(ok_on, "on_dv_dt_max", VNS)
    dv_off = _col(ok_off, "off_dv_dt_max", VNS)
    # Rg,on does not change Eoff, so its cost is Eon + Err. Rg,off also sets the
    # free-wheeler's hold-off impedance (same driver), so its cost is the whole cycle.
    cost_on = [a + b for a, b in zip(_col(ok_on, "on_e_on"), _col(ok_on, "on_e_rr"), strict=True)]
    cost_off = _col(ok_off, "e_cycle")
    fw_pk_on, dut_pk_off = _col(ok_on, "on_vds_hs_peak"), _col(ok_off, "off_vds_peak")
    xo, xf = "Rg,on (Ω)", "Rg,off (Ω)"
    plots.panel_grid(
        [
            [
                Panel(
                    x_on,
                    xo,
                    "Energy (µJ)",
                    [
                        ("Eon", _col(ok_on, "on_e_on", UJ)),
                        ("Err", _col(ok_on, "on_e_rr", UJ)),
                        ("Eon + Err", [c * UJ for c in cost_on]),
                    ],
                    xlog=True,
                    title="Turn-on sweep (Rg,off = 4.5 Ω)",
                ),
                Panel(x_on, xo, "Peak |dv/dt| at turn-on (V/ns)", [("dv/dt", dv_on)], lim_dv, True),
                Panel(x_on, xo, "Free-wheel peak vDS (V)", [("peak", fw_pk_on)], lim_v, True),
                Panel(
                    x_on,
                    xo,
                    "Free-wheel vGS peak (V)",
                    [("peak", _col(ok_on, "on_vgs_hs_max"))],
                    lim_th,
                    True,
                ),
            ],
            [
                Panel(
                    x_off,
                    xf,
                    "Energy (µJ)",
                    [
                        ("Eoff", _col(ok_off, "off_e_off", UJ)),
                        (
                            "Eon + Err",
                            [
                                (a + b) * UJ
                                for a, b in zip(
                                    _col(ok_off, "on_e_on"), _col(ok_off, "on_e_rr"), strict=True
                                )
                            ],
                        ),
                        ("Eon + Eoff + Err", [c * UJ for c in cost_off]),
                    ],
                    xlog=True,
                    title="Turn-off sweep (Rg,on = 4.5 Ω)",
                ),
                Panel(
                    x_off, xf, "Peak |dv/dt| at turn-off (V/ns)", [("dv/dt", dv_off)], lim_dv, True
                ),
                Panel(x_off, xf, "DUT peak vDS (V)", [("peak", dut_pk_off)], lim_v, True),
                Panel(
                    x_off,
                    xf,
                    "Free-wheel vGS peak (V)",
                    [("peak", _col(ok_off, "on_vgs_hs_max"))],
                    lim_th,
                    True,
                ),
            ],
        ],
        f"{dev.part}: gate-resistor trade-off at 800 V, 40 A, 30 nH",
        FIGURES / "rg_tradeoff.png",
    )
    ok_dv_on = [v <= DVDT_LIMIT * VNS for v in dv_on]
    ok_pk_on = [v <= v_lim for v in fw_pk_on]
    ok_dv_off = [v <= DVDT_LIMIT * VNS for v in dv_off]
    ok_pk_off = [v <= v_lim for v in dut_pk_off]
    i_on_both = best_under_constraints(x_on, cost_on, [ok_dv_on, ok_pk_on])
    i_on_pk = best_under_constraints(x_on, cost_on, [ok_pk_on])
    i_off_both = best_under_constraints(x_off, cost_off, [ok_dv_off, ok_pk_off])
    i_dvmin = int(np.argmin(dv_on))
    # Rg,off that would reproduce the datasheet Eoff (linear interpolation in the sweep).
    ds_eoff = next(p.e_off for p in DATASHEET_POINTS if p.device.part == NOMINAL.device.part)
    on_keys = ("on_e_on", "on_e_rr", "on_dv_dt", "on_dv_dt_max", "on_vds_hs_peak", "on_vgs_hs_max")
    off_keys = (
        "off_e_off",
        "on_e_on",
        "on_e_rr",
        "on_q_rr",
        "e_cycle",
        "off_dv_dt",
        "off_dv_dt_max",
        "off_vds_peak",
        "on_vgs_hs_max",
        "off_vgs_hs_min",
    )
    nom_off = next(r for r in ok_off if r.cfg.rg_off == NOMINAL.rg_off)
    summary["rg_off_matching_datasheet_eoff"] = interp_at(ds_eoff, _col(ok_off, "off_e_off"), x_off)
    summary["rg"] = {
        "rg_on_best_both": x_on[i_on_both] if i_on_both is not None else None,
        "rg_on_best_peak_only": x_on[i_on_pk] if i_on_pk is not None else None,
        "rg_on_lowest_dvdt": dv_on[i_dvmin],
        "rg_on_lowest_dvdt_at": x_on[i_dvmin],
        "rg_off_best_both": x_off[i_off_both] if i_off_both is not None else None,
        "nominal_e_cycle": nom_off.metrics["e_cycle"],
        "nominal_e_off": nom_off.metrics["off_e_off"],
        "rule_on": "lowest Eon + Err with every limit met",
        "rule_off": "lowest Eon + Eoff + Err with every limit met",
        "rg_on_rows": [
            {"rg": x, **{k: r.metrics[k] for k in on_keys}}
            for x, r in zip(x_on, ok_on, strict=True)
        ],
        "rg_off_rows": [
            {"rg": x, **{k: r.metrics[k] for k in off_keys}}
            for x, r in zip(x_off, ok_off, strict=True)
        ],
    }


def _stray_budget(sw: Sweeps, summary: Row) -> None:
    dev = NOMINAL.device
    v_lim = V_PEAK_LIMIT_FRACTION * dev.vds_rating
    ok = _ok(sw["l_loop"][1])
    xl = [round(r.cfg.l_loop / NH, 6) for r in ok]
    pk_off, pk_on = _col(ok, "off_vds_peak"), _col(ok, "on_vds_hs_peak")
    x = "External loop inductance (nH)"
    plots.panel_grid(
        [
            [
                Panel(
                    xl,
                    x,
                    "Peak vDS (V)",
                    [("DUT at turn-off", pk_off), ("Free-wheel at turn-on", pk_on)],
                    (v_lim, f"{V_PEAK_LIMIT_FRACTION:.0%} of V(BR)DSS"),
                ),
                Panel(
                    xl,
                    x,
                    "Energy (µJ)",
                    [("Eon", _col(ok, "on_e_on", UJ)), ("Eoff", _col(ok, "off_e_off", UJ))],
                ),
                Panel(
                    xl,
                    x,
                    "Turn-off ringing frequency (MHz)",
                    [("f_ring", _col(ok, "off_f_ring", 1e-6))],
                ),
            ]
        ],
        f"{dev.part}: stray-inductance budget at 800 V, 40 A, Rg = 4.5 Ω",
        FIGURES / "stray_budget.png",
    )
    l_off = max_x_below(xl, pk_off, v_lim)
    l_on = max_x_below(xl, pk_on, v_lim)
    summary["stray_budget"] = {
        "l_max_nH_turn_off": l_off,
        "l_max_nH_turn_on_freewheel": l_on,
        "l_max_nH_both": combined_limit(l_off, l_on),
        "rows": [
            {"l_nH": a, "pk_off": b, "pk_on": c, "f_ring_MHz": r.metrics["off_f_ring"] * 1e-6}
            for a, b, c, r in zip(xl, pk_off, pk_on, ok, strict=True)
        ],
    }


def _kelvin(sw: Sweeps, kel: dict[str, float], summary: Row) -> None:
    ok = _ok(sw["l_cs"][1])
    # Only gate-stable runs carry a meaningful loss value; oscillating ones are listed
    # in the results table instead of being plotted as if they were valid points.
    st = [r for r in ok if r.metrics.get("gate_ok", 0.0) == 1.0]
    xs = [round(r.cfg.l_cs / NH, 6) for r in st]
    x = "Extra common-source inductance (nH)"
    kel_line = "Kelvin drive"
    plots.panel_grid(
        [
            [
                Panel(
                    xs,
                    x,
                    "Eon (µJ)",
                    [("non-Kelvin", _col(st, "on_e_on", UJ))],
                    (kel["on_e_on"] * UJ, kel_line),
                ),
                Panel(
                    xs,
                    x,
                    "Eoff (µJ)",
                    [("non-Kelvin", _col(st, "off_e_off", UJ))],
                    (kel["off_e_off"] * UJ, kel_line),
                ),
                Panel(
                    xs,
                    x,
                    "Eon + Eoff + Err (µJ)",
                    [("non-Kelvin", _col(st, "e_cycle", UJ))],
                    (kel["e_cycle"] * UJ, kel_line),
                ),
            ]
        ],
        "Kelvin vs non-Kelvin gate return (800 V, 40 A, Rg = 4.5 Ω, 30 nH)",
        FIGURES / "kelvin_benefit.png",
    )
    # Fairer view of the turn-on: compare with a Kelvin-driven DUT slowed down by a larger
    # Rg,on to the same turn-on di/dt (interpolated in the Kelvin Rg,on sweep). Equal
    # di/dt means equal free-wheel recovery stress, the quantity that limits the layout.
    pkg = next(r for r in ok if r.cfg.l_cs == 0.0)
    k_on = _ok(sw["rg_on"][1])
    didt = _col(k_on, "on_di_dt")
    target = pkg.metrics["on_di_dt"]
    k_eon = interp_at(target, didt, _col(k_on, "on_e_on"))
    k_err = interp_at(target, didt, _col(k_on, "on_e_rr"))
    summary["kelvin"] = {
        "kelvin_e_on_uJ": kel["on_e_on"] * UJ,
        "kelvin_e_off_uJ": kel["off_e_off"] * UJ,
        "kelvin_e_ts_uJ": kel["e_ts"] * UJ,
        "kelvin_e_cycle_uJ": kel["e_cycle"] * UJ,
        "non_kelvin_e_on_increase_pct": deviation_pct(pkg.metrics["on_e_on"], kel["on_e_on"]),
        "equal_didt": {
            "di_dt_A_per_ns": target * ANS,
            "kelvin_rg_on": interp_at(target, didt, [r.cfg.rg_on for r in k_on]),
            "kelvin_e_on_uJ": k_eon * UJ,
            "kelvin_e_on_err_uJ": (k_eon + k_err) * UJ,
            "non_kelvin_e_on_uJ": pkg.metrics["on_e_on"] * UJ,
            "non_kelvin_e_on_err_uJ": (pkg.metrics["on_e_on"] + pkg.metrics["on_e_rr"]) * UJ,
            "e_on_increase_pct": deviation_pct(pkg.metrics["on_e_on"], k_eon),
            "e_on_err_increase_pct": deviation_pct(
                pkg.metrics["on_e_on"] + pkg.metrics["on_e_rr"], k_eon + k_err
            ),
        },
        "rows": [
            {
                "l_cs_nH": round(r.cfg.l_cs / NH, 6),
                "e_on_uJ": r.metrics["on_e_on"] * UJ,
                "e_off_uJ": r.metrics["off_e_off"] * UJ,
                "e_ts_uJ": r.metrics["e_ts"] * UJ,
                "e_rr_uJ": r.metrics["on_e_rr"] * UJ,
                "e_cycle_uJ": r.metrics["e_cycle"] * UJ,
                "e_ts_increase_pct": deviation_pct(r.metrics["e_ts"], kel["e_ts"]),
                "e_cycle_increase_pct": deviation_pct(r.metrics["e_cycle"], kel["e_cycle"]),
                "di_dt_on_A_per_ns": r.metrics["on_di_dt"] * ANS,
                "vgs_max": max(r.metrics["on_vgs_max"], r.metrics["off_vgs_max"]),
                "vgs_min": min(r.metrics["on_vgs_min"], r.metrics["off_vgs_min"]),
                "gate_ok": bool(r.metrics["gate_ok"]),
            }
            for r in ok
        ],
    }


def _operating_point(sw: Sweeps, summary: Row) -> None:
    panels = []
    for name, label in (
        ("i_load", "Switched current at turn-off (A)"),
        ("vbus", "DC-link voltage (V)"),
        ("temp", "Temperature (°C)"),
    ):
        s, recs = sw[name]
        ok = _ok(recs)
        xs = [r.metrics["i_off"] if name == "i_load" else getattr(r.cfg, s.param) for r in ok]
        series = [
            ("Eon", _col(ok, "on_e_on", UJ)),
            ("Eoff", _col(ok, "off_e_off", UJ)),
            ("Err", _col(ok, "on_e_rr", UJ)),
        ]
        panels.append(Panel(xs, label, "Energy (µJ)", series))
        summary[f"sweep_{name}"] = [
            {
                "x": x,
                "e_on_uJ": r.metrics["on_e_on"] * UJ,
                "e_off_uJ": r.metrics["off_e_off"] * UJ,
                "e_rr_uJ": r.metrics["on_e_rr"] * UJ,
            }
            for x, r in zip(xs, ok, strict=True)
        ]
    plots.panel_grid(
        [panels],
        f"{NOMINAL.device.part}: switching energy vs operating point (Rg = 4.5 Ω, 30 nH)",
        FIGURES / "operating_point.png",
    )


def make_outputs(val_rows: list[Row], conv_rows: list[Row], sw: Sweeps, budget: list[Row]) -> Row:
    dev = NOMINAL.device
    summary: Row = {
        "limits": {
            "v_peak_limit_V": V_PEAK_LIMIT_FRACTION * dev.vds_rating,
            "dv_dt_limit_V_per_ns": DVDT_LIMIT * VNS,
            "vth_min_V": dev.vth_min,
            "vgs_abs_min_V": dev.vgs_abs_min,
        }
    }
    m_nom = _nominal(summary)
    _convergence(conv_rows, summary)
    _validation(val_rows, summary)
    _rg_tradeoff(sw, summary)
    _stray_budget(sw, summary)
    _kelvin(sw, m_nom, summary)
    _operating_point(sw, summary)
    summary["vgs_off"] = [
        {
            "vgs_off": r.cfg.vgs_off,
            "vgs_hs_max": r.metrics["on_vgs_hs_max"],
            "margin": r.metrics["vgs_hs_margin"],
            "e_on_uJ": r.metrics["on_e_on"] * UJ,
            "e_off_uJ": r.metrics["off_e_off"] * UJ,
            "e_rr_uJ": r.metrics["on_e_rr"] * UJ,
            "vgs_hs_min": r.metrics["off_vgs_hs_min"],
        }
        for r in _ok(sw["vgs_off"][1])
    ]
    summary["error_budget"] = budget
    all_runs = [r for _, recs in sw.values() for r in recs]
    summary["failed_runs"] = [r.cfg.key() for r in all_runs if r.error]
    summary["cshunt_used"] = {
        f"{k:.0e}": n
        for k, n in sorted(Counter(r.metrics["cshunt"] for r in _ok(all_runs)).items())
    }
    return summary


def _jsonable(o: object) -> object:
    if isinstance(o, float | np.floating):
        return None if math.isnan(float(o)) or math.isinf(float(o)) else float(o)
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_jsonable(v) for v in o]
    return o


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the full DPT study")
    ap.add_argument("--workers", type=int, default=1, help="parallel LTspice runs (default 1)")
    args = ap.parse_args(argv)
    if fetch_main(["--check"]) != 0:
        print("onsemi model files missing - see the messages above.", file=sys.stderr)
        return 1
    checks = run_model_checks()
    parts = {str(r["part"]) for r in checks if r["passed"]}
    val_rows = run_validation(args.workers, parts)
    conv_rows = run_convergence()
    sw = run_sweeps(args.workers)
    budget = run_error_budget(val_rows, conv_rows)
    summary = make_outputs(val_rows, conv_rows, sw, budget)
    summary["model_checks"] = checks
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    report.update_blocks(README, report.readme_blocks(summary))
    RESULTS_MD.write_text(report.results_markdown(summary), encoding="utf-8", newline="\n")
    print(f"done: results in {RESULTS.name}/, figures in docs/figures/, report in docs/results.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
