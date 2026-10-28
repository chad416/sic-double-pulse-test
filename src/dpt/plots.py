"""Figures for the README (static PNG, matplotlib).

Style: one y-axis per panel, thin 2 px lines, recessive grid, categorical colours
taken in fixed order from a colour-vision-deficiency-checked palette, every
series labelled (legend or direct label).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from dpt.extract import TurnOffResult, TurnOnResult, Waveforms

# Categorical slots 1-4 (blue, orange, aqua, yellow), validated adjacent-pair CVD-safe
# on the light surface. Slots 2-4 sit below 3:1 contrast -> always direct-labelled.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
WINDOW_FILL = "#86b6ef"  # sequential blue step 250, for integration windows

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": INK_2,
        "axes.labelcolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "lines.linewidth": 1.8,
        "font.family": "DejaVu Sans",
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _shade(ax: Axes, t0: float, t1: float, label: str | None = None) -> None:
    ax.axvspan(t0, t1, color=WINDOW_FILL, alpha=0.25, lw=0, label=label)


def switching_waveforms(
    w: Waveforms,
    t_off_cmd: float,
    t_on_cmd: float,
    off: TurnOffResult,
    on: TurnOnResult,
    title: str,
    path: Path,
    span: tuple[float, float] = (-20e-9, 180e-9),
) -> Path:
    """Turn-off and turn-on waveforms side by side with the integration windows shaded."""
    fig, axes = plt.subplots(3, 2, figsize=(10, 7.2), sharex="col")
    for col, (t_cmd, win, name) in enumerate(
        (
            (t_off_cmd, (off.t_start, off.t_end), "Turn-off"),
            (t_on_cmd, (on.t_start, on.t_end), "Turn-on"),
        )
    ):
        m = (w.t >= t_cmd + span[0]) & (w.t <= t_cmd + span[1])
        tn = (w.t[m] - t_cmd) * 1e9
        w0, w1 = ((win[0] - t_cmd) * 1e9, (win[1] - t_cmd) * 1e9)
        ax_v, ax_i, ax_g = axes[0, col], axes[1, col], axes[2, col]
        ax_v.plot(tn, w.vds[m], color=SERIES[0], label="DUT $v_{DS}$")
        ax_v.plot(tn, w.vds_hs[m], color=SERIES[1], label="free-wheel $v_{DS}$")
        ax_i.plot(tn, w.id[m], color=SERIES[0], label="DUT $i_D$")
        ax_i.plot(tn, w.i_hs[m], color=SERIES[1], label="free-wheel $i_D$")
        ax_g.plot(tn, w.vgs[m], color=SERIES[0], label="DUT $v_{GS}$ (Kelvin)")
        ax_g.plot(tn, w.vgs_hs[m], color=SERIES[1], label="free-wheel $v_{GS}$")
        for ax in (ax_v, ax_i, ax_g):
            _shade(ax, w0, w1, label="E integration window" if ax is ax_v else None)
        ax_v.set_title(name)
        ax_v.set_ylabel("Voltage (V)")
        ax_i.set_ylabel("Current (A)")
        ax_g.set_ylabel("Gate voltage (V)")
        ax_g.set_xlabel("Time from gate command (ns)")
        for ax in (ax_v, ax_i, ax_g):
            ax.legend(loc="best")
    fig.suptitle(title, fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    return _save(fig, path)


def sweep_panels(
    x: Sequence[float],
    panels: Sequence[tuple[str, Sequence[tuple[str, Sequence[float]]]]],
    xlabel: str,
    title: str,
    path: Path,
    xlog: bool = False,
    hlines: dict[int, tuple[float, str]] | None = None,
) -> Path:
    """Small multiples sharing one x-axis; each panel holds <= 3 labelled series."""
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.4))
    axes_list = [axes] if n == 1 else list(axes)
    xs = np.asarray(x, dtype=float)
    for k, (ax, (ylabel, series)) in enumerate(zip(axes_list, panels, strict=True)):
        for j, (name, ys) in enumerate(series):
            ya = np.asarray(ys, dtype=float)
            ax.plot(xs, ya, color=SERIES[j], marker="o", ms=4.5, label=name)
            if len(series) > 1:
                ax.annotate(
                    name,
                    (xs[-1], ya[-1]),
                    xytext=(4, 0),
                    textcoords="offset points",
                    fontsize=8,
                    color=INK_2,
                    va="center",
                )
        if hlines and k in hlines:
            yv, lab = hlines[k]
            ax.axhline(yv, color=INK_2, lw=1.0, ls="--")
            ax.annotate(
                lab, (xs[0], yv), xytext=(2, 3), textcoords="offset points", fontsize=8, color=INK_2
            )
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        if xlog:
            ax.set_xscale("log")
        if len(series) > 1:
            ax.legend(loc="best")
    fig.suptitle(title, fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    return _save(fig, path)
