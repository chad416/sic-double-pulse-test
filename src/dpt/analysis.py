"""Engineering analysis on top of the sweep results (pure functions, unit-tested)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def max_x_below(xs: Sequence[float], ys: Sequence[float], limit: float) -> float:
    """Largest ``x`` for which the piecewise-linear curve ``y(x)`` stays <= ``limit``.

    ``xs`` must be increasing. Returns ``nan`` if the first point already exceeds the
    limit and ``xs[-1]`` if the whole curve is below it (the true budget is then
    larger than the swept range).
    """
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if np.any(np.diff(x) <= 0):
        raise ValueError("xs must be strictly increasing")
    if y[0] > limit:
        return float("nan")
    for k in range(1, len(x)):
        if y[k] > limit:
            # linear interpolation inside the first segment that crosses the limit
            return float(x[k - 1] + (limit - y[k - 1]) * (x[k] - x[k - 1]) / (y[k] - y[k - 1]))
    return float(x[-1])


def best_under_constraints(
    xs: Sequence[float],
    cost: Sequence[float],
    constraints: Sequence[Sequence[bool]],
) -> int | None:
    """Index of the lowest-``cost`` point that satisfies every constraint, or ``None``."""
    ok = np.ones(len(xs), dtype=bool)
    for c in constraints:
        ok &= np.asarray(c, dtype=bool)
    if not ok.any():
        return None
    idx = np.nonzero(ok)[0]
    return int(idx[np.argmin(np.asarray(cost, dtype=float)[idx])])


def deviation_pct(sim: float, ref: float) -> float:
    """Signed deviation of ``sim`` from ``ref`` in percent."""
    return 100.0 * (sim - ref) / ref


def combined_limit(*limits: float) -> float:
    """Budget that satisfies several independent limits: the smallest of them.

    ``nan`` (a limit already violated at the first swept point) propagates. Taking the
    minimum of the individual interpolated limits avoids the bias of interpolating the
    point-wise maximum of two curves, which kinks between swept points.
    """
    if any(math.isnan(x) for x in limits):
        return float("nan")
    return min(limits)


def interp_at(x: float, xs: Sequence[float], ys: Sequence[float]) -> float:
    """Linear interpolation of ``y(x)`` for a monotonic but unsorted ``xs``.

    Returns ``nan`` outside the sampled range instead of extrapolating.
    """
    xa = np.asarray(xs, dtype=float)
    ya = np.asarray(ys, dtype=float)
    order = np.argsort(xa)
    xa, ya = xa[order], ya[order]
    if np.any(np.diff(xa) <= 0):
        raise ValueError("xs must be strictly monotonic")
    if not xa[0] <= x <= xa[-1]:
        return float("nan")
    return float(np.interp(x, xa, ya))
