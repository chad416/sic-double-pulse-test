from __future__ import annotations

import math

import pytest

from dpt.analysis import (
    best_under_constraints,
    combined_limit,
    deviation_pct,
    interp_at,
    max_x_below,
)


def test_max_x_below_interpolates_first_crossing() -> None:
    assert max_x_below([0, 10, 20], [100, 150, 250], 200) == pytest.approx(15.0)
    assert max_x_below([0, 10, 20], [100, 150, 160], 200) == 20.0
    assert math.isnan(max_x_below([0, 10], [300, 400], 200))
    with pytest.raises(ValueError):
        max_x_below([0, 0], [1, 2], 3)


def test_best_under_constraints() -> None:
    xs = [1, 2, 3, 4]
    cost = [1.0, 2.0, 3.0, 4.0]
    assert best_under_constraints(xs, cost, [[False, True, True, True]]) == 1
    assert (
        best_under_constraints(xs, cost, [[False, True, True, True], [True, False, True, True]])
        == 2
    )
    assert best_under_constraints(xs, cost, [[False] * 4]) is None


def test_deviation_pct() -> None:
    assert deviation_pct(450.0, 500.0) == pytest.approx(-10.0)


def test_combined_limit_is_min_of_individual_limits() -> None:
    off = max_x_below([10, 20, 30], [890, 940, 981], 960)
    fw = max_x_below([10, 20, 30], [877, 1013, 1145], 960)
    assert combined_limit(off, fw) == pytest.approx(fw)
    assert math.isnan(combined_limit(off, float("nan")))


def test_interp_at_handles_decreasing_x_and_range() -> None:
    assert interp_at(5.0, [6.0, 4.0], [100.0, 200.0]) == pytest.approx(150.0)
    assert math.isnan(interp_at(7.0, [6.0, 4.0], [100.0, 200.0]))
    with pytest.raises(ValueError):
        interp_at(1.0, [1.0, 1.0], [0.0, 1.0])
