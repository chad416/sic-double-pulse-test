from __future__ import annotations

import math

import pytest

from dpt.analysis import best_under_constraints, deviation_pct, max_x_below


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
