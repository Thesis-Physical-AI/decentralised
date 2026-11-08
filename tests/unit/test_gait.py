import math

import pytest

from puppy_agent.gait import LEGS, MAX_STRIDE, STAND_HEIGHT, TrotGait, leg_fk, leg_ik, side_strides


@pytest.mark.parametrize("y,z", [(0.0, -0.10), (0.02, -0.10), (-0.02, -0.08), (0.03, -0.12), (0.0, -0.05)])
def test_ik_round_trip(y, z):
    hip, knee = leg_ik(y, z)
    assert leg_fk(hip, knee) == pytest.approx((y, z), abs=1e-6)


def test_ik_clamps_unreachable_targets():
    hip, knee = leg_ik(0.0, -1.0)  # beyond THIGH + SHIN
    assert all(math.isfinite(v) for v in (hip, knee))


def test_strides_follow_command_and_clamp():
    left, right = side_strides(0.1, 0.0)
    assert left == pytest.approx(right) and left > 0
    left, right = side_strides(0.0, 0.5)
    assert left < 0 < right  # turning left: right side pushes forward
    assert max(abs(s) for s in side_strides(10.0, 10.0)) == MAX_STRIDE


def test_zero_command_stands():
    feet = TrotGait().update(0.01, 0.0, 0.0)
    assert feet == {leg: (0.0, -STAND_HEIGHT) for leg in LEGS}


def test_diagonal_pairs_are_in_phase():
    gait = TrotGait()
    for _ in range(37):
        feet = gait.update(0.01, 0.1, 0.0)
    assert feet["lf"] == pytest.approx(feet["rb"]) and feet["rf"] == pytest.approx(feet["lb"])
    assert feet["lf"] != pytest.approx(feet["rf"])
