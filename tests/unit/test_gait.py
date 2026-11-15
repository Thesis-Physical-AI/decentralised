import math

import pytest

from puppy_agent.gait import LEGS, MAX_STRIDE, TrotGait, leg_fk, leg_ik, side_strides, stand_feet


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
    assert feet == stand_feet()


def test_diagonal_pairs_are_in_phase():
    gait = TrotGait()
    for _ in range(37):
        feet = gait.update(0.01, 0.1, 0.0)
    lift = {leg: feet[leg][1] - stand_feet()[leg][1] for leg in LEGS}  # height above that leg's ground
    assert feet["lf"][0] == pytest.approx(feet["rb"][0]) and lift["lf"] == pytest.approx(lift["rb"])
    assert feet["rf"][0] == pytest.approx(feet["lb"][0]) and lift["rf"] == pytest.approx(lift["lb"])
    assert feet["lf"][0] != pytest.approx(feet["rf"][0])


def test_saturated_command_keeps_turn_ratio():
    left, right = side_strides(10.0, 2.0)
    assert max(abs(left), abs(right)) == pytest.approx(MAX_STRIDE)
    assert 0 < left < right  # still turning left, not two equal clamped strides


def _max_jump(commands, dt=1 / 120):
    """Largest foot move in one physics step over a list of (steps, vx, wz) commands."""
    gait, prev, worst = TrotGait(), stand_feet(), 0.0
    for steps, vx, wz in commands:
        for _ in range(steps):
            feet = gait.update(dt, vx, wz)
            worst = max(worst, max(math.dist(feet[leg], prev[leg]) for leg in LEGS))
            prev = feet
    return worst, gait, prev


def test_feet_never_jump_on_start_stop_or_command_changes():
    commands = [(60, 0.12, 0.0), (30, 0.0, 0.0), (45, 0.0, 0.8), (20, 0.15, -0.5), (90, 0.0, 0.0), (40, -0.08, 0.0)]
    worst, _, _ = _max_jump(commands)
    assert worst < 0.005  # 5 mm per 1/120 s step is swing speed; the old gait jumped by a half stride (~20 mm)


def test_zero_command_steps_back_to_standing():
    worst, gait, feet = _max_jump([(100, 0.15, 0.3), (240, 0.0, 0.0)])
    assert not gait.walking
    assert feet == stand_feet()
