import math

import pytest

from puppy_agent.navigation import ABORTED, CANCELED, SUCCEEDED, Pose2D, SimpleNavigator

from .fakes import KinematicRobot, StuckRobot


def run(nav, robot, limit_s=120.0):
    while nav.result() is None and robot.now() < limit_s:
        nav.spin_once()
        robot.advance(0.05)
    return nav.result()


@pytest.mark.parametrize("goal", [Pose2D(1.0, 0.0, 0.0), Pose2D(-0.5, 1.2, math.pi / 2), Pose2D(0.3, -0.8, None)])
def test_reaches_goal_pose(goal):
    robot = KinematicRobot()
    nav = SimpleNavigator(robot)
    nav.go_to(goal)
    assert run(nav, robot) == SUCCEEDED
    pose = robot.odom()
    assert math.hypot(pose.x - goal.x, pose.y - goal.y) < nav.pos_tol
    if goal.yaw is not None:
        assert abs((pose.yaw - goal.yaw + math.pi) % (2 * math.pi) - math.pi) < nav.yaw_tol
    assert (robot.vx, robot.wz) == (0.0, 0.0)


def test_aborts_when_no_progress():
    robot = StuckRobot()
    nav = SimpleNavigator(robot, stuck_window_s=2.0)
    nav.go_to(Pose2D(2.0, 0.0))
    assert run(nav, robot) == ABORTED


def test_spin_turns_requested_angle():
    robot = KinematicRobot(yaw=3.0)  # crosses the +-pi wrap
    nav = SimpleNavigator(robot)
    nav.spin(math.pi / 2)
    assert run(nav, robot) == SUCCEEDED
    turned = (robot.odom().yaw - 3.0 + math.pi) % (2 * math.pi) - math.pi
    assert turned == pytest.approx(math.pi / 2, abs=nav.yaw_tol)


def test_cancel_stops_robot():
    robot = KinematicRobot()
    nav = SimpleNavigator(robot)
    nav.go_to(Pose2D(5.0, 0.0))
    nav.spin_once()
    nav.cancel()
    assert nav.result() == CANCELED and (robot.vx, robot.wz) == (0.0, 0.0)
