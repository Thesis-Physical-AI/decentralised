"""Live selector: closed action menu, validation of model output, and the keep-moving-while-thinking loop
on the live camera feed."""
import math

import pytest

from puppy_agent import contracts as c
from puppy_agent.executor import ExecutorConfig, StepExecutor
from puppy_agent.navigation import SUCCEEDED, Pose2D, SimpleNavigator
from puppy_agent.live_feed import FrameAnalysis
from puppy_agent.selector import MotionLimits, motion_for, validate
from puppy_agent.semantic_map import SemanticMap

from .fakes import KinematicRobot
from .test_executor import WORLD, step
from .test_visual_approach import HFOV, ground_truth

def live_executor(robot, perceiver, events=None, **config):
    config.setdefault("feed_mode", "lockstep")  # deterministic; async is covered in test_live_feed.py
    cfg = ExecutorConfig(approach="live", max_queries=150, camera_hfov_rad=HFOV, **config)
    return StepExecutor(robot, SimpleNavigator(robot), perceiver, SemanticMap.from_world(WORLD),
                        camera=robot.camera, safety_check=robot.safety, config=cfg,
                        on_event=(events.append if events is not None else None))


class Scripted:
    """Plays a list of actions, then hands over to a fallback perceiver (or repeats the last action)."""

    def __init__(self, actions, latency_s=0.0, fallback=None, robot=None):
        self.actions, self.latency_s, self.fallback, self.robot = list(actions), latency_s, fallback, robot
        self.moving_when_asked = []

    def analyze(self, image, questions, entity):
        if self.robot is not None:
            self.moving_when_asked.append(self.robot.vx != 0.0 or self.robot.wz != 0.0)
        if self.actions:
            action = self.actions.pop(0) if len(self.actions) > 1 or self.fallback else self.actions[0]
            return FrameAnalysis("vlm", self.latency_s, {}, action=action, action_scores={action: 0.9},
                                 target=f"scripted {action}")
        analysis = self.fallback.analyze(image, questions, entity)
        analysis.latency_s = self.latency_s
        return analysis


def commands(events):
    return [e["action"] for e in events if e["event"] == "command"]


# -- menu and validation --------------------------------------------------------------------------
@pytest.mark.parametrize("raw", ["jump", "", None, 3, "forward please", "turn_left"])
def test_validate_rejects_anything_off_the_menu(raw):
    assert validate(raw) is None


def test_validate_normalises_menu_words():
    assert validate(" Forward ") == "forward"


def test_motions_are_bounded():
    limits, pose = MotionLimits(), Pose2D(1.0, 2.0, 0.3)
    for action in ("forward", "left", "right"):
        goal = motion_for(action, pose, [], limits).goal
        assert math.hypot(goal.x - pose.x, goal.y - pose.y) <= max(limits.step_m, limits.veer_m) + 1e-9
    assert motion_for("left", pose, [], limits).goal.yaw > pose.yaw > motion_for("right", pose, [], limits).goal.yaw
    assert abs(motion_for("search", pose, [], limits).angle) == pytest.approx(limits.search_rad)
    assert motion_for("back", pose, [], limits).distance == limits.back_m
    assert motion_for("arrived", pose, [], limits).kind == "stop"


def test_search_turns_toward_the_side_last_seen():
    assert motion_for("search", Pose2D(0, 0, 0), ["left", "right"]).angle < 0
    assert motion_for("search", Pose2D(0, 0, 0), ["right", "left"]).angle > 0


def test_navigator_backs_up():
    robot = KinematicRobot()
    nav = SimpleNavigator(robot)
    nav.back_up(0.2)
    while nav.result() is None:
        nav.spin_once()
        robot.advance(0.05)
    assert nav.result() == SUCCEEDED
    assert robot.pose.x == pytest.approx(-0.2, abs=0.05) and robot.pose.yaw == pytest.approx(0.0)


# -- live loop ------------------------------------------------------------------------------------
def test_live_approach_reaches_target():
    robot = KinematicRobot()
    result = live_executor(robot, ground_truth(robot)).execute(step("movetowards", {"target": "table"}), "puppypi_1")
    assert result.status == c.SUCCEEDED, result.reason
    assert math.hypot(1.5 - robot.pose.x, robot.pose.y) <= 0.6


def test_live_approach_searches_for_target_behind():
    robot, events = KinematicRobot(yaw=math.pi), []
    result = live_executor(robot, ground_truth(robot), events).execute(step("movetowards", {"target": "table"}),
                                                                       "puppypi_1")
    assert result.status == c.SUCCEEDED, result.reason
    assert "search" in commands(events)


def test_robot_keeps_moving_while_the_model_thinks():
    robot = KinematicRobot()
    perceiver = Scripted([], latency_s=1.5, fallback=ground_truth(robot), robot=robot)
    result = live_executor(robot, perceiver).execute(step("movetowards", {"target": "table"}), "puppypi_1")
    assert result.status == c.SUCCEEDED, result.reason
    assert sum(perceiver.moving_when_asked) >= len(perceiver.moving_when_asked) // 2


def test_off_menu_answers_are_never_executed():
    robot, events = KinematicRobot(), []
    perceiver = Scripted(["jump", "turn_left", "forward"], fallback=ground_truth(robot))
    executor = live_executor(robot, perceiver, events)
    result = executor.execute(step("movetowards", {"target": "table"}), "puppypi_1")
    assert result.status == c.SUCCEEDED, result.reason
    assert "jump" not in commands(events) and "turn_left" not in commands(events)
    assert sum(e["event"] == "rejected" for e in events) == 2


def test_stale_answers_are_discarded():
    robot, events = KinematicRobot(), []
    executor = live_executor(robot, Scripted(["forward"], latency_s=8.0), events, max_decision_age_s=6.0)
    result = executor.execute(step("movetowards", {"target": "table"}, timeout_s=60), "puppypi_1")
    assert (result.status, result.reason_code) == (c.TIMED_OUT, "TIMEOUT")
    assert commands(events) == [] and robot.pose.x == pytest.approx(0.0)


def test_arrival_claim_far_from_target_is_rejected_by_map_distance():
    robot, events = KinematicRobot(), []
    perceiver = Scripted(["arrived"], fallback=ground_truth(robot))
    result = live_executor(robot, perceiver, events, near_source="map").execute(
        step("movetowards", {"target": "table"}), "puppypi_1")
    assert result.status == c.SUCCEEDED, result.reason
    assert math.hypot(1.5 - robot.pose.x, robot.pose.y) <= 0.6
    assert any(e["event"] == "rejected" and "map distance" in e["reason"] for e in events)


def test_arrival_must_be_confirmed_on_a_still_image():
    robot, events = KinematicRobot(), []
    perceiver = Scripted(["arrived", "forward"], fallback=ground_truth(robot))
    result = live_executor(robot, perceiver, events).execute(step("movetowards", {"target": "table"}), "puppypi_1")
    assert result.status == c.SUCCEEDED, result.reason
    assert any(e["event"] == "rejected" and "still image" in e["reason"] for e in events)
    assert math.hypot(1.5 - robot.pose.x, robot.pose.y) <= 0.6


def test_target_never_seen_ends_blocked_after_a_full_turn():
    robot = KinematicRobot()
    result = live_executor(robot, Scripted(["search"])).execute(step("movetowards", {"target": "table"}), "puppypi_1")
    assert (result.status, result.reason_code) == (c.BLOCKED, "OBJECT_NOT_FOUND")


def test_decision_cap_bounds_a_model_that_never_arrives():
    robot = KinematicRobot(yaw=math.pi / 2)  # walks along +y, never toward the table
    executor = live_executor(robot, Scripted(["forward"]))
    executor.config.max_queries = 10
    result = executor.execute(step("movetowards", {"target": "table"}, timeout_s=600), "puppypi_1")
    assert (result.status, result.reason_code) == (c.FAILED, "DECISION_LIMIT")
