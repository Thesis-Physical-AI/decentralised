"""VLM-picks-the-goal approach, onboard latency charging, and the direction -> goal conversion."""
import math

import pytest

from puppy_agent import contracts as c
from puppy_agent.executor import ExecutorConfig, StepExecutor
from puppy_agent.navigation import Pose2D, SimpleNavigator
from puppy_agent.perception import Detection, GroundTruthPerceiver, direction_bearing
from puppy_agent.semantic_map import SemanticMap
from puppy_agent.visual_goal import GroundGoalAdapter

from .fakes import KinematicRobot, ScriptedPerceiver
from .test_executor import WORLD, step

HFOV = math.radians(82)


def visual_executor(robot, perceiver, **config):
    cfg = ExecutorConfig(approach="visual", max_queries=60, camera_hfov_rad=HFOV, **config)
    return StepExecutor(robot, SimpleNavigator(robot), perceiver, SemanticMap.from_world(WORLD),
                        camera=robot.camera, safety_check=robot.safety, config=cfg)


def ground_truth(robot, world=WORLD):
    def locate(name):
        spec = world["entities"].get(name, {})
        return tuple(spec["position"][:2]) if "position" in spec else None
    return GroundTruthPerceiver(locate, robot.odom, hfov_rad=HFOV, near_m=0.55)


def test_direction_bearing_left_is_positive_and_symmetric():
    assert direction_bearing("center", HFOV) == 0
    assert direction_bearing("left", HFOV) > 0
    assert direction_bearing("left", HFOV) == pytest.approx(-direction_bearing("right", HFOV))
    assert direction_bearing("left", HFOV) < HFOV / 2


def test_ground_goal_adapter_steps_along_bearing():
    goal = GroundGoalAdapter().goal_toward(Pose2D(1.0, 2.0, 0.0), math.pi / 2, 0.5)
    assert (goal.x, goal.y, goal.yaw) == pytest.approx((1.0, 2.5, math.pi / 2))


def test_visual_approach_reaches_target_in_view():
    robot = KinematicRobot()
    result = visual_executor(robot, ground_truth(robot)).execute(step("movetowards", {"target": "table"}), "puppypi_1")
    assert result.status == c.SUCCEEDED, result.reason
    assert math.hypot(1.5 - robot.pose.x, 0.0 - robot.pose.y) <= 0.6


def test_visual_approach_turns_to_find_target_behind():
    robot = KinematicRobot(yaw=math.pi)  # table is behind the robot
    executor = visual_executor(robot, ground_truth(robot))
    result = executor.execute(step("movetowards", {"target": "table"}), "puppypi_1")
    assert result.status == c.SUCCEEDED, result.reason
    assert any(q["question"] == "where" and q["direction"] == "none" for q in executor.trace.queries)


def test_visual_approach_not_found_after_full_turn():
    world = {"entities": {**WORLD["entities"], "ghost": {"kind": "object", "description": "a ghost", "in": "living_room"}}}
    robot = KinematicRobot()
    executor = StepExecutor(robot, SimpleNavigator(robot), ground_truth(robot, world), SemanticMap.from_world(world),
                            camera=robot.camera, safety_check=robot.safety,
                            config=ExecutorConfig(approach="visual", max_queries=60))
    result = executor.execute(step("movetowards", {"target": "ghost"}), "puppypi_1")
    assert (result.status, result.reason_code) == (c.BLOCKED, "OBJECT_NOT_FOUND")


class SlowPerceiver(ScriptedPerceiver):
    def is_visible(self, image, phrase, entity):
        detection = super().is_visible(image, phrase, entity)
        return Detection(detection.visible, detection.confidence, "vlm", latency_s=2.0)


def test_onboard_latency_is_charged_to_robot_time():
    robot = KinematicRobot()
    perceiver = SlowPerceiver([False, False, True])
    executor = StepExecutor(robot, SimpleNavigator(robot), perceiver, SemanticMap.from_world(WORLD),
                            camera=robot.camera, safety_check=robot.safety)
    t0 = robot.now()
    result = executor.execute(step("search", {"object": "apple", "area": "living_room"}), "puppypi_1")
    assert result.status == c.SUCCEEDED
    assert robot.now() - t0 >= 3 * 2.0  # three answers at 2 s each, plus turning and settling


def test_slow_onboard_computer_can_time_out():
    robot = KinematicRobot()
    executor = StepExecutor(robot, SimpleNavigator(robot), SlowPerceiver([]), SemanticMap.from_world(WORLD),
                            camera=robot.camera, safety_check=robot.safety)
    result = executor.execute(step("search", {"object": "apple", "area": "living_room"}, timeout_s=5.0), "puppypi_1")
    assert (result.status, result.reason_code) == (c.TIMED_OUT, "TIMEOUT")


class CenterOnlyPerceiver(ScriptedPerceiver):
    """Always says the target is straight ahead and never says it is near."""

    def where(self, image, phrase, entity):
        from puppy_agent.perception import Direction
        return Direction("center", 0.9, "vlm")

    def is_near(self, image, phrase, entity):
        return Detection(False, 0.9, "vlm")


def test_map_near_source_stops_by_distance_when_vlm_never_says_near():
    robot = KinematicRobot()
    executor = visual_executor(robot, CenterOnlyPerceiver([]), near_source="map", near_m=0.6)
    result = executor.execute(step("movetowards", {"target": "table"}), "puppypi_1")
    assert result.status == c.SUCCEEDED, result.reason
    assert math.hypot(1.5 - robot.pose.x, robot.pose.y) <= 0.6


def test_vlm_near_source_gives_up_when_never_near():
    robot = KinematicRobot()
    executor = visual_executor(robot, CenterOnlyPerceiver([]), approach_max_moves=3)
    result = executor.execute(step("movetowards", {"target": "table"}), "puppypi_1")
    assert (result.status, result.reason_code) == (c.FAILED, "DECISION_LIMIT")
