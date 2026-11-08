import pytest

from puppy_agent import contracts as c
from puppy_agent.executor import StepExecutor
from puppy_agent.navigation import SimpleNavigator
from puppy_agent.semantic_map import SemanticMap

from .fakes import KinematicRobot, ScriptedPerceiver, StuckRobot

WORLD = {"entities": {
    "living_room": {"kind": "room", "description": "a room"},
    "table": {"kind": "surface", "description": "a brown table", "position": [1.5, 0.0, 0.15],
              "approach": [0.9, 0.0, 0], "in": "living_room"},
    "apple": {"kind": "object", "description": "a red apple", "position": [1.5, 0.0, 0.35], "in": "table"},
}}


def step(action, args, **extra):
    data = dict(mission_id="M-1", plan_revision=0, step_id="S1", task_id="T1", robot_id="puppypi_1",
                action=action, args=args, instruction=f"[{action}]", timeout_s=120.0)
    data.update(extra)
    return c.StepAssignment.from_dict(data)


def make(answers, robot=None):
    robot = robot or KinematicRobot()
    perceiver = ScriptedPerceiver(answers)
    executor = StepExecutor(robot, SimpleNavigator(robot), perceiver, SemanticMap.from_world(WORLD),
                            camera=robot.camera, safety_check=robot.safety)
    return executor, robot, perceiver


def test_movetowards_succeeds_when_target_in_view():
    executor, robot, _ = make([True])
    result = executor.execute(step("movetowards", {"target": "table"}), "puppypi_1")
    assert result.status == c.SUCCEEDED and result.reason_code == ""
    assert robot.odom().x == pytest.approx(0.9, abs=0.1)


def test_movetowards_blocked_when_target_not_seen():
    executor, _, _ = make([False])
    result = executor.execute(step("movetowards", {"target": "table"}), "puppypi_1")
    assert (result.status, result.reason_code) == (c.BLOCKED, "OBJECT_NOT_FOUND")


def test_search_reports_location_observation():
    executor, _, perceiver = make([False, False, True])
    result = executor.execute(step("search", {"object": "apple", "area": "living_room"}), "puppypi_1")
    assert result.status == c.SUCCEEDED
    assert len(perceiver.calls) == 3
    [obs] = result.observations
    assert (obs["kind"], obs["subject"], obs["value"]) == ("entity_located", "apple", {"in": "living_room"})


def test_search_not_found_after_full_turn():
    executor, _, perceiver = make([])
    result = executor.execute(step("search", {"object": "apple", "area": "living_room"}), "puppypi_1")
    assert (result.status, result.reason_code) == (c.BLOCKED, "OBJECT_NOT_FOUND")
    assert len(perceiver.calls) == executor.config.search_turns
    assert result.observations[0]["kind"] == "entity_absent"


@pytest.mark.parametrize("action,args,code", [
    ("grab", {"object": "apple"}, "UNKNOWN_ACTION"),
    ("fly", {}, "UNKNOWN_ACTION"),
    ("movetowards", {"target": "kitchen"}, "INVALID_ARGS"),
    ("movetowards", {"object": "table"}, "INVALID_ARGS"),
])
def test_rejects_without_moving(action, args, code):
    executor, robot, perceiver = make([True])
    result = executor.execute(step(action, args), "puppypi_1")
    assert (result.status, result.reason_code) == (c.REJECTED, code)
    assert robot.now() == 0.0 and perceiver.calls == []


def test_rejects_step_for_another_robot():
    executor, _, _ = make([True])
    result = executor.execute(step("movetowards", {"target": "table"}, robot_id="drone_1"), "puppypi_1")
    assert (result.status, result.reason_code) == (c.REJECTED, "INVALID_ARGS")


def test_timeout():
    executor, _, _ = make([True])
    result = executor.execute(step("movetowards", {"target": "table"}, timeout_s=1.0), "puppypi_1")
    assert (result.status, result.reason_code) == (c.TIMED_OUT, "TIMEOUT")


def test_path_blocked():
    executor, _, _ = make([True], robot=StuckRobot())
    executor.nav.stuck_window_s = 2.0
    result = executor.execute(step("movetowards", {"target": "table"}), "puppypi_1")
    assert (result.status, result.reason_code) == (c.BLOCKED, "PATH_BLOCKED")


def test_safety_stop():
    executor, robot, _ = make([True])
    robot.fallen = True
    result = executor.execute(step("movetowards", {"target": "table"}), "puppypi_1")
    assert (result.status, result.reason_code) == (c.FAILED, "SAFETY_LIMIT")
    assert (robot.vx, robot.wz) == (0.0, 0.0)


def test_result_contract_invariants():
    with pytest.raises(c.ContractError):
        c.StepResult("M", 0, "S", "r", 1, c.SUCCEEDED, reason_code="TIMEOUT")
    with pytest.raises(c.ContractError):
        c.StepResult("M", 0, "S", "r", 1, c.FAILED, reason_code="")
    with pytest.raises(c.ContractError):
        c.StepAssignment.from_dict({"mission_id": "M"})
