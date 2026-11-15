"""Exercise the central wire format through the real executor without GPU/Isaac."""
import copy
import json
from pathlib import Path

import pytest

from puppy_agent import contracts as c
from puppy_agent.dispatch import adapt_dispatch

from .test_executor import make


def dispatch():
    return {
        "mission_id": "M-INTEGRATION", "world": {},
        "allocations": [{"task_id": "T1", "robots": ["puppypi_1"]}], "unassigned": [],
        "steps": [
            {"id": "S1", "task_id": "T1", "robot": "puppypi_1", "action": "search",
             "args": {"object": "apple", "area": "living_room"},
             "instruction": "[search] apple in living_room", "depends_on": []},
            {"id": "S2", "task_id": "T1", "robot": "puppypi_1", "action": "grab",
             "args": {"object": "apple"}, "instruction": "[grab] apple", "depends_on": []},
        ],
    }


def adapt(plan):
    return adapt_dispatch(plan, plan_revision=3, timeout_s=120)


@pytest.mark.parametrize("visible,status,code", [
    (True, c.SUCCEEDED, ""), (False, c.BLOCKED, "OBJECT_NOT_FOUND"),
])
def test_dispatch_executor_result_round_trip(visible, status, code):
    plan = dispatch()
    original = copy.deepcopy(plan)
    mission = json.loads(json.dumps(adapt(plan)))
    assert plan == original
    executor, _, _ = make([visible])
    assignment = c.StepAssignment.from_dict(mission["steps"][0])
    result = json.loads(json.dumps(executor.execute(assignment, "puppypi_1").to_dict()))
    assert (result["status"], result["reason_code"]) == (status, code)
    for key in ("mission_id", "plan_revision", "step_id", "robot_id", "attempt"):
        assert result[key] == mission["steps"][0][key]
    assert result["observations"][0]["subject"] == "apple"
    assert mission["steps"][1]["depends_on"] == ["S1"]


def test_manipulation_reaches_existing_rejection_path():
    executor, robot, _ = make([])
    assignment = c.StepAssignment.from_dict(adapt(dispatch())["steps"][1])
    result = executor.execute(assignment, "puppypi_1")
    assert (result.status, result.reason_code) == (c.REJECTED, "UNKNOWN_ACTION")
    assert robot.now() == 0


@pytest.mark.parametrize("change", [
    lambda p: p["steps"][1].update(id="S1"),
    lambda p: p["steps"][0].update(depends_on=["S2"]),
    lambda p: p["steps"][0].update(depends_on=["missing"]),
    lambda p: p["steps"][0].update(args={"object": 1}),
    lambda p: p.update(unassigned=[{"task_id": "T2"}]),
])
def test_rejects_unexecutable_envelopes(change):
    plan = dispatch()
    change(plan)
    with pytest.raises(c.ContractError):
        adapt(plan)


@pytest.mark.parametrize("revision,timeout", [(True, 60), (-1, 60), (0, 0), (0, float("nan"))])
def test_requires_explicit_valid_execution_policy(revision, timeout):
    with pytest.raises(c.ContractError):
        adapt_dispatch(dispatch(), plan_revision=revision, timeout_s=timeout)


def test_existing_demo_still_defaults_revision_and_timeout():
    path = Path(__file__).resolve().parents[2] / "missions/demo_one_room.json"
    mission = json.loads(path.read_text())
    raw = mission["steps"][0]
    assignment = c.StepAssignment.from_dict({"mission_id": mission["mission_id"],
                                           "plan_revision": 0, "robot_id": "puppypi_1", **raw})
    assert assignment.plan_revision == 0
    assert assignment.timeout_s == 90
