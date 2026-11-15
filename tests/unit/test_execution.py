import copy
import json
from pathlib import Path

import pytest

from puppy_agent import contracts as c
from puppy_agent.dispatch import adapt_dispatch
from puppy_agent.execution import ExecutionManager

from .test_dispatch import dispatch
from .test_executor import make


def fleet_plan():
    plan = dispatch()
    # Later list entries can be prerequisites when they belong to another robot.
    plan["steps"][0]["depends_on"] = ["SCOUT"]
    scout = copy.deepcopy(plan["steps"][0])
    scout.update(id="SCOUT", robot="drone_1", depends_on=[])
    plan["steps"].append(scout)
    return plan


def manager(plan=None):
    return ExecutionManager(plan or fleet_plan(), plan_revision=2, timeout_s=120)


def success(step):
    return c.StepResult(**{k: getattr(step, k) for k in
                         ("mission_id", "plan_revision", "step_id", "robot_id", "attempt")},
                        status=c.SUCCEEDED)


def test_full_fleet_routes_and_correlates_real_puppy_feedback():
    executor, _, _ = make([True])
    calls = []

    def scout(step):
        calls.append(step.step_id)
        return success(step)

    def puppy(step):
        calls.append(step.step_id)
        return executor.execute(step, "puppypi_1")

    report = manager().run({"drone_1": scout, "puppypi_1": puppy})
    assert calls == ["SCOUT", "S1", "S2"]
    assert report["status"] == "HALTED"  # unsupported grab
    assert [r["status"] for r in report["steps"]] == [c.SUCCEEDED, c.SUCCEEDED, c.REJECTED]
    assert report["steps"][1]["observations"][0]["subject"] == "apple"


def test_missing_peer_does_not_release_cross_robot_wait():
    executor, robot, _ = make([True])
    report = manager().run({"puppypi_1": lambda s: executor.execute(s, "puppypi_1")})
    assert report["status"] == "PENDING"
    assert all(s["status"] == "NOT_DISPATCHED" for s in report["steps"])
    assert robot.now() == 0


def test_failure_stops_dependents_and_independent_work():
    plan = fleet_plan()
    plan["steps"][0]["depends_on"] = []
    executor, _, _ = make([])
    report = manager(plan).run({"puppypi_1": lambda s: executor.execute(s, "puppypi_1"),
                                "drone_1": lambda s: pytest.fail("must halt before scout")})
    assert report["status"] == "HALTED"
    assert report["steps"][0]["status"] == c.BLOCKED
    assert all(r["status"] == "NOT_DISPATCHED" for r in report["steps"][1:])


@pytest.mark.parametrize("key,value", [("mission_id", "other"), ("plan_revision", 0),
                                      ("step_id", "S1"), ("robot_id", "puppypi_1"),
                                      ("attempt", 2), ("status", "NOT_DISPATCHED")])
def test_rejects_wrong_or_stale_feedback(key, value):
    run = manager()
    raw = success(run.start("SCOUT")).to_dict()
    raw[key] = value
    with pytest.raises(c.ContractError):
        run.accept(raw)
    assert run.results == {}
    assert run.ready() == []


def test_accepts_once_and_finishes_successfully():
    run = manager(dispatch())
    first = success(run.start("S1"))
    run.accept(first)
    with pytest.raises(c.ContractError):
        run.accept(first)
    assert run.run({"puppypi_1": success})["status"] == "SUCCEEDED"


def test_central_worked_example_preserves_every_robot_and_dependency():
    path = Path(__file__).resolve().parents[3] / "centralized/mission-orchestrator/examples/dispatch_plan.json"
    if not path.exists():
        pytest.skip("sibling centralized checkout not present")
    plan = json.loads(path.read_text())
    adapted = adapt_dispatch(plan, plan_revision=0, timeout_s=120)
    by_id = {s["step_id"]: s for s in adapted["steps"]}
    assert len(by_id) == len(plan["steps"])
    previous = {}
    for raw in plan["steps"]:
        step = by_id[raw["id"]]
        assert step["robot_id"] == raw["robot"]
        assert set(raw["depends_on"]) <= set(step["depends_on"])
        if raw["robot"] in previous:
            assert previous[raw["robot"]] in step["depends_on"]
        previous[raw["robot"]] = raw["id"]
