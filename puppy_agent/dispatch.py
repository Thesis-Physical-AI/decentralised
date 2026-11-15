"""Adapt an already validated central dispatch to the prototype mission format.

This is a transport adapter, not the central physical/semantic plan validator.
All robot assignments and cross-robot dependencies are preserved.
"""
import argparse
import copy
import json
import math
from dataclasses import asdict

from .contracts import ContractError, StepAssignment


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{name} must be a nonempty string")
    return value


def adapt_dispatch(plan, *, plan_revision, timeout_s):
    """Return a mission accepted by run_prototype.py; never mutate the input.

    Revision and timeout are caller policy because central dispatch has neither.
    Central world data is intentionally not converted into simulation geometry.
    """
    if type(plan_revision) is not int or plan_revision < 0:
        raise ContractError("plan_revision must be a nonnegative integer")
    if (isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float))
            or not math.isfinite(timeout_s) or timeout_s <= 0):
        raise ContractError("timeout_s must be finite and positive")
    if not isinstance(plan, dict):
        raise ContractError("dispatch must be an object")
    mission_id = _text(plan.get("mission_id"), "mission_id")
    if plan.get("unassigned") != []:
        raise ContractError("dispatch must explicitly have no unassigned tasks")
    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ContractError("steps must be a nonempty list")

    assignments, seen, previous = [], set(), {}
    for raw in steps:
        if not isinstance(raw, dict):
            raise ContractError("each step must be an object")
        sid = _text(raw.get("id"), "step id")
        if sid in seen:
            raise ContractError(f"duplicate step id {sid}")
        robot_id = _text(raw.get("robot"), "robot")
        for key in ("task_id", "action", "instruction"):
            _text(raw.get(key), key)
        dependencies = raw.get("depends_on", [])
        if not isinstance(dependencies, list):
            raise ContractError(f"{sid}: depends_on must be a list")
        for dependency in dependencies:
            _text(dependency, "dependency")
        # Central list order is an implicit dependency, including after failures.
        dependencies = list(dict.fromkeys(dependencies + ([previous[robot_id]] if robot_id in previous else [])))
        assignment = StepAssignment.from_dict(dict(
            mission_id=mission_id, plan_revision=plan_revision, step_id=sid,
            task_id=raw["task_id"], robot_id=robot_id, action=raw["action"],
            args=copy.deepcopy(raw.get("args")), instruction=raw["instruction"],
            depends_on=dependencies, timeout_s=timeout_s,
        ))
        assignments.append(asdict(assignment))
        seen.add(sid)
        previous[robot_id] = sid
    # Topological order may differ from list order across different robots.
    ordered, done = [], set()
    while len(ordered) < len(assignments):
        ready = [a for a in assignments if a["step_id"] not in done
                 and set(a["depends_on"]) <= done]
        if not ready:
            raise ContractError("unknown dependency or cycle (including per-robot ordering)")
        ordered.extend(ready)
        done.update(a["step_id"] for a in ready)
    assignments = ordered
    return {"mission_id": mission_id, "plan_revision": plan_revision, "steps": assignments}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dispatch")
    parser.add_argument("--plan-revision", required=True, type=int)
    parser.add_argument("--timeout-s", required=True, type=float)
    parser.add_argument("--output", required=True, help="new prototype mission JSON file")
    options = parser.parse_args()
    try:
        with open(options.dispatch, encoding="utf-8") as stream:
            mission = adapt_dispatch(json.load(stream), plan_revision=options.plan_revision, timeout_s=options.timeout_s)
        with open(options.output, "x", encoding="utf-8") as stream:
            json.dump(mission, stream, indent=2)
    except (OSError, ValueError) as error:
        parser.exit(2, f"dispatch adaptation failed: {error}\n")


if __name__ == "__main__":
    main()
