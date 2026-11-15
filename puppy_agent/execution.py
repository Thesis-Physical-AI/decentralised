"""Synchronous fleet dispatch and result correlation; no ROS or model calls.

Independent robots are serialized in this first runner. Dependencies remain global.
Any failure stops new dispatches; unregistered executors leave work pending.
"""
import copy

from . import contracts as c
from .dispatch import adapt_dispatch

TERMINAL = {c.SUCCEEDED, c.FAILED, c.BLOCKED, c.TIMED_OUT, c.CANCELED, c.REJECTED}
IDENTITY = ("mission_id", "plan_revision", "step_id", "robot_id", "attempt")


class ExecutionManager:
    def __init__(self, plan, *, plan_revision, timeout_s):
        self.mission = adapt_dispatch(plan, plan_revision=plan_revision, timeout_s=timeout_s)
        self.assignments = [c.StepAssignment.from_dict(s) for s in self.mission["steps"]]
        self.results = {}
        self.in_flight = None
        self.halted = False

    def ready(self):
        if self.halted or self.in_flight is not None:
            return []
        return [copy.deepcopy(s) for s in self.assignments if s.step_id not in self.results
                and all(self.results.get(d, {}).get("status") == c.SUCCEEDED for d in s.depends_on)]

    def start(self, step_id):
        step = next((s for s in self.ready() if s.step_id == step_id), None)
        if step is None:
            raise c.ContractError(f"step {step_id} is not ready")
        self.in_flight = copy.deepcopy(step)
        return step

    def accept(self, result):
        raw = result.to_dict() if isinstance(result, c.StepResult) else copy.deepcopy(result)
        step = self.in_flight
        if step is None or any(raw.get(k) != getattr(step, k) for k in IDENTITY):
            raise c.ContractError("result does not match the in-flight assignment")
        if raw.get("status") not in TERMINAL:
            raise c.ContractError("result must have a terminal executor status")
        # Apply existing status/reason invariants before recording feedback.
        checked = c.StepResult(**raw)
        self.results[step.step_id] = checked.to_dict()
        self.in_flight = None
        self.halted = checked.status != c.SUCCEEDED

    def run(self, executors):
        """executors maps robot IDs to synchronous callables: assignment -> StepResult.

        Exceptions and invalid feedback halt this run visibly; they are never
        retried or converted into successful results. The assignment remains in flight.
        """
        while True:
            step = next((s for s in self.ready() if s.robot_id in executors), None)
            if step is None:
                return self.report(executors)
            self.accept(executors[step.robot_id](self.start(step.step_id)))

    def report(self, executors=()):
        steps = []
        for step in self.assignments:
            if step.step_id in self.results:
                steps.append(copy.deepcopy(self.results[step.step_id]))
                continue
            waiting = [d for d in step.depends_on if self.results.get(d, {}).get("status") != c.SUCCEEDED]
            reason = ("mission halted after unsuccessful step" if self.halted else
                      f"dependencies not succeeded: {waiting}" if waiting else
                      "executor unavailable" if step.robot_id not in executors else "pending")
            steps.append({"step_id": step.step_id, "robot_id": step.robot_id,
                          "status": "NOT_DISPATCHED", "reason": reason})
        return {"mission_id": self.mission["mission_id"],
                "plan_revision": self.mission["plan_revision"],
                "status": "HALTED" if self.halted else
                          "SUCCEEDED" if len(self.results) == len(self.assignments) else "PENDING",
                "steps": steps}
