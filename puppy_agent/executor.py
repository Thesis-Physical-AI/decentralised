"""Step executor (architecture layer 6): runs one dispatched step and returns exactly one result.

Supported PuppyPi actions in this prototype:
  movetowards{target}     navigate to the target's approach pose, then confirm it is in view
  search{object, area}    go to the area (if it has an approach pose), turn in place looking for the object
  inspect{target}         navigate to the target and confirm it is in view
Actions the simulated PuppyPi cannot do yet (open, close, grab, putinto, puton, push) are REJECTED.

Safety and limits are deterministic and independent of any model: a sim-time timeout per step,
a cap on perception queries, and a fall check on every control tick.
"""
import math
from dataclasses import dataclass, field

from . import contracts as c
from .navigation import ABORTED, CANCELED, SUCCEEDED

SUPPORTED = {"movetowards": ("target",), "search": ("object", "area"), "inspect": ("target",)}
NOT_IN_SIM = {"open", "close", "grab", "putinto", "puton", "push"}


@dataclass
class ExecutorConfig:
    control_dt: float = 0.05  # s between navigator updates
    search_turns: int = 8  # look-around increments for a full turn
    max_queries: int = 12  # perception queries per step (DECISION_LIMIT)
    settle_s: float = 0.4  # wait after stopping before capturing an image


@dataclass
class DecisionTrace:
    """Every perception query made during a step, for logs and evaluation."""
    queries: list = field(default_factory=list)


class _StepEnded(Exception):
    def __init__(self, status, reason_code, reason):
        super().__init__(reason)
        self.status, self.reason_code, self.reason = status, reason_code, reason


class StepExecutor:
    def __init__(self, robot, navigator, perceiver, semantic_map, camera, safety_check, config=None,
                 on_query=None):
        """camera() -> RGB image (H, W, 3 uint8). safety_check() -> "" if safe, else a reason string.
        on_query(step_id, entity, image, detection) is called after each perception query (e.g. to save images)."""
        self.robot, self.nav, self.perceiver, self.map = robot, navigator, perceiver, semantic_map
        self.camera, self.safety_check = camera, safety_check
        self.config = config or ExecutorConfig()
        self.on_query = on_query

    # -- public ------------------------------------------------------------------------------
    def execute(self, assignment, robot_id):
        started = c.now_iso()
        self.trace = DecisionTrace()
        self.observations = []
        base = dict(mission_id=assignment.mission_id, plan_revision=assignment.plan_revision,
                    step_id=assignment.step_id, robot_id=robot_id, attempt=assignment.attempt)
        try:
            self._check_assignment(assignment, robot_id)
            self.deadline = self.robot.now() + assignment.timeout_s
            self.step = assignment
            reason = getattr(self, "_" + assignment.action)(**assignment.args)
            status, code = c.SUCCEEDED, ""
        except _StepEnded as end:
            status, code, reason = end.status, end.reason_code, end.reason
        finally:
            self.robot.set_cmd_vel(0.0, 0.0)
        return c.StepResult(**base, status=status, reason_code=code, reason=reason,
                            observations=[c.asdict(o) for o in self.observations],
                            started_at=started, ended_at=c.now_iso())

    # -- validation --------------------------------------------------------------------------
    def _check_assignment(self, a, robot_id):
        if a.robot_id != robot_id:
            raise _StepEnded(c.REJECTED, "INVALID_ARGS", f"step is for {a.robot_id}, this is {robot_id}")
        if a.action in NOT_IN_SIM:
            raise _StepEnded(c.REJECTED, "UNKNOWN_ACTION",
                             f"'{a.action}' needs a manipulator; not available on the simulated PuppyPi")
        if a.action not in SUPPORTED:
            raise _StepEnded(c.REJECTED, "UNKNOWN_ACTION", f"unknown action '{a.action}'")
        if set(a.args) != set(SUPPORTED[a.action]):
            raise _StepEnded(c.REJECTED, "INVALID_ARGS",
                             f"{a.action} expects args {SUPPORTED[a.action]}, got {sorted(a.args)}")
        for name in a.args.values():
            if self.map.get(name) is None:
                raise _StepEnded(c.REJECTED, "INVALID_ARGS", f"unknown entity '{name}'")

    # -- actions -----------------------------------------------------------------------------
    def _movetowards(self, target):
        entity = self.map.get(target)
        if entity.approach is None:
            raise _StepEnded(c.REJECTED, "INVALID_ARGS", f"'{target}' has no approach pose in the semantic map")
        self._navigate_to(entity.approach)
        if not self._look_for(entity).visible:
            raise _StepEnded(c.BLOCKED, "OBJECT_NOT_FOUND", f"arrived at {target} but it is not in view")
        return f"at {target}"

    def _inspect(self, target):
        return self._movetowards(target).replace("at", "inspected", 1)

    def _search(self, object, area):  # noqa: A002 - matches the action's argument name
        target, region = self.map.get(object), self.map.get(area)
        if region.approach is not None:
            self._navigate_to(region.approach)
        for turn in range(self.config.search_turns):
            detection = self._look_for(target)
            if detection.visible:
                self._observe("entity_located", object, {"in": area}, detection)
                return f"found {object} in {area} after {turn} turns"
            self._spin(2 * math.pi / self.config.search_turns)
        self._observe("entity_absent", object, {"in": area}, None)
        raise _StepEnded(c.BLOCKED, "OBJECT_NOT_FOUND", f"{object} not seen in {area} after a full turn")

    # -- building blocks ---------------------------------------------------------------------
    def _run_nav(self, what):
        while self.nav.result() is None:
            self._tick()
            self.nav.spin_once()
            self.robot.advance(self.config.control_dt)
        result = self.nav.result()
        if result == ABORTED:
            raise _StepEnded(c.BLOCKED, "PATH_BLOCKED", f"{what}: navigation made no progress")
        if result == CANCELED:
            raise _StepEnded(c.CANCELED, "ESTOPPED", f"{what}: navigation canceled")
        assert result == SUCCEEDED

    def _navigate_to(self, pose):
        self.nav.go_to(pose)
        self._run_nav(f"go_to({pose.x:.2f}, {pose.y:.2f})")

    def _spin(self, angle):
        self.nav.spin(angle)
        self._run_nav(f"spin({math.degrees(angle):.0f}deg)")

    def _tick(self):
        problem = self.safety_check()
        if problem:
            self.nav.cancel()
            raise _StepEnded(c.FAILED, "SAFETY_LIMIT", problem)
        if self.robot.now() > self.deadline:
            self.nav.cancel()
            raise _StepEnded(c.TIMED_OUT, "TIMEOUT", f"exceeded {self.step.timeout_s:.0f} s")

    def _look_for(self, entity):
        if len(self.trace.queries) >= self.config.max_queries:
            raise _StepEnded(c.FAILED, "DECISION_LIMIT", f"more than {self.config.max_queries} perception queries")
        self.robot.set_cmd_vel(0.0, 0.0)
        self.robot.advance(self.config.settle_s)
        self._tick()
        image = self.camera()
        detection = self.perceiver.is_visible(image, entity.description, entity.name)
        self.trace.queries.append({"entity": entity.name, "phrase": entity.description,
                                   "visible": detection.visible, "confidence": round(detection.confidence, 3),
                                   "source": detection.source, "latency_s": round(detection.latency_s, 3),
                                   "detail": detection.detail})
        if self.on_query:
            self.on_query(self.step.step_id, entity.name, image, detection)
        return detection

    def _observe(self, kind, subject, value, detection):
        self.observations.append(c.Observation(
            robot_id=self.step.robot_id, mission_id=self.step.mission_id, step_id=self.step.step_id,
            kind=kind, subject=subject, value=value,
            confidence=round(detection.confidence, 3) if detection else 1.0,
            source=detection.source if detection else "executor"))
