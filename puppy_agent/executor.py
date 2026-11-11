"""Step executor (architecture layer 6): runs one dispatched step and returns exactly one result.

Supported PuppyPi actions in this prototype:
  movetowards{target}     approach="map": navigate to the target's approach pose, then confirm it is in view
                          approach="visual": the VLM picks each goal from the camera image (left / center /
                          right / not in view, then near?) and the navigator drives to it
                          approach="live": the VLM watches a live camera feed (live_feed.py, >= 1 frame/s),
                          answers a set of questions about each frame it analyses (open ones in words, closed
                          ones scored), and picks the next primitive from a closed menu (selector.py) while the
                          robot keeps moving; the executor validates each choice and turns it into a bounded command
  search{object, area}    go to the area (if it has an approach pose), turn in place looking for the object;
                          approach="live": turn while watching the feed, stop and confirm when the object is seen
  inspect{target}         navigate to the target and confirm it is in view
Actions the simulated PuppyPi cannot do yet (open, close, grab, putinto, puton, push) are REJECTED.

Safety and limits are deterministic and independent of any model: a sim-time timeout per step,
a cap on perception queries, and a fall check on every control tick. Each model answer costs its
onboard latency in robot time, so a slow onboard computer shows up as a slower robot and can trigger
the step timeout. In the map and visual approaches the robot stands still while it waits. In the live
approach it keeps executing its last validated command (at most one short goal) while the feed's analyses
arrive, answers older than max_decision_age_s are discarded, and "arrived" / "found" claims are confirmed
on a frame taken after the robot stopped.
"""
import math
from dataclasses import dataclass, field

from . import contracts as c
from .navigation import ABORTED, CANCELED, SUCCEEDED
from .live_feed import LiveFeed, Rotation, frame_questions
from .perception import Detection, direction_bearing
from .selector import MotionLimits, motion_for, validate
from .visual_goal import GroundGoalAdapter

SUPPORTED = {"movetowards": ("target",), "search": ("object", "area"), "inspect": ("target",)}
NOT_IN_SIM = {"open", "close", "grab", "putinto", "puton", "push"}


@dataclass
class ExecutorConfig:
    control_dt: float = 0.05  # s between navigator updates
    search_turns: int = 8  # look-around increments for a full turn
    max_queries: int = 12  # perception queries per step (DECISION_LIMIT)
    settle_s: float = 0.4  # wait after stopping before capturing an image
    approach: str = "map"  # movetowards: "map" (approach pose from the semantic map) or "visual" (VLM picks goals)
    approach_step_m: float = 0.4  # distance of each VLM-chosen goal
    approach_max_moves: int = 10  # VLM-chosen goals per step before giving up (DECISION_LIMIT)
    near_source: str = "vlm"  # visual approach stop: "vlm" asks "is it close?"; "map" uses distance to the map position
    near_m: float = 0.6  # near_source="map": stop within this distance of the entity's map position
    camera_hfov_rad: float = math.radians(82)  # to turn left/center/right into a bearing
    charge_perception_time: bool = True  # robot waits (sim time) for each answer's latency
    # approach="live"
    feed_hz: float = 2.0  # camera frames per second of robot time streamed to the VLM (>= 1)
    feed_mode: str = "async"  # "async": the sim keeps running while the VLM thinks; "lockstep": it waits
    open_tokens: int = 32  # length cap of each open (worded) answer
    max_decision_age_s: float = 6.0  # discard a decision whose image is older than this when it arrives
    confirm_arrival: bool = True  # "arrived" while moving: stop, and accept only if a still image agrees
    motion: MotionLimits = field(default_factory=MotionLimits)


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
                 on_query=None, goal_adapter=None, on_event=None):
        """camera() -> RGB image (H, W, 3 uint8). safety_check() -> "" if safe, else a reason string.
        on_query(step_id, entity, image, answer) is called after each perception query (e.g. to save images).
        on_event(dict) is called for each navigation goal and answer (e.g. for a live log).
        goal_adapter turns a bearing into a goal for this robot type (default: ground robot)."""
        self.robot, self.nav, self.perceiver, self.map = robot, navigator, perceiver, semantic_map
        self.camera, self.safety_check = camera, safety_check
        self.config = config or ExecutorConfig()
        self.on_query = on_query
        self.on_event = on_event or (lambda event: None)
        self.goals = goal_adapter or GroundGoalAdapter()
        self.feed = None
        self.feed_stats = None

    # -- public ------------------------------------------------------------------------------
    def execute(self, assignment, robot_id):
        started = c.now_iso()
        self.trace = DecisionTrace()
        self.feed_stats = None
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
            self._close_feed()
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
        if self.config.approach == "visual":
            return self._approach_visually(entity)
        if self.config.approach == "live":
            return self._approach_live(entity)
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
        if self.config.approach == "live":
            return self._search_live(target, region)
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

    def _approach_visually(self, entity):
        """The VLM picks each goal; the navigator drives there. Ends when the target is centred and near."""
        moves = scans = 0
        while True:
            direction = self._ask("where", entity)
            if direction.label == "none":
                if scans >= self.config.search_turns:
                    raise _StepEnded(c.BLOCKED, "OBJECT_NOT_FOUND", f"{entity.name} not in view after a full turn")
                scans += 1
                self._spin(2 * math.pi / self.config.search_turns)
                continue
            scans = 0
            if direction.label == "center" and self._is_near(entity):
                return f"at {entity.name} (visual approach, {moves} goals)"
            if moves >= self.config.approach_max_moves:
                raise _StepEnded(c.FAILED, "DECISION_LIMIT",
                                 f"not near {entity.name} after {moves} VLM-chosen goals")
            bearing = direction_bearing(direction.label, self.config.camera_hfov_rad)
            goal = self.goals.goal_toward(self.robot.odom(), bearing, self.config.approach_step_m)
            self.on_event({"step_id": self.step.step_id, "event": "goal", "from": direction.label,
                           "goal": [round(goal.x, 3), round(goal.y, 3), round(goal.yaw, 3)]})
            moves += 1
            self._navigate_to(goal)

    def _approach_live(self, entity):
        """Live selector loop on the camera feed: each analysed frame's action is validated and acted on
        while the robot keeps executing its last command."""
        history, rotation = [], Rotation()
        self._open_feed(entity, history)
        while True:
            analysis = self._next_analysis()
            # rotation while searching, including while the model was thinking
            searched = rotation.update(self.robot.odom().yaw) if history[-1:] == ["search"] else rotation.reset()
            action = validate(analysis.action)
            if action is None:
                self._event("rejected", analysis, f"'{analysis.action}' is not on the action menu")
                continue
            if analysis.age_s > self.config.max_decision_age_s:
                self._event("rejected", analysis, f"stale: image was {analysis.age_s:.1f} s old")
                continue
            if action == "arrived":
                arrived, why = self._accept_arrival(entity)
                if arrived:
                    return f"at {entity.name} (live selector, {len(history)} decisions)"
                self._event("rejected", analysis, why)
                action = "forward"  # not there yet: keep approaching
                rotation.reset()
            if action == "search" and searched >= 2 * math.pi:
                raise _StepEnded(c.BLOCKED, "OBJECT_NOT_FOUND", f"{entity.name} not in view after a full turn")
            history.append(action)
            self._start(motion_for(action, self.robot.odom(), history, self.config.motion), analysis)

    def _search_live(self, target, region):
        """Turn in search_turns increments while the feed is analysed; a frame where the object is visible
        stops the robot, and a frame taken after stopping must agree. The robot turns on only after a frame
        taken at the current heading has been analysed, so a slow model cannot turn past an unseen view."""
        self._open_feed(target, [])
        if region.approach is not None:
            self._navigate_to(region.approach)
        step = 2 * math.pi / self.config.search_turns
        for turn in range(self.config.search_turns):
            if self._next_analysis(newer_than=self.robot.now()).visible and self._confirm_seen(target):
                return self._found(target, region, turn)
            self.nav.spin(step)
            while self.nav.result() is None:
                self._drive_for(self.config.control_dt)
                seen = self._take_analysis()
                if seen is not None and seen.visible and self._confirm_seen(target):
                    return self._found(target, region, turn)
            if self.nav.result() == ABORTED:
                raise _StepEnded(c.BLOCKED, "PATH_BLOCKED", "spin made no progress")
        self._observe("entity_absent", target.name, {"in": region.name}, None)
        raise _StepEnded(c.BLOCKED, "OBJECT_NOT_FOUND", f"{target.name} not seen in {region.name} after a full turn")

    def _found(self, target, region, turn):
        self._observe("entity_located", target.name, {"in": region.name}, self._confirmed)
        return f"found {target.name} in {region.name} after {turn} turns (live feed)"

    def _confirm_seen(self, target):
        analysis = self._still_analysis()
        self._confirmed = Detection(analysis.visible, analysis.visible_p, analysis.source)
        if not analysis.visible:
            self._event("rejected", analysis, f"{target.name} not confirmed on a still image")
        return analysis.visible

    def _accept_arrival(self, entity):
        """Deterministic checks before an "arrived" claim ends the step."""
        self.nav.cancel()
        self.robot.set_cmd_vel(0.0, 0.0)
        if self.config.near_source == "map" and entity.position is not None:
            return (self._is_near(entity), "map distance says not near yet")
        if not self.config.confirm_arrival:
            return True, ""
        analysis = self._still_analysis()
        return validate(analysis.action) == "arrived", f"not confirmed on a still image ({analysis.action})"

    # -- live feed ---------------------------------------------------------------------------
    def _open_feed(self, entity, history):
        """Start streaming frames to the VLM; every analysis asks the live question set about `entity`."""
        self._close_feed()
        instruction, cfg = self.step.instruction, self.config
        self._feed_entity, self._latest_analysis, self._confirmed = entity, None, None
        self.feed = LiveFeed(
            self.camera, self.robot.now,
            lambda image, questions: self.perceiver.analyze(image, questions, entity.name),
            lambda: frame_questions(entity.description, instruction, history, cfg.open_tokens),
            publish=getattr(self.perceiver, "publish_frame", None), feed_hz=cfg.feed_hz, mode=cfg.feed_mode,
            charge_latency=cfg.charge_perception_time)

    def _close_feed(self):
        if self.feed is not None:
            self.feed.close()
            self.feed_stats = self.feed.stats()
            self.on_event({"step_id": self.step.step_id, "event": "feed", **self.feed_stats})
            self.feed = None

    def _pump(self):
        """One control tick of the feed: record any analysis it releases."""
        if self.feed is None:
            return
        analysis = self.feed.tick()
        if analysis is not None:
            self._check_query_cap()
            self._record("analyze", self._feed_entity, analysis.image, analysis, frame_id=analysis.frame_id,
                         action=analysis.action, visible=analysis.visible, where=analysis.where,
                         near=analysis.near, blocked=analysis.blocked, scene=analysis.scene,
                         objects=analysis.objects, target=analysis.target, age_s=round(analysis.age_s, 3),
                         late_s=round(analysis.late_s, 3),
                         p={k: round(v["p_yes"], 4) for k, v in analysis.answers.items() if "p_yes" in v},
                         scores={k: round(v, 4) for k, v in (analysis.action_scores or {}).items()})
            self._latest_analysis = analysis

    def _take_analysis(self, newer_than=-math.inf):
        analysis, self._latest_analysis = self._latest_analysis, None
        return analysis if analysis is not None and analysis.frame_t >= newer_than - 1e-9 else None

    def _next_analysis(self, newer_than=-math.inf):
        """The next analysis of a frame taken at or after `newer_than`; the robot keeps moving meanwhile."""
        while True:
            analysis = self._take_analysis(newer_than)
            if analysis is not None:
                return analysis
            self._drive_for(self.config.control_dt)

    def _still_analysis(self):
        """Stop, settle, and wait for an analysis of a frame taken after the robot settled."""
        self.nav.cancel()
        self.robot.set_cmd_vel(0.0, 0.0)
        self._drive_for(self.config.settle_s)
        return self._next_analysis(newer_than=self.robot.now())

    def _start(self, motion, choice):
        """Send a validated motion to the navigator, replacing the one in progress."""
        event = {"step_id": self.step.step_id, "event": "command", "action": choice.action, "motion": motion.kind}
        if motion.kind == "go_to":
            self.nav.go_to(motion.goal)
            event["goal"] = [round(motion.goal.x, 3), round(motion.goal.y, 3), round(motion.goal.yaw, 3)]
        elif motion.kind == "spin":
            self.nav.spin(motion.angle)
            event["angle_deg"] = round(math.degrees(motion.angle), 1)
        elif motion.kind == "back_up":
            self.nav.back_up(motion.distance)
            event["distance_m"] = motion.distance
        self.on_event(event)

    def _drive_for(self, seconds):
        """Let time pass while the navigator executes its current command (if any)."""
        end = self.robot.now() + seconds
        while self.robot.now() < end - 1e-9:
            self._tick()
            if self.nav.result() is None:
                self.nav.spin_once()
            self.robot.advance(min(self.config.control_dt, end - self.robot.now()))
            self._pump()
            if self.nav.result() == ABORTED:
                raise _StepEnded(c.BLOCKED, "PATH_BLOCKED", "navigation made no progress")

    def _event(self, kind, choice, reason):  # choice: anything with .action
        self.on_event({"step_id": self.step.step_id, "event": kind, "action": choice.action, "reason": reason})

    def _is_near(self, entity):
        if self.config.near_source == "map" and entity.position is not None:
            pose = self.robot.odom()
            distance = math.hypot(entity.position[0] - pose.x, entity.position[1] - pose.y)
            self.on_event({"step_id": self.step.step_id, "event": "near_check", "source": "map",
                           "distance_m": round(distance, 3), "near": distance <= self.config.near_m})
            return distance <= self.config.near_m
        return self._ask("near", entity).visible

    # -- building blocks ---------------------------------------------------------------------
    def _run_nav(self, what):
        while self.nav.result() is None:
            self._tick()
            self.nav.spin_once()
            self.robot.advance(self.config.control_dt)
            self._pump()
        result = self.nav.result()
        if result == ABORTED:
            raise _StepEnded(c.BLOCKED, "PATH_BLOCKED", f"{what}: navigation made no progress")
        if result == CANCELED:
            raise _StepEnded(c.CANCELED, "ESTOPPED", f"{what}: navigation canceled")
        assert result == SUCCEEDED

    def _navigate_to(self, pose):
        self.on_event({"step_id": self.step.step_id, "event": "navigate",
                       "goal": [round(pose.x, 3), round(pose.y, 3), None if pose.yaw is None else round(pose.yaw, 3)]})
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
        return self._ask("visible", entity)

    def _check_query_cap(self):
        if len(self.trace.queries) >= self.config.max_queries:
            raise _StepEnded(c.FAILED, "DECISION_LIMIT", f"more than {self.config.max_queries} perception queries")

    def _record(self, question, entity, image, answer, **fields):
        record = {"question": question, "entity": entity.name, "phrase": entity.description,
                  "confidence": round(answer.confidence, 3), "source": answer.source,
                  "latency_s": round(answer.latency_s, 3), "detail": answer.detail, **fields}
        self.trace.queries.append(record)
        self.on_event({"step_id": self.step.step_id, "event": "answer", **record})
        if self.on_query:
            self.on_query(self.step.step_id, entity.name, image, answer)

    def _ask(self, question, entity):
        """Stop, settle, capture one image and ask the perceiver one closed question about it."""
        self._check_query_cap()
        self.robot.set_cmd_vel(0.0, 0.0)
        self.robot.advance(self.config.settle_s)
        self._tick()
        image = self.camera()
        method = {"visible": "is_visible", "near": "is_near", "where": "where"}[question]
        answer = getattr(self.perceiver, method)(image, entity.description, entity.name)
        if self.config.charge_perception_time and answer.latency_s > 0:
            self.robot.advance(answer.latency_s)  # still standing: the robot waits for its onboard computer
            self._tick()
        if question == "where":
            self._record(question, entity, image, answer, direction=answer.label)
        else:
            self._record(question, entity, image, answer, **{"visible" if question == "visible" else "near": answer.visible})
        return answer

    def _observe(self, kind, subject, value, detection):
        self.observations.append(c.Observation(
            robot_id=self.step.robot_id, mission_id=self.step.mission_id, step_id=self.step.step_id,
            kind=kind, subject=subject, value=value,
            confidence=round(detection.confidence, 3) if detection else 1.0,
            source=detection.source if detection else "executor"))
