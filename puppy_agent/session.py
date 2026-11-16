"""Revision handshake for one persistent simulator session (local JSON mailbox).

The simulator owns this state machine on its main thread. New plans can only be
accepted at an idle checkpoint; no background thread touches Isaac Sim objects.
"""
from copy import deepcopy
import json
from pathlib import Path
import uuid

from .contracts import ContractError
from .execution import ExecutionManager


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


class ExecutionSession:
    def __init__(self, mission_id, *, revision=0, timeout_s=120, max_revisions=3):
        self.mission_id = mission_id
        self.session_id = uuid.uuid4().hex
        self.revision = revision
        self.timeout_s = timeout_s
        self.max_revision = revision + max_revisions
        self.phase = "RUNNING"
        self.token = None
        self.checkpoint = None
        self.seen = set()

    def idle(self, *, pose, sim_time, safety=""):
        self.phase = "IDLE"
        self.token = uuid.uuid4().hex
        self.checkpoint = {"pose": list(pose), "sim_time": sim_time, "safety": safety}
        return self.state()

    def state(self):
        return {"mission_id": self.mission_id, "session_id": self.session_id,
                "active_revision": self.revision, "phase": self.phase,
                "quiescence_token": self.token, "checkpoint": deepcopy(self.checkpoint)}

    def accept(self, command):
        """Return the replacement manager, or None for pause/close; reject stale data."""
        if command.get("command_id") in self.seen or not isinstance(command.get("command_id"), str):
            raise ContractError("duplicate or missing command ID")
        if command.get("mission_id") != self.mission_id or command.get("session_id") != self.session_id:
            raise ContractError("command belongs to another mission/session")
        if self.phase != "IDLE":
            raise ContractError("executor is not idle")
        if (type(command.get("expected_revision")) is not int
                or command["expected_revision"] != self.revision
                or command.get("quiescence_token") != self.token):
            raise ContractError("stale revision or checkpoint token")
        action = command.get("action")
        if action == "activate":
            revision = command.get("revision")
            if type(revision) is not int or revision != self.revision + 1 or revision > self.max_revision:
                raise ContractError("revision must be the next revision within the recovery limit")
            plan = command.get("dispatch")
            if not isinstance(plan, dict) or plan.get("mission_id") != self.mission_id:
                raise ContractError("replacement dispatch mission mismatch")
            if self.checkpoint["safety"]:
                raise ContractError("robot safety check prevents activation")
            manager = ExecutionManager(plan, plan_revision=revision, timeout_s=self.timeout_s)
            self.revision, self.phase, self.token = revision, "RUNNING", None
        elif action == "close":
            self.phase, self.token, manager = "CLOSED", None, None
        elif action == "pause":
            # Already idle: acknowledge without changing pose, revision or token.
            manager = None
        else:
            raise ContractError("unknown session command")
        self.seen.add(command["command_id"])
        return manager


class Mailbox:
    def __init__(self, directory):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)
        if (self.root / "status.json").exists():
            raise ContractError("session directory already used; a fresh scene needs a fresh session directory")
        for name in ("commands", "acks", "reports"):
            (self.root / name).mkdir(exist_ok=True)

    def publish(self, session):
        write_json(self.root / "status.json", session.state())

    def next_command(self, session, measure=None):
        for path in sorted((self.root / "commands").glob("*.json")):
            ack = self.root / "acks" / path.name
            if ack.exists():
                continue
            try:
                command = json.loads(path.read_text())
                if not isinstance(command, dict) or command.get("command_id") != path.stem:
                    raise ContractError("command ID must match its mailbox filename")
                if measure and command.get("action") == "activate":
                    fresh = measure()
                    if fresh != session.checkpoint:
                        session.idle(**fresh)
                        self.publish(session)
                        raise ContractError("physical state changed after checkpoint; replan before activation")
                manager = session.accept(command)
                if measure and command.get("action") == "pause":
                    session.idle(**measure())  # fresh telemetry and a new single-use checkpoint token
                self.publish(session)
                write_json(ack, {"accepted": True, "command_id": command["command_id"], **session.state()})
                return manager
            except (ValueError, TypeError, KeyError) as error:
                write_json(ack, {"accepted": False, "error": str(error), **session.state()})
        return None
