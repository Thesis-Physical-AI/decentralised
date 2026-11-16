import json
import tempfile
from pathlib import Path

import pytest

from puppy_agent.contracts import ContractError, StepAssignment
from puppy_agent.session import ExecutionSession, Mailbox, write_json

from .test_dispatch import dispatch
from .test_executor import make


def activation(session, plan=None):
    return {"command_id": "activate-1", "action": "activate", "mission_id": session.mission_id,
            "session_id": session.session_id, "expected_revision": session.revision,
            "quiescence_token": session.token, "revision": session.revision + 1,
            "dispatch": plan or dispatch()}


def test_revision_continues_same_robot_without_repeating_previous_results():
    plan = dispatch()
    executor, robot, _ = make([True, True])
    session = ExecutionSession(plan["mission_id"])
    first = executor.execute(StepAssignment.from_dict({
        "mission_id": plan["mission_id"], "plan_revision": 0, "step_id": "S0", "task_id": "T0",
        "robot_id": "puppypi_1", "action": "movetowards", "args": {"target": "table"},
        "instruction": "[movetowards] table", "timeout_s": 120}), "puppypi_1")
    assert first.status == "SUCCEEDED"
    before = robot.odom(), robot.now()
    session.idle(pose=[before[0].x, before[0].y, before[0].yaw], sim_time=before[1])
    plan["steps"] = plan["steps"][:1]
    manager = session.accept(activation(session, plan))
    report = manager.run({"puppypi_1": lambda step: executor.execute(step, "puppypi_1")})
    assert report["status"] == "SUCCEEDED"
    assert report["plan_revision"] == 1
    assert robot.now() >= before[1] and robot.odom().x == before[0].x
    assert len(report["steps"]) == 1


@pytest.mark.parametrize("field,value", [("expected_revision", -1), ("revision", 2),
                                       ("session_id", "other"), ("mission_id", "other"),
                                       ("quiescence_token", "old")])
def test_rejects_stale_activation_without_modifying_session(field, value):
    session = ExecutionSession("M-INTEGRATION")
    session.idle(pose=[0, 0, 0], sim_time=3)
    command = activation(session)
    command[field] = value
    before = session.state()
    with pytest.raises(ContractError):
        session.accept(command)
    assert session.state() == before


def test_requires_idle_and_rejects_duplicate_and_unsafe_activation():
    session = ExecutionSession("M-INTEGRATION")
    with pytest.raises(ContractError):
        session.accept(activation(session))
    session.idle(pose=[0, 0, 0], sim_time=1, safety="fell")
    with pytest.raises(ContractError):
        session.accept(activation(session))
    session.idle(pose=[0, 0, 0], sim_time=1)
    command = activation(session)
    session.accept(command)
    session.idle(pose=[0, 0, 0], sim_time=2)
    with pytest.raises(ContractError):
        session.accept(command)


def test_mailbox_pause_activate_close_and_acknowledgments():
    with tempfile.TemporaryDirectory() as folder:
        mailbox = Mailbox(folder)
        session = ExecutionSession("M-INTEGRATION", max_revisions=1)
        session.idle(pose=[.8, 0, 0], sim_time=10)
        mailbox.publish(session)
        command = activation(session)
        command.update(action="pause", command_id="01-pause")
        write_json(Path(folder) / "commands/01-pause.json", command)
        assert mailbox.next_command(session) is None
        assert json.loads((Path(folder) / "acks/01-pause.json").read_text())["accepted"]
        command = activation(session)
        write_json(Path(folder) / "commands/activate-1.json", command)
        assert mailbox.next_command(session) is not None
        session.idle(pose=[.8, 0, 0], sim_time=11)
        command = activation(session)
        command["command_id"] = "02-over-limit"
        write_json(Path(folder) / "commands/02-over-limit.json", command)
        mailbox.next_command(session)
        assert not json.loads((Path(folder) / "acks/02-over-limit.json").read_text())["accepted"]
        command.update(action="close", command_id="03-close")
        write_json(Path(folder) / "commands/03-close.json", command)
        mailbox.next_command(session)
        assert session.phase == "CLOSED"
        with pytest.raises(ContractError):
            Mailbox(folder)


def test_pause_refreshes_token_and_activation_detects_physical_drift():
    with tempfile.TemporaryDirectory() as folder:
        mailbox = Mailbox(folder)
        session = ExecutionSession("M-INTEGRATION")
        session.idle(pose=[0, 0, 0], sim_time=2)
        old_token = session.token
        command = activation(session)
        command.update(action="pause", command_id="pause")
        write_json(Path(folder) / "commands/pause.json", command)
        mailbox.next_command(session, lambda: {"pose": [0, 0, 0], "sim_time": 2, "safety": ""})
        assert session.token != old_token
        command = activation(session)
        write_json(Path(folder) / "commands/activate-1.json", command)
        mailbox.next_command(session, lambda: {"pose": [1, 0, 0], "sim_time": 2, "safety": ""})
        ack = json.loads((Path(folder) / "acks/activate-1.json").read_text())
        assert not ack["accepted"]
        assert session.revision == 0 and session.phase == "IDLE"
