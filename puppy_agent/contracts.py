"""Step assignment and step result, following centralized/docs/INTERFACE_CONTRACTS.md (draft v0.1).

Plain dataclasses, no ROS: the same objects become the ExecuteStep action goal/result later.
"""
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

CONTRACT_VERSION = "0.1.0"

SUCCEEDED, FAILED, BLOCKED, TIMED_OUT, CANCELED, REJECTED = (
    "SUCCEEDED", "FAILED", "BLOCKED", "TIMED_OUT", "CANCELED", "REJECTED")

REASON_CODES = {
    "", "UNKNOWN_ACTION", "INVALID_ARGS", "STALE_PLAN", "ROBOT_BUSY", "ESTOPPED", "PRECONDITION_FAILED",
    "OBJECT_NOT_FOUND", "PATH_BLOCKED", "EXECUTION_ERROR", "SAFETY_LIMIT", "DECISION_LIMIT", "COMMS_LOST",
    "TIMEOUT",
}

REQUIRED = ("mission_id", "plan_revision", "step_id", "task_id", "robot_id", "action", "args", "instruction")


class ContractError(ValueError):
    pass


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class StepAssignment:
    mission_id: str
    plan_revision: int
    step_id: str
    task_id: str
    robot_id: str
    action: str
    args: dict
    instruction: str
    depends_on: list = field(default_factory=list)
    attempt: int = 1
    timeout_s: float = 60.0
    contract_version: str = CONTRACT_VERSION

    @classmethod
    def from_dict(cls, data):
        missing = [key for key in REQUIRED if key not in data]
        if missing:
            raise ContractError(f"missing fields: {missing}")
        if not str(data.get("contract_version", CONTRACT_VERSION)).startswith("0.1."):
            raise ContractError(f"unsupported contract_version {data['contract_version']}")
        if not isinstance(data["args"], dict) or not all(isinstance(v, str) for v in data["args"].values()):
            raise ContractError("args must be an object of strings")
        known = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in data.items() if key in known})


@dataclass
class Observation:
    robot_id: str
    kind: str  # entity_located | entity_absent | entity_state | robot_state
    subject: str
    value: dict
    confidence: float
    source: str  # executor | perception | vlm
    mission_id: str = ""
    step_id: str = ""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    stamp: str = field(default_factory=now_iso)


@dataclass
class StepResult:
    mission_id: str
    plan_revision: int
    step_id: str
    robot_id: str
    attempt: int
    status: str
    reason_code: str = ""
    reason: str = ""
    observations: list = field(default_factory=list)
    started_at: str = ""
    ended_at: str = ""
    decision_trace_ref: str = ""
    contract_version: str = CONTRACT_VERSION

    def __post_init__(self):
        if self.reason_code not in REASON_CODES:
            raise ContractError(f"unknown reason_code {self.reason_code!r}")
        if (self.status == SUCCEEDED) != (self.reason_code == ""):
            raise ContractError("reason_code must be empty exactly when status is SUCCEEDED")

    def to_dict(self):
        return asdict(self)
