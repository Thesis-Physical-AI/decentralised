# AGENTS.md — decentralised

Instructions for AI coding agents (and humans) working in this repository.

## Read before changing anything
The system-wide documents live in the `centralized` repo, so there is one copy of each:

1. System architecture (source of truth for **both** repos): [`centralized/docs/SYSTEM_ARCHITECTURE.md`](https://github.com/Thesis-Physical-AI/centralized/blob/main/docs/SYSTEM_ARCHITECTURE.md) — §4 responsibilities (layers ⑥–⑨), §6 model roles, §9 robot execution, §11 safety, §13 status.
2. Interface contracts: [`centralized/docs/INTERFACE_CONTRACTS.md`](https://github.com/Thesis-Physical-AI/centralized/blob/main/docs/INTERFACE_CONTRACTS.md) — the `ExecuteStep` action, results, observations, status, e-stop.
3. Open decisions and known discrepancies: [`centralized/docs/DECISIONS.md`](https://github.com/Thesis-Physical-AI/centralized/blob/main/docs/DECISIONS.md).
4. Workflow rules: [`centralized/docs/AGENT_WORKFLOW.md`](https://github.com/Thesis-Physical-AI/centralized/blob/main/docs/AGENT_WORKFLOW.md).

If both repos are checked out side by side, the same files are at `../centralized/docs/`.

## What this repo owns
Per-robot execution: ⑥ Step Executor, ⑦ Action Selector (the agent's VLA slot — currently a VLM, plus a rule-based baseline), ⑧ capabilities (gait, Nav2 integration, skills, PX4 integration), ⑨ platform drivers (Isaac Sim, later hardware). It does **not** decompose missions, allocate tasks, or own the world model.

Current state (see architecture §13): only `hello_puppy.py` (PuppyPi in Isaac Sim, scripted gait, no ROS) and the SmolVLM download exist. `docs/architecture.dot` is the **target design**, not current state.

## Rules
- Follow the shared rules in `AGENT_WORKFLOW.md` §2. In short: stay in your layer; contract changes need an ADR + version bump; never claim planned work is done; report doc/code discrepancies instead of choosing one; don't weaken tests; no new frameworks without an ADR.
- **Safety first:** motion, balance, stop and timeouts never depend on a model or the network. Selector output is untrusted: validate it against the closed action list before acting.
- Executors accept only actions listed for their robot in `centralized/mission-orchestrator/robots.json` and return exactly one terminal result per step.
- Model weights and large assets are optional. The simulation must run without downloading any model.
- Don't hard-code machine paths (e.g. `~/Projects/isaacsim`). Use `ISAACSIM_PATH`.
- Isaac Sim code runs under Isaac Sim's Python; VLM code under the `.venv`; ROS nodes (later) under ROS 2 Jazzy's Python. Keep pure logic (IK, action mapping, validation) importable without any of them so it can be unit-tested.

## Checks (run before finishing)
There is no automated test suite yet. If you changed simulation code, run the headless check and paste the output into the PR:
```bash
$ISAACSIM_PATH/python.sh hello_puppy.py --headless --check --moves stand walk_forward turn_left
```
Expected today: `CHECKS: ALL PASS`, exit code 0, ~1 minute on an RTX 4080. Note that the joint check is weak (it passes on any joint motion); a stricter smoke test is planned (`centralized/docs/ROADMAP.md`, phase P1).
