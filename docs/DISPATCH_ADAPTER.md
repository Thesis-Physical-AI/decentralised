# Central dispatch and execution bridge

Multi-robot support is an intentional requirement. The adapter preserves every
robot assignment and cross-robot dependency; it does not select a PuppyPi subset.

## Contract mapping

| Boundary | Handling |
| --- | --- |
| Central `id` / `robot` | Edge `step_id` / `robot_id` |
| Missing execution metadata | Caller supplies revision and timeout; attempt defaults to 1, version to 0.1.0 |
| Implicit per-robot list ordering | Add predecessor dependencies per robot |
| Cross-robot waits | Preserve and topologically sort; forward references across robots are valid |
| Symbolic versus geometric world | Keep simulation geometry separate; caller ensures matching entity identities |
| Unsupported local actions | Existing executor returns REJECTED; adapter does not invent capabilities |

`adapt_dispatch` expects an already centrally validated plan with empty
`unassigned`. It checks transport structure and the dependency graph, not physical
feasibility. World and allocation validation remain central responsibilities.
The original dispatch is not mutated. The converted mission contains assignments,
not a replacement global world snapshot.

`ExecutionManager` routes ready steps to a mapping of robot IDs to synchronous
executor callables. It preserves global dependencies and serializes execution in
this first implementation, including independent robots. It correlates feedback
by mission, revision, step, robot and attempt, checks status/reason invariants, and
records results and observations. It rejects duplicate or mismatched feedback.
Missing executors leave work pending; no synthetic peer successes are produced.
Any unsuccessful result halts new dispatches, including independent work, until
a later recovery/replanning mechanism is implemented. Exceptions or malformed
feedback propagate without retry; the assignment stays in flight.

## Run in Isaac Sim

From the edge repository root:

```bash
$ISAACSIM_PATH/python.sh run_prototype.py \
  --dispatch missions/dispatch_one_room.json --plan-revision 0 \
  --step-timeout-s 120 --perceiver ground_truth --headless
```

This saves `results.json` and `events.jsonl` under the usual run directory (or
`--out`). The execution report contains mission status SUCCEEDED, HALTED or PENDING
and each result or NOT_DISPATCHED reason. The simulation registers only its local
PuppyPi executor; multi-robot plans remain intact but peers need their own adapters
registered through the manager API before their work can execute.

The one-room fixture is hand-authored central-format JSON, not a Gemini-generated
plan. It passes the central symbolic validator without warnings. The toy is
intentionally absent from the physical scene: that search should block and halt
later work. Manipulation rejection is tested separately offline.

Standalone conversion remains available, and refuses to overwrite output:

```bash
python3 -m puppy_agent.dispatch /path/to/dispatch.json \
  --plan-revision 0 --timeout-s 120 --output /tmp/mission.json
```

Use `--dispatch` for managed execution. The existing `--mission` demo retains its
original local dependency behavior; it is not a fleet execution entry point.

## Impact and remaining scope

Central planning, VLM inference, navigation, gait and executor actions are unchanged.
Legacy missions default to revision 0 and keep their own timeouts. The new dispatch
path uses the caller's uniform timeout policy and stops globally on failure.
Revision correlation is implemented; persistence, freshness across manager
instances, cancellation, retries, parallel dispatch, ROS 2 transport, central world
updates and automatic replanning remain future work. Observations are recorded,
not automatically trusted or merged into the central world.

Offline checks: 102 unit tests pass, including fleet routing with the real PuppyPi
executor and a stub peer, cross-robot waits, missing executors, failure gating,
feedback correlation and preservation of the central worked example's full fleet.

## Isaac Sim integration evidence — 15 November 2026

On Isaac Sim 6.1.0 / RTX 4080 with ground-truth perception, the saved fixture
completed in roughly 390 wall-clock seconds:

| Step | Result | Simulation seconds |
| --- | --- | --- |
| S1 move to table | SUCCEEDED | 32.2 |
| S2 search apple | SUCCEEDED | 0.4 |
| S3 move to crate | SUCCEEDED | 53.05 |
| S4 search absent toy | BLOCKED / OBJECT_NOT_FOUND | 12.8 |
| S5 inspect table | NOT_DISPATCHED after halt | — |

The manager report is HALTED and preserves the executor observations and identity.
Full local artifacts: `/tmp/physical-ai-dispatch-complete/results.json` and
`/tmp/physical-ai-dispatch-complete/events.jsonl`. A preceding 180-second trial
was cut short during S3; the complete run used a 600-second wall limit.

The completed run revealed that SimulationApp's fast shutdown exits the process
before a subsequent SystemExit can run. The runner now passes the desired exit
code directly to `SimulationApp.close(exit_code=...)`, whose signature was checked
in the installed implementation. That small shutdown correction has not been
rerun in Isaac Sim; use the JSON execution status as the authoritative outcome.

The physical check used only PuppyPi. Other robots were exercised via offline
peer stubs, not through PX4, Nav2 or real hardware. This is a dispatch/result
integration demonstration, not a completed closed-loop replanning demonstration.
