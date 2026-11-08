# Prototype: PuppyPi step executor with SmolVLM perception in Isaac Sim

Status: **🧪 experimental** (architecture §13). First working decentralized loop: a dispatched step goes in, the
PuppyPi navigates and looks with its camera in Isaac Sim, a perceiver decides, and exactly one contract v0.1 result
comes out. No ROS 2 yet; built so Nav2 can replace the built-in navigator without changing the executor.

## How it fits the architecture

```
missions/*.json (contract v0.1 steps)            ← later: ExecuteStep action from the centralized Execution Manager
        │
run_prototype.py   dispatches steps whose depends_on succeeded (stand-in for the Execution Manager)
        │
puppy_agent/executor.py      ⑥ Step Executor: validate → act → one StepResult; timeouts, query cap, fall check
   │            │                 │
   │   puppy_agent/perception.py  ⑦ VLA slot: closed yes/no questions ("is a red ball visible?")
   │       ├─ VlmPerceiver ──HTTP──▶ vlm_server/smolvlm_server.py (SmolVLM, separate venv)
   │       └─ GroundTruthPerceiver   (baseline / oracle from sim geometry)
   │
puppy_agent/navigation.py    ⑧ Navigator interface shaped like Nav2 (go_to ≈ NavigateToPose, spin ≈ Spin)
   │       └─ SimpleNavigator (prototype)      later: Nav2Navigator (rclpy action client)
   │  set_cmd_vel(vx, wz) ≈ /cmd_vel     odom() ≈ /odom
sim/puppy_sim.py             ⑨ Isaac Sim driver: trot gait from velocity, front RGB camera, ground-truth odom
```

The model is never asked to plan or choose an action. The step says *what* to do; the executor asks narrow visual
questions whose answers decide the outcome. The VLM server scores the next token "Yes" against "No", so every
answer is valid and has a probability.

| Action | Behaviour | Outcome |
|---|---|---|
| `movetowards{target}` | navigate to the target's approach pose, confirm it is in view | SUCCEEDED, or BLOCKED/OBJECT_NOT_FOUND, BLOCKED/PATH_BLOCKED |
| `search{object, area}` | go to the area (if it has a pose), look in 8 × 45° turns | SUCCEEDED + `entity_located` observation, or BLOCKED/OBJECT_NOT_FOUND + `entity_absent` |
| `inspect{target}` | as `movetowards` | |
| `open, close, grab, putinto, puton, push` | not available without a manipulator | REJECTED/UNKNOWN_ACTION |

## Run it

```bash
# Unit tests (no GPU, no Isaac Sim): 28 tests
python -m pytest -q tests/unit

# Without a model: ground-truth perceiver
$ISAACSIM_PATH/python.sh run_prototype.py --perceiver ground_truth --headless --save-images

# With SmolVLM: start the server in the VLM venv, then run the sim
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # once
.venv/bin/python vlm_server/smolvlm_server.py --model models/SmolVLM-Instruct &
$ISAACSIM_PATH/python.sh run_prototype.py --perceiver vlm --headless --save-images
```

Results go to `runs/<timestamp>-<perceiver>/results.json` (git-ignored), with each perception image if
`--save-images` is set. A run takes ~6.5 min wall time on an RTX 4080 (see limitations).

## Results (2026-11-08, Isaac Sim 6.1.0, RTX 4080, SmolVLM-Instruct 2.2B bf16)

Mission `missions/demo_one_room.json` in `sim/worlds/one_room.json`:

| Step | Ground-truth perceiver | SmolVLM perceiver |
|---|---|---|
| S1 `movetowards table` | SUCCEEDED | SUCCEEDED (p_yes 0.980) |
| S2 `search apple in living_room` | SUCCEEDED, 0 turns | SUCCEEDED, 0 turns (p_yes 0.989; ball partly hidden behind the table edge) |
| S3 `movetowards crate` | SUCCEEDED | SUCCEEDED (p_yes 0.988) |
| S4 `search toy in living_room` (toy absent) | BLOCKED/OBJECT_NOT_FOUND, 8 views | BLOCKED/OBJECT_NOT_FOUND, 8 views (p_yes ≤ 0.029) |
| S5 `grab apple` | REJECTED/UNKNOWN_ACTION | REJECTED/UNKNOWN_ACTION |

- SmolVLM agreed with ground truth on **11/11** queries; median latency **0.26 s**, 4.8 GB GPU memory.
- Navigation arrived within ~5 cm of each approach pose (S1 end pose (0.716, 0.754) vs. goal (0.75, 0.8)).
- Phrasing matters: on the same image SmolVLM answered **no** to "a red apple" (p 0.001) and **yes** to "a red ball"
  (p 0.998), because sim objects are primitives. Entity descriptions must describe appearance.

This is one scene with easy, distinct primitives — evidence that the loop works, not an evaluation. The evaluation
(architecture §12) needs varied scenes, hard negatives, and more steps.

## Path to Nav2 (roadmap P4–P5)

Nav2 is **not installed** on the workstation (`ros-jazzy-navigation2` missing). The seams it needs are in place:

| Nav2 needs | Prototype today | To do |
|---|---|---|
| `/cmd_vel` (geometry_msgs/Twist) → robot | `PuppySim.set_cmd_vel(vx, wz)` | Isaac Sim ROS 2 bridge subscriber calling it |
| `/odom` + TF `odom → base_link` | `PuppySim.odom()` (ground truth) | publish via the bridge |
| `/clock` | sim time `PuppySim.now()` | bridge clock publisher; `use_sim_time:=true` |
| Laser scan for costmaps | none | RTX lidar on the base, or depth → scan |
| Map | walls/boxes in the world file | generate an occupancy grid from `sim/worlds/*.json` for `map_server` |
| NavigateToPose / Spin action servers | `SimpleNavigator` | `Nav2Navigator` (rclpy `ActionClient`) implementing the same `Navigator` protocol |
| Velocity limits | gait limits `MAX_VX` ≈ 0.17 m/s, `MAX_WZ` ≈ 0.98 rad/s | set in the Nav2 controller params |

The executor and the perceivers do not change.

## Limitations

- `SimpleNavigator` drives in a straight line with **no obstacle avoidance** (that is Nav2's job); the demo routes are clear.
- Odometry is ground truth from the simulator.
- ~3.5× slower than real time: the app (including the camera render product) updates every 1/120 s physics step.
- One room, primitive shapes, one camera, fixed lighting.
- The VLM link is local HTTP, to become a ROS 2 service; timestamps are wall-clock, not ROS time.
- `run_prototype.py` dispatches steps itself; in the full system the centralized Execution Manager does this.
- **Contract gap:** v0.1 has no reason code for "action exists for this robot type but this unit lacks the hardware";
  the prototype uses `UNKNOWN_ACTION`. To be raised as a discrepancy / ADR.
