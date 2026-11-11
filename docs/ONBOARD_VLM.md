# Onboard VLM: which model can run on the PuppyPi's Raspberry Pi 5

Status: **🧪 experimental**, branch `feat/onboard-vlm-pi5` (2026-11-09). Goal set by the team: the robot's VLM runs
on the robot itself (Raspberry Pi 5 on the PuppyPi), and the simulation behaves as if it did. Architecture context:
centralized `docs/SYSTEM_ARCHITECTURE.md` §6, §9, §10 and `docs/DECISIONS.md` D2.

## Recommendation

**SmolVLM-500M-Instruct, image splitting off, fp32**, with the calibrated thresholds in
`THRESHOLDS` in [`run_prototype.py`](../run_prototype.py) (visible ≥ 0.0151, near ≥ 0.3203).

- It is the only small model that answers all three questions the robot asks well enough: is the target there
  (held-out balanced accuracy 0.82), which side is it on (0.75, chance 0.44), and is it close (0.83). The 256M model
  matches it on "is it there" but its "is it close" answer is no better than a coin (0.54), so it cannot stop a
  VLM-steered approach by itself.
- It takes **1.70 s per question** on the emulated Pi (estimated **~3.0 s** on a real Pi 5) and peaks at
  **3.6 GB RAM**: it fits a Pi 5 with 8 GB or 16 GB.
- **If the lab's PuppyPi has the 4 GB Pi 5** (unknown, D4), use **SmolVLM2-500M-Video-Instruct** instead
  (2.6 GB, 1.89 s, similar accuracy, more false "yes" answers). Neither 2.2B model nor image splitting is usable
  onboard.
- For the stop decision there is a fallback that needs no model: `--near-source map` stops by distance to the
  target's map position. Use it with any model whose "is it close" answer is weak.
- **The model choice is for the onboard computer; the VLM-steered approach is not reliable yet.** In the Isaac Sim
  demo (below) SmolVLM-500M answered "is it there" correctly and steered toward a visible target, but a false
  "blue box" next to a blue-tiled floor sent the robot the wrong way, and its "is it close" answer stopped it too
  early or not at all. Until those are fixed, use the VLM for `search` and confirmation, and the map approach (or
  `--near-source map`) to reach targets.

The numbers marked *estimated* are not measurements on a Pi. They must be confirmed on the real board (next steps).

## What the robot asks the model

Every question is closed and answered by scoring the next token, so every answer is valid
([`vlm_server/smolvlm.py`](../vlm_server/smolvlm.py)):

| Question | Prompt | Used for |
|---|---|---|
| visible | "Is there {phrase} in this image? Answer with yes or no." | `search`, confirming arrival |
| where | the visible question on three overlapping vertical crops (left 0–45 %, center 27.5–72.5 %, right 55–100 % of the width); the best crop wins, "none" if no crop passes the threshold | choosing the next navigation goal |
| near | "Is {phrase} close to the camera, taking up a large part of the image? Answer with yes or no." | stopping the visual approach |

Tried and rejected: asking for the side in words ("Answer with one word: left, center, or right"), which scored
0.32–0.61, below the crop method for every model; and judging "near" by asking the visible question on an image
shrunk 8× (far objects vanish), which did not separate near from far (AUC 0.53–0.66 on the 256M model).

## Method

### Emulating the Raspberry Pi 5

`PROFILES` in [`vlm_server/smolvlm_server.py`](../vlm_server/smolvlm_server.py), profile `pi5_emulated`:

| Setting | Value | Measured or assumed |
|---|---|---|
| Device, precision | CPU only, fp32 | Matches the Pi: no CUDA GPU; the Cortex-A76 has no bf16 instructions |
| Threads, cores | 4 threads, pinned to CPUs 16–19 of the i9-13900F (efficiency cores, max 4.2 GHz) | The Pi 5 has 4 Cortex-A76 cores at 2.4 GHz. E-cores were chosen because they are closer to an A76 than the 5.6 GHz performance cores |
| Image splitting | off | Measured: on costs ~12× more time per question (below) |
| Latency scale | ×1.75 | **Assumed**: clock ratio 4.2 / 2.4 GHz with equal work per clock. The Pi's lower memory bandwidth is not modelled, so real latency is likely higher |
| RAM budget | none enforced; reported against 4 and 8 GB | **Unknown**: the PuppyPi's Pi 5 RAM variant is open (D4) |

The VLM server applies the profile (`--profile pi5_emulated`): it pins itself to the profile's cores, sets the
threads and precision, and returns each answer's `onboard_latency_s` = measured × 1.75. The step executor then
**advances simulation time by that latency with the robot standing still**, so a slow onboard computer makes the
robot slower and can trigger step timeouts, as it would on the real robot.

No published SmolVLM latency for the Pi 5 was found (searched 2026-11-09), so no published number could be used
instead of the assumed scale.

### Labelled images

`sim/capture_dataset.py` (no longer in the working tree; in git at commit 913239e) teleported the PuppyPi to 120 random poses in
`sim/worlds/one_room_eval.json` (seed 0; about half aimed near an object) and
saves the 320×240 front-camera image. That world is `one_room` plus the toy placed in the scene and two decoys that
share a colour or a shape with the targets (a green box, an orange ball). Ground truth comes from Isaac Sim's
`bounding_box_2d_tight` annotator: the box of each object's **visible** pixels, so occlusion is included.

- visible: box ≥ 150 px (about 12×12); slivers of 1–149 px are left out as ambiguous
- where: box centre in the left, middle or right third; centres within 4 % of a third-boundary are left out
- near: box height ≥ 25 % of the image height

Result: 120 images; 661 presence questions (147 visible, 514 absent), 122 where questions, 147 near questions
(32 near). Labels are never shown to the model. The images were git-ignored; regenerating them needs
`sim/capture_dataset.py` and `sim/worlds/one_room_eval.json` from git (commit 913239e).

### Measurements

`bench/vlm_bench.py` (no longer in the working tree; in git at commit 913239e):

- **accuracy** — every question on every image. Run on the GPU in fp32 (the Pi's precision; accuracy does not
  depend on which chip does the arithmetic) except int8 models, which run on the CPU.
- **thresholds** — small models put most of their probability on "no", so 0.5 is the wrong cut-off. Each
  model's threshold is chosen on the even-numbered images and the reported balanced accuracy is measured on the
  odd-numbered ones (held out).
- **latency** — each model in its own process under the profile: 1 warm-up round, then 6 timed rounds (3 for
  the 2.2B model) of visible + where + near; median reported. Peak RAM is the process's maximum resident set size.

```bash
.venv/bin/python bench/vlm_bench.py accuracy --models models/SmolVLM-500M-Instruct --splitting 0
.venv/bin/python bench/vlm_bench.py latency --profile pi5_emulated --models models/SmolVLM-500M-Instruct --splitting 0
.venv/bin/python bench/vlm_bench.py report      # tables below + bench/results/thresholds.json
```

## Results

Raw results: `bench/results/` (one JSON line per question; latency JSON), in git at commit 913239e.

### Accuracy (120 labelled sim images)

| Model | Image splitting | Visible: AUC | Visible: held-out balanced acc. (recall / specificity), threshold | Look-alike false "yes" (target absent, decoy in view) | Where (crops), chance 0.44 | Near: AUC | Near: held-out balanced acc. |
|---|---|---|---|---|---|---|---|
| SmolVLM-256M-Instruct | off | 0.92 | 0.83 (0.72 / 0.95), 0.0102 | 14 / 164 | 0.71 | 0.61 | 0.54 |
| SmolVLM-256M-Instruct | on | 0.91 | 0.90 (0.85 / 0.95), 0.0108 | 20 / 164 | 0.68 | 0.66 | 0.63 |
| **SmolVLM-500M-Instruct** | **off** | 0.87 | **0.82 (0.79 / 0.86), 0.0151** | 34 / 164 | **0.75** | 0.80 | **0.83** |
| SmolVLM-500M-Instruct | on | 0.89 | 0.85 (0.82 / 0.88), 0.0031 | 30 / 164 | 0.71 | 0.81 | 0.74 |
| SmolVLM-Instruct (2.2B) | off | 0.92 | 0.90 (0.85 / 0.96), 0.422 | 14 / 164 | 0.72 | 0.85 | 0.78 |
| SmolVLM-Instruct (2.2B) | on | 0.95 | 0.92 (0.88 / 0.96), 0.707 | 11 / 164 | 0.64 | 0.84 | 0.78 |
| SmolVLM2-256M-Video-Instruct | off | 0.82 | 0.71 (0.58 / 0.83), 0.0664 | 21 / 164 | 0.44 | 0.63 | 0.68 |
| SmolVLM2-500M-Video-Instruct | off | 0.86 | 0.80 (0.82 / 0.79), 0.648 | 53 / 164 | 0.79 | 0.73 | 0.76 |

int8 (PyTorch dynamic quantization of every Linear layer, qnnpack engine, run on the CPU), image splitting off:

| Model | Visible: AUC | Visible: held-out balanced acc. (recall / specificity), threshold | Look-alike false "yes" | Where (crops) | Near: AUC | Near: held-out balanced acc. |
|---|---|---|---|---|---|---|
| SmolVLM-256M-Instruct, int8 | 0.82 | 0.77 (0.71 / 0.83), 0.172 | 33 / 164 | 0.53 | 0.58 | 0.61 |
| SmolVLM-500M-Instruct, int8 | 0.84 | 0.77 (0.67 / 0.87), 0.151 | 36 / 164 | 0.71 | 0.63 | 0.64 |

int8 costs accuracy on every question, most on "near" for the 500M model (0.83 → 0.64) and on "where" for the 256M
model (0.71 → 0.53).

### Speed and memory on the emulated Pi 5

| Model | Image splitting | Peak RAM (GB) | Visible: s per question (measured) | Where, 3 crops: s (measured) | Visible: s on a Pi 5 (*estimated*, ×1.75) | Fits Pi 5 4 GB / 8 GB |
|---|---|---|---|---|---|---|
| SmolVLM-256M-Instruct | off | 2.19 | 1.48 | 4.38 | 2.6 | tight / yes |
| SmolVLM-256M-Instruct | on | 2.41 | 18.10 | 54.37 | 31.7 | tight / yes |
| SmolVLM-256M-Instruct, int8 | off | 2.79 | 2.88 | 8.51 | 5.0 | tight / yes |
| **SmolVLM-500M-Instruct** | **off** | **3.59** | **1.70** | **5.06** | **3.0** | no / yes |
| SmolVLM-500M-Instruct | on | 3.59 | 20.19 | 60.53 | 35.3 | no / yes |
| SmolVLM-500M-Instruct, int8 | off | 4.66 | 3.43 | 10.25 | 6.0 | no / yes |
| SmolVLM-Instruct (2.2B) | off | 13.31 | 7.06 | 21.10 | 12.4 | no / no (16 GB only) |
| SmolVLM2-256M-Video-Instruct | off | 1.76 | 1.62 | 4.84 | 2.8 | yes / yes |
| SmolVLM2-500M-Video-Instruct | off | 2.62 | 1.89 | 5.61 | 3.3 | tight / yes |

"Tight" means the model alone uses more than half of the RAM; the OS, ROS 2 and the camera pipeline also need memory.
"Near" takes the same time as "visible" (one forward pass).

### What the numbers say

1. **Image splitting is out onboard.** It cuts each 320×240 frame into 13 tiles (≈830–1050 image tokens instead of
   64–81) and costs 12× more time for at most a few points of accuracy.
2. **The 2.2B model is out onboard**: 13.3 GB and ~12 s per question estimated.
3. **int8 dynamic quantization (PyTorch, qnnpack) was slower, larger and less accurate** than fp32 in this
   emulation. qnnpack is tuned for ARM and slow on x86, so this does not show int8 is slow on the Pi; it shows the
   gain cannot be assumed and that this quantization method loses accuracy. Measure on the Pi, and try llama.cpp /
   ONNX Runtime quantized builds (next steps).
4. **Small models are badly calibrated, not blind.** At the default 0.5 cut-off SmolVLM-256M finds only 17 % of
   visible objects, yet its AUC is 0.92; with a calibrated cut-off it finds 72 % on held-out images. Every model
   needs its own thresholds, which `run_prototype.py` reads from `bench/results/thresholds.json` automatically.
5. **False "yes" next to look-alikes is the main weakness of the 500M models** (21 % for SmolVLM-500M, 32 % for
   SmolVLM2-500M when a decoy is in view). A wrong "found" corrupts the center's world model, so a found report
   should be confirmed with a second question (e.g. from a closer pose) before it is sent.
6. **A full where answer costs three questions** (~9 s estimated on a Pi). A VLM-steered approach therefore moves
   in steps of several seconds of thinking per 0.4 m; step timeouts written for the workstation are too short.

## Demo in Isaac Sim (VLM picks the goals, emulated Pi latency)

Setup: mission `missions/demo_one_room.json` in `sim/worlds/one_room.json` (toy deliberately absent), SmolVLM-500M
server on `pi5_emulated` (4 pinned E-cores, latency ×1.75 charged to robot time), Isaac Sim 6.1.0 headless pinned to
CPUs 0–15, `--approach visual`. Each "where" answer cost ~10 s of robot time and each yes/no ~3.5 s. Runs are in
`runs/` (git-ignored) with every image and `events.jsonl`.

| Step | Run A: calibrated thresholds (visible 0.0151, near 0.3203), VLM decides "near" | Run B: thresholds 0.5, VLM decides "near" | Run C: thresholds 0.5, `--near-source map` (stop within 0.6 m) |
|---|---|---|---|
| S1 movetowards table (90 s) | SUCCEEDED after 1 VLM goal, 28 s; stopped **~1.0 m** from the table (VLM said near, p 0.58) | **TIMED_OUT**: reached the table edge, VLM never said near (p 0.39), then lost the table filling the view and turned | **SUCCEEDED** after 4 VLM goals, 89 s, 0.47 m from the table centre |
| S2 search apple (90 s) | SUCCEEDED, first look (p 0.97) | not dispatched | BLOCKED/OBJECT_NOT_FOUND: robot too close, the table top is out of the camera's view (VLM correctly said no 8×) |
| S3 movetowards crate (120 s) | **TIMED_OUT**: crate behind the robot, but the VLM reported "a blue box" ahead (crop p up to 0.58; the scene has a blue-tiled floor) and the robot walked the wrong way | not dispatched | not dispatched |
| S4 search toy (absent) | not dispatched (S3) | not dispatched | not dispatched |
| S5 grab apple | REJECTED/UNKNOWN_ACTION (no arm) | not dispatched | not dispatched |

What this shows:

- The loop works end to end: the VLM's left/center/right answers became goals the navigator drove to, and every
  answer cost its estimated Pi latency in robot time.
- **False "yes" is the blocking problem.** With the calibrated (low) threshold a crop almost never reads "not in
  view", so the robot never turns to search; a target that is behind it is "seen" in front. Raising the threshold
  fixes that but loses the target when it is close and fills the view.
- **The stop decision needs more than the VLM.** The VLM's "near" stopped too early (A) or never (B). Map distance (C)
  works, but stopping must also leave the target's surface in view (S2 in C).
- **Onboard latency eats the step budget.** At ~14 s of robot time per goal, S1 in run C used 89 of 90 s.
  Workstation-sized timeouts are too short for a Pi-hosted VLM.

Before the visual approach can replace the map approach: confirm a "where" with a full-image yes/no question
before moving, stop at a distance chosen per target type (the map's approach distance), give the center
step timeouts that include onboard latency, and evaluate in scenes whose floor and walls are not look-alikes of the
targets.

## Scaling to other robots

The VLM questions and the executor are robot-independent. A robot type adds:

1. an **onboard profile** in `config/onboard_profiles.json` (its board: device, precision, cores, latency scale,
   RAM), and
2. a **goal adapter** (`puppy_agent/visual_goal.py`) that turns "the target is at this bearing" into a goal for
   its navigation stack: `GroundGoalAdapter` → Nav2 `NavigateToPose` pose for the PuppyPi (and a TurtleBot); a
   drone adapter would return a PX4 position setpoint holding altitude (not written yet; drone hardware is open, D5).

## Limitations

- **Pi latency is estimated**, from an assumed ×1.75 scale; memory bandwidth and thermal throttling are not modelled.
- One room, primitive shapes, fixed lighting, 120 images. Differences of a few points between models are within
  noise at this size (e.g. 4 of 122 where questions separate 0.75 from 0.79).
- Ground truth "near" is box height ≥ 25 % of the image, a proxy for distance.
- Thresholds are fitted on 60 images of the same scene; they will need re-fitting for other scenes and the real
  camera.
- PyTorch fp32 only; the deployment runtime on the Pi may differ (llama.cpp, ONNX Runtime).
- The robot waits while the model thinks; it does not move and think at the same time.

## Next steps

1. **Measure on a real Pi 5** (needs the board, and its RAM size for D4): run
   `bench/vlm_bench.py latency --profile <a new pi5 profile without cpu_affinity>` on the Pi, then replace
   `latency_scale` in `pi5_emulated` with *Pi latency ÷ emulated latency*.
2. **Try llama.cpp and ONNX Runtime builds** of SmolVLM-256M/500M (int8 / q4 weights) on the Pi; both are the
   usual ways to run small VLMs on ARM CPUs and may be several times faster than PyTorch fp32.
3. **Confirm "found" reports** with a second question to cut look-alike false yeses.
4. Re-capture the benchmark with richer scenes (textured objects, clutter, lighting changes) and re-fit thresholds.
5. Nav2 in place of `SimpleNavigator` (roadmap P4–P5); the goal adapter already produces `NavigateToPose` goals.
