"""Run a PuppyPi mission (contract v0.1 steps) in Isaac Sim with the step executor.

    # 1. optional, for --perceiver vlm: start the SmolVLM server in the VLM venv
    .venv/bin/python vlm_server/smolvlm_server.py --model models/SmolVLM-Instruct
    # 2. run with Isaac Sim's Python
    $ISAACSIM_PATH/python.sh run_prototype.py --perceiver vlm --headless
    $ISAACSIM_PATH/python.sh run_prototype.py --perceiver ground_truth --headless   # no model needed
    # the VLM picks each navigation goal (left / center / right / near) instead of the map's approach pose
    $ISAACSIM_PATH/python.sh run_prototype.py --perceiver vlm --approach visual --headless
    # live: camera frames stream to the VLM (--feed-hz, >= 1); for each frame it analyses it answers a set of
    # questions (open ones in words, closed ones scored) and picks each command while the robot moves.
    # Watch it at http://127.0.0.1:8765/view
    $ISAACSIM_PATH/python.sh run_prototype.py --perceiver vlm --approach live --headless

Steps run in order; a step is dispatched only if all its depends_on steps SUCCEEDED (a stand-in for
the centralized Execution Manager). Writes runs/<timestamp>/results.json and, with --save-images,
the camera image of every perception query. Every navigation goal and model answer is printed as it
happens and appended to runs/<timestamp>/events.jsonl.
"""
import argparse
import json
import os
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument("--world", default=os.path.join(HERE, "sim", "worlds", "one_room.json"))
parser.add_argument("--mission", default=os.path.join(HERE, "missions", "demo_one_room.json"))
parser.add_argument("--usd", default=os.path.join(HERE, "puppy_isaac", "puppy_isaac.usda"))
parser.add_argument("--perceiver", choices=["vlm", "ground_truth"], default="vlm")
parser.add_argument("--vlm-url", default="http://127.0.0.1:8765")
parser.add_argument("--vlm-threshold", type=float, default=None,
                    help="p_yes needed for 'yes' (overrides --vlm-thresholds; default 0.5 if neither applies)")
parser.add_argument("--vlm-thresholds", default=None,
                    help="optional JSON file of thresholds per model, to use instead of THRESHOLDS")
parser.add_argument("--out", default=None, help="output directory (default: runs/<timestamp>-<perceiver>)")
parser.add_argument("--save-images", action="store_true")
parser.add_argument("--approach", choices=["map", "visual", "live"], default="map",
                    help="movetowards: drive to the map's approach pose, let the VLM pick each goal from closed "
                         "questions (visual), or let it reason and choose each command while moving (live)")
parser.add_argument("--feed-hz", type=float, default=2.0,
                    help="live: camera frames per second of robot time streamed to the VLM (at least 1)")
parser.add_argument("--feed-mode", choices=["async", "lockstep"], default="async",
                    help="live: async = the sim keeps running while the VLM thinks (live in wall-clock time too); "
                         "lockstep = the sim waits for each answer (exact robot-time timing)")
parser.add_argument("--open-tokens", type=int, default=32, help="live: max tokens of each open (worded) answer")
parser.add_argument("--near-source", choices=["vlm", "map"], default="vlm",
                    help="visual approach: stop when the VLM says the target is close, or by map distance")
parser.add_argument("--headless", action="store_true")
args = parser.parse_args()
if args.feed_hz < 1.0:
    parser.error("--feed-hz must be at least 1")

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": args.headless})

from puppy_agent.contracts import SUCCEEDED, StepAssignment  # noqa: E402
from puppy_agent.executor import ExecutorConfig, StepExecutor  # noqa: E402
from puppy_agent.navigation import SimpleNavigator  # noqa: E402
from puppy_agent.perception import GroundTruthPerceiver, VlmPerceiver  # noqa: E402
from puppy_agent.semantic_map import SemanticMap  # noqa: E402
from sim.puppy_sim import CAMERA_HFOV, PuppySim  # noqa: E402


# p_yes thresholds per model and image-splitting setting, calibrated on the 120-image benchmark set
# (docs/ONBOARD_VLM.md): small models put most probability on "no", so 0.5 is the wrong cut-off.
# Picked by the server's model; models not listed fall back to 0.5.
THRESHOLDS = {
    "SmolVLM-256M-Instruct-int8__split0": {"visible": 0.1715, "near": 0.5451},
    "SmolVLM-256M-Instruct__split0": {"visible": 0.0102, "near": 0.0353},
    "SmolVLM-256M-Instruct__split1": {"visible": 0.0108, "near": 0.2969},
    "SmolVLM-500M-Instruct-int8__split0": {"visible": 0.151, "near": 0.2632},
    "SmolVLM-500M-Instruct__split0": {"visible": 0.0151, "near": 0.3203},
    "SmolVLM-500M-Instruct__split1": {"visible": 0.0031, "near": 0.0943},
    "SmolVLM-Instruct__split0": {"visible": 0.4217, "near": 0.4183},
    "SmolVLM-Instruct__split1": {"visible": 0.7071, "near": 0.5255},
    "SmolVLM2-256M-Video-Instruct__split0": {"visible": 0.0664, "near": 0.0656},
    "SmolVLM2-500M-Video-Instruct__split0": {"visible": 0.6478, "near": 0.5831},
}


def main():
    with open(args.world, encoding="utf-8") as f:
        world = json.load(f)
    with open(args.mission, encoding="utf-8") as f:
        mission = json.load(f)
    robot_id = world["robot"]["id"]
    out = args.out or os.path.join(HERE, "runs", datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + args.perceiver
                                   + ("" if args.approach == "map" else "-" + args.approach))
    os.makedirs(out, exist_ok=True)

    sim = PuppySim(simulation_app, world, args.usd)
    if args.perceiver == "vlm":
        perceiver = VlmPerceiver(args.vlm_url)
        server = perceiver.health()
        key = f"{server['model']}{'-int8' if server.get('quantize') else ''}__split{int(server.get('image_splitting', 0))}"
        calibrated = THRESHOLDS.get(key, {})
        if args.vlm_thresholds:
            with open(args.vlm_thresholds, encoding="utf-8") as f:
                calibrated = json.load(f).get(key, {})
        visible = args.vlm_threshold if args.vlm_threshold is not None else calibrated.get("visible", 0.5)
        near = args.vlm_threshold if args.vlm_threshold is not None else calibrated.get("near", 0.5)
        perceiver = VlmPerceiver(args.vlm_url, threshold=visible, near_threshold=near)
        print(f"VLM server: {server['model']} profile={server.get('profile')} thresholds visible={visible:.4g} "
              f"near={near:.4g} ({'calibrated: ' + key if calibrated and args.vlm_threshold is None else 'manual/default'})",
              flush=True)
    else:
        perceiver = GroundTruthPerceiver(sim.locate, sim.camera_pose, hfov_rad=CAMERA_HFOV)

    image_count = [0]

    def save_image(step_id, entity, image, answer):
        if args.save_images:
            from PIL import Image
            label = getattr(answer, "label", None) or getattr(answer, "action", None) or ("yes" if answer.visible else "no")
            Image.fromarray(image).save(os.path.join(out, f"{image_count[0]:03d}_{step_id}_{entity}_{label}.png"))
            image_count[0] += 1

    events = open(os.path.join(out, "events.jsonl"), "w", encoding="utf-8")

    def log_event(event):
        event = {"sim_t": round(sim.now(), 2), **event}
        events.write(json.dumps(event) + "\n")
        events.flush()
        if event["event"] == "answer" and event["question"] == "analyze":
            pad = " " * 22
            text = (f"see #{event['frame_id']} \"{event['phrase']}\": action={event['action']} "
                    f"visible={'yes' if event['visible'] else 'no'} where={event['where']} "
                    f"near={'yes' if event['near'] else 'no'} blocked={'yes' if event['blocked'] else 'no'}  "
                    f"({event['latency_s']} s, frame {event['age_s']} s old"
                    + (f", {event['late_s']} s late" if event["late_s"] else "") + ")"
                    + f"\n{pad}scene: {event['scene']}\n{pad}objects: {event['objects']}\n{pad}target: {event['target']}")
        elif event["event"] == "feed":
            text = (f"feed    {event['frames']} frames, {event['analyses']} analysed in {event['robot_seconds']} s robot time: "
                    f"{event['frame_hz']} frames/s, {event['analysis_hz']} analyses/s, median latency "
                    f"{event['median_latency_s']} s, {event['late']} late")
            if event["analyses"] and event["analysis_hz"] < 1.0:
                text += "  WARNING: below 1 analysis/s"
        elif event["event"] == "command":
            text = f"command {event['action']} -> {event['motion']} " + str(event.get("goal", event.get("angle_deg", event.get("distance_m", ""))))
        elif event["event"] == "rejected":
            text = f"reject  {event['action']}: {event['reason']}"
        elif event["event"] == "answer":
            answer = event.get("direction", event.get("visible", event.get("near")))
            text = f"ask {event['question']:7s} \"{event['phrase']}\" -> {answer}  [{event['detail']}] ({event['latency_s']} s)"
        elif event["event"] == "near_check":
            text = f"near?   map distance {event['distance_m']} m -> {event['near']}"
        else:
            text = f"{event['event']:8s} goal {event['goal']}" + (f" (VLM said {event['from']})" if "from" in event else "")
        print(f"    t={event['sim_t']:7.2f} {event['step_id']} {text}", flush=True)

    config = ExecutorConfig(approach=args.approach, near_source=args.near_source, camera_hfov_rad=CAMERA_HFOV,
                            max_queries={"map": 12, "visual": 60, "live": 1000}[args.approach],
                            feed_hz=args.feed_hz, feed_mode=args.feed_mode, open_tokens=args.open_tokens)
    executor = StepExecutor(sim, SimpleNavigator(sim), perceiver, SemanticMap.from_world(world),
                            camera=sim.capture, safety_check=sim.safety, config=config,
                            on_query=save_image, on_event=log_event)

    status, records = {}, []
    for raw in mission["steps"]:
        step = StepAssignment.from_dict({"mission_id": mission["mission_id"], "plan_revision": 0,
                                         "robot_id": robot_id, **raw})
        blocked_by = [d for d in step.depends_on if status.get(d) != SUCCEEDED]
        if blocked_by:
            status[step.step_id] = "NOT_DISPATCHED"
            records.append({"step_id": step.step_id, "instruction": step.instruction,
                            "status": "NOT_DISPATCHED", "reason": f"dependencies not succeeded: {blocked_by}"})
            print(f"[{step.step_id}] {step.instruction:32s} NOT_DISPATCHED (waiting on {blocked_by})", flush=True)
            continue
        wall, sim_t0 = time.perf_counter(), sim.now()
        result = executor.execute(step, robot_id)
        status[step.step_id] = result.status
        pose = sim.odom()
        record = {"step_id": step.step_id, "instruction": step.instruction, **result.to_dict(),
                  "sim_seconds": round(sim.now() - sim_t0, 2), "wall_seconds": round(time.perf_counter() - wall, 2),
                  "end_pose": [round(pose.x, 3), round(pose.y, 3), round(pose.yaw, 3)],
                  "perception_queries": executor.trace.queries, "live_feed": executor.feed_stats}
        records.append(record)
        line = f"{result.status}" + (f"/{result.reason_code}" if result.reason_code else "")
        print(f"[{step.step_id}] {step.instruction:32s} {line:28s} {result.reason}  "
              f"(sim {record['sim_seconds']} s, {len(executor.trace.queries)} queries)", flush=True)

    events.close()
    summary = {"mission_id": mission["mission_id"], "perceiver": args.perceiver, "approach": args.approach,
               "world": os.path.basename(args.world), "steps": records}
    with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"results: {os.path.join(out, 'results.json')}", flush=True)
    sim.close()


main()
simulation_app.close()
