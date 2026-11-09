"""Run a PuppyPi mission (contract v0.1 steps) in Isaac Sim with the step executor.

    # 1. optional, for --perceiver vlm: start the SmolVLM server in the VLM venv
    .venv/bin/python vlm_server/smolvlm_server.py --model models/SmolVLM-Instruct
    # 2. run with Isaac Sim's Python
    $ISAACSIM_PATH/python.sh run_prototype.py --perceiver vlm --headless
    $ISAACSIM_PATH/python.sh run_prototype.py --perceiver ground_truth --headless   # no model needed
    # the VLM picks each navigation goal (left / center / right / near) instead of the map's approach pose
    $ISAACSIM_PATH/python.sh run_prototype.py --perceiver vlm --approach visual --headless

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
parser.add_argument("--vlm-thresholds", default=os.path.join(HERE, "bench", "results", "thresholds.json"),
                    help="calibrated thresholds per model (bench/vlm_bench.py report); picked by the server's model")
parser.add_argument("--out", default=None, help="output directory (default: runs/<timestamp>-<perceiver>)")
parser.add_argument("--save-images", action="store_true")
parser.add_argument("--approach", choices=["map", "visual"], default="map",
                    help="movetowards: drive to the map's approach pose, or let the VLM pick each goal")
parser.add_argument("--near-source", choices=["vlm", "map"], default="vlm",
                    help="visual approach: stop when the VLM says the target is close, or by map distance")
parser.add_argument("--headless", action="store_true")
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": args.headless})

from puppy_agent.contracts import SUCCEEDED, StepAssignment  # noqa: E402
from puppy_agent.executor import ExecutorConfig, StepExecutor  # noqa: E402
from puppy_agent.navigation import SimpleNavigator  # noqa: E402
from puppy_agent.perception import GroundTruthPerceiver, VlmPerceiver  # noqa: E402
from puppy_agent.semantic_map import SemanticMap  # noqa: E402
from sim.puppy_sim import CAMERA_HFOV, PuppySim  # noqa: E402


def main():
    with open(args.world, encoding="utf-8") as f:
        world = json.load(f)
    with open(args.mission, encoding="utf-8") as f:
        mission = json.load(f)
    robot_id = world["robot"]["id"]
    out = args.out or os.path.join(HERE, "runs", datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + args.perceiver
                                   + ("-visual" if args.approach == "visual" else ""))
    os.makedirs(out, exist_ok=True)

    sim = PuppySim(simulation_app, world, args.usd)
    if args.perceiver == "vlm":
        perceiver = VlmPerceiver(args.vlm_url)
        server = perceiver.health()
        key = f"{server['model']}{'-int8' if server.get('quantize') else ''}__split{int(server.get('image_splitting', 0))}"
        calibrated = {}
        if os.path.exists(args.vlm_thresholds):
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
            label = getattr(answer, "label", None) or ("yes" if answer.visible else "no")
            Image.fromarray(image).save(os.path.join(out, f"{image_count[0]:03d}_{step_id}_{entity}_{label}.png"))
            image_count[0] += 1

    events = open(os.path.join(out, "events.jsonl"), "w", encoding="utf-8")

    def log_event(event):
        event = {"sim_t": round(sim.now(), 2), **event}
        events.write(json.dumps(event) + "\n")
        events.flush()
        if event["event"] == "answer":
            answer = event.get("direction", event.get("visible", event.get("near")))
            text = f"ask {event['question']:7s} \"{event['phrase']}\" -> {answer}  [{event['detail']}] ({event['latency_s']} s)"
        elif event["event"] == "near_check":
            text = f"near?   map distance {event['distance_m']} m -> {event['near']}"
        else:
            text = f"{event['event']:8s} goal {event['goal']}" + (f" (VLM said {event['from']})" if "from" in event else "")
        print(f"    t={event['sim_t']:7.2f} {event['step_id']} {text}", flush=True)

    config = ExecutorConfig(approach=args.approach, near_source=args.near_source, camera_hfov_rad=CAMERA_HFOV,
                            max_queries=60 if args.approach == "visual" else 12)
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
                  "perception_queries": executor.trace.queries}
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
