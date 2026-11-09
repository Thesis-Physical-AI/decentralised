"""Capture a labelled image set from the PuppyPi's front camera for VLM benchmarking.

    $ISAACSIM_PATH/python.sh sim/capture_dataset.py --headless --out bench/data/one_room_eval --count 80

The robot is teleported to random standing poses in the room (about half aimed near an object), and
each frame is saved with exact ground truth from Isaac Sim's bounding_box_2d_tight annotator: the
pixel box of every labelled object that has at least one visible pixel (occlusion included).
Writes <out>/images/NNN.png and <out>/labels.jsonl. Labels are never shown to the model.
"""
import argparse
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
parser = argparse.ArgumentParser()
parser.add_argument("--world", default=os.path.join(HERE, "worlds", "one_room_eval.json"))
parser.add_argument("--usd", default=os.path.join(ROOT, "puppy_isaac", "puppy_isaac.usda"))
parser.add_argument("--out", default=os.path.join(ROOT, "bench", "data", "one_room_eval"))
parser.add_argument("--count", type=int, default=80)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--headless", action="store_true")
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": args.headless})

from PIL import Image  # noqa: E402

from sim.puppy_sim import CAMERA_HFOV, PuppySim  # noqa: E402


def footprints(world, margin=0.22):
    """Axis-aligned keep-out rectangles around boxes in the scene."""
    rects = []
    for spec in world["entities"].values():
        shape = spec.get("shape")
        if shape and shape["type"] == "box" and spec.get("present_in_sim", True):
            (x, y, _), (sx, sy, _) = spec["position"], shape["size"]
            rects.append((x - sx / 2 - margin, x + sx / 2 + margin, y - sy / 2 - margin, y + sy / 2 + margin))
    return rects


def sample_pose(rng, world, rects, targets):
    half = world["room"]["size"][0] / 2 - 0.25
    while True:
        x, y = rng.uniform(-half, half), rng.uniform(-half, half)
        if any(x0 <= x <= x1 and y0 <= y <= y1 for x0, x1, y0, y1 in rects):
            continue
        if rng.random() < 0.55:  # aim near an object, offset so it lands left, centre or right
            tx, ty = targets[rng.choice(sorted(targets))]
            yaw = math.atan2(ty - y, tx - x) + rng.uniform(-0.6, 0.6) * CAMERA_HFOV
        else:
            yaw = rng.uniform(-math.pi, math.pi)
        return x, y, yaw


def main():
    with open(args.world, encoding="utf-8") as f:
        world = json.load(f)
    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    sim = PuppySim(simulation_app, world, args.usd)
    sim.enable_ground_truth()
    rng = random.Random(args.seed)
    rects = footprints(world)
    targets = {n: tuple(s["position"][:2]) for n, s in world["entities"].items()
               if s.get("shape") and s.get("present_in_sim", True)}
    descriptions = {n: world["entities"][n]["description"] for n in targets}
    with open(os.path.join(args.out, "labels.jsonl"), "w", encoding="utf-8") as labels:
        for i in range(args.count):
            x, y, yaw = sample_pose(rng, world, rects, targets)
            sim.teleport(x, y, yaw)
            image = sim.capture()
            boxes = sim.visible_boxes()
            pose = sim.odom()
            if sim.safety():  # robot fell after teleport: skip this pose
                continue
            name = f"{i:03d}.png"
            Image.fromarray(image).save(os.path.join(args.out, "images", name))
            labels.write(json.dumps({"image": name, "pose": [round(pose.x, 3), round(pose.y, 3), round(pose.yaw, 3)],
                                     "width": image.shape[1], "height": image.shape[0],
                                     "boxes": boxes, "descriptions": descriptions}) + "\n")
            print(f"{name} pose=({pose.x:.2f},{pose.y:.2f},{math.degrees(pose.yaw):.0f}deg) visible={sorted(boxes)}",
                  flush=True)
    sim.close()


main()
simulation_app.close()
