"""Perception interface: narrow visual questions answered from the robot's camera image.

The executor never asks a model to plan or to choose an action. It asks closed questions whose answers
decide the step outcome or the next navigation goal:
  is_visible(image, phrase)  is <thing> in view?
  where(image, phrase)       is it to the left, center or right, or not in view?
  is_near(image, phrase)     is it close to the camera?
Any perceiver can answer them:
  - VlmPerceiver: a VLM served by vlm_server/ (the agent's VLA slot, currently SmolVLM)
  - GroundTruthPerceiver: geometry from the simulator; the rule-based baseline / oracle
"""
import base64
import io
import json
import math
import time
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional, Protocol


@dataclass
class Detection:
    visible: bool
    confidence: float  # probability that the answer "visible" is correct, 0..1
    source: str  # "vlm" or "perception"
    latency_s: float = 0.0
    detail: str = ""


@dataclass
class Direction:
    label: str  # "left", "center", "right" or "none" (not in view)
    confidence: float  # score of the chosen label, 0..1
    source: str
    latency_s: float = 0.0
    scores: dict = None
    detail: str = ""


DIRECTIONS = ("left", "center", "right")
# Image-x of the centre of each crop the VLM server scores (vlm_server/smolvlm.py CROPS), as a fraction of width.
DIRECTION_X = {"left": 0.225, "center": 0.5, "right": 0.775}


def direction_bearing(label, hfov_rad):
    """Bearing (rad, positive = left) of a direction label for a pinhole camera with this horizontal FOV."""
    return math.atan((0.5 - DIRECTION_X[label]) * 2 * math.tan(hfov_rad / 2))


class Perceiver(Protocol):
    def is_visible(self, image, phrase: str, entity: str) -> Detection: ...
    def where(self, image, phrase: str, entity: str) -> Direction: ...
    def is_near(self, image, phrase: str, entity: str) -> Detection: ...


class VlmPerceiver:
    """Asks the robot's VLM server (vlm_server/smolvlm_server.py) closed questions about the camera image.

    latency_s of each answer is the server's onboard_latency_s when present: the time the answer would
    take on the emulated onboard computer, which the executor charges to simulation time.
    """

    def __init__(self, url="http://127.0.0.1:8765", threshold=0.5, where_threshold=None, near_threshold=None,
                 timeout_s=120.0):
        """threshold: p_yes for "visible"; where_threshold: best crop score for "in view" (default: threshold);
        near_threshold: p_yes for "near" (default: threshold). Small models need calibrated values."""
        self.url, self.threshold, self.timeout_s = url.rstrip("/"), threshold, timeout_s
        self.where_threshold = threshold if where_threshold is None else where_threshold
        self.near_threshold = threshold if near_threshold is None else near_threshold

    def health(self):
        with urllib.request.urlopen(self.url + "/health", timeout=10) as response:
            return json.loads(response.read())

    def _ask(self, endpoint, image, phrase):
        from PIL import Image  # only needed when a VLM is used

        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="PNG")
        body = json.dumps({"image_png_b64": base64.b64encode(buffer.getvalue()).decode(), "phrase": phrase})
        request = urllib.request.Request(self.url + endpoint, data=body.encode(),
                                         headers={"Content-Type": "application/json"})
        start = time.perf_counter()
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            reply = json.loads(response.read())
        reply["_latency"] = float(reply.get("onboard_latency_s", time.perf_counter() - start))
        return reply

    def _yes_no(self, endpoint, image, phrase, threshold):
        reply = self._ask(endpoint, image, phrase)
        p_yes = float(reply["p_yes"])
        yes = p_yes >= threshold
        return Detection(yes, p_yes if yes else 1 - p_yes, "vlm", reply["_latency"],
                         f"p_yes={p_yes:.3f} model={reply.get('model', '?')} profile={reply.get('profile', '?')}")

    def is_visible(self, image, phrase, entity):
        return self._yes_no("/is_visible", image, phrase, self.threshold)

    def is_near(self, image, phrase, entity):
        return self._yes_no("/is_near", image, phrase, self.near_threshold)

    def where(self, image, phrase, entity):
        reply = self._ask("/where", image, phrase)
        scores = {d: float(reply["scores"][d]) for d in DIRECTIONS}
        best = max(scores, key=scores.get)
        label = best if scores[best] >= self.where_threshold else "none"
        confidence = scores[best] if label != "none" else 1 - scores[best]
        detail = " ".join(f"{d}={scores[d]:.3f}" for d in DIRECTIONS)
        return Direction(label, confidence, "vlm", reply["_latency"], scores,
                         f"{detail} model={reply.get('model', '?')} profile={reply.get('profile', '?')}")


class GroundTruthPerceiver:
    """Visible if the entity's centre is inside the camera's horizontal field of view and range.

    Ignores occlusion and lighting. `locate(entity)` returns (x, y) in the map frame or None;
    `camera_pose()` returns the camera's Pose2D. Used as the baseline the VLM is compared against.
    """

    def __init__(self, locate: Callable[[str], Optional[tuple]], camera_pose, hfov_rad=math.radians(70),
                 max_range_m=4.0, near_m=0.55):
        self.locate, self.camera_pose = locate, camera_pose
        self.hfov, self.half_fov, self.max_range, self.near_m = hfov_rad, hfov_rad / 2, max_range_m, near_m

    def _geometry(self, entity):
        target = self.locate(entity)
        if target is None:
            return None
        pose = self.camera_pose()
        dx, dy = target[0] - pose.x, target[1] - pose.y
        bearing = (math.atan2(dy, dx) - pose.yaw + math.pi) % (2 * math.pi) - math.pi
        distance = math.hypot(dx, dy)
        return bearing, distance, abs(bearing) <= self.half_fov and distance <= self.max_range

    def is_visible(self, image, phrase, entity):
        geometry = self._geometry(entity)
        if geometry is None:
            return Detection(False, 1.0, "perception", detail="entity not in the world")
        bearing, distance, visible = geometry
        return Detection(visible, 1.0, "perception",
                         detail=f"bearing={math.degrees(bearing):.0f}deg distance={distance:.2f}m")

    def is_near(self, image, phrase, entity):
        geometry = self._geometry(entity)
        near = geometry is not None and geometry[2] and geometry[1] <= self.near_m
        return Detection(near, 1.0, "perception",
                         detail="entity not in the world" if geometry is None else f"distance={geometry[1]:.2f}m")

    def where(self, image, phrase, entity):
        geometry = self._geometry(entity)
        if geometry is None or not geometry[2]:
            return Direction("none", 1.0, "perception", detail="not in view")
        bearing = geometry[0]
        x = 0.5 - math.tan(bearing) / (2 * math.tan(self.half_fov))  # image-x of the entity, 0 = left edge
        label = "left" if x < 1 / 3 else "right" if x > 2 / 3 else "center"
        return Direction(label, 1.0, "perception", detail=f"bearing={math.degrees(bearing):.0f}deg x={x:.2f}")
