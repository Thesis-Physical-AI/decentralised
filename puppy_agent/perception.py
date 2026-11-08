"""Perception interface: narrow visual questions answered from the robot's camera image.

The executor never asks a model to plan or to choose an action. It asks closed questions
("is <thing> visible?") whose answers decide the step outcome. Any perceiver can answer them:
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


class Perceiver(Protocol):
    def is_visible(self, image, phrase: str, entity: str) -> Detection: ...


class VlmPerceiver:
    """Asks the VLM server whether `phrase` (e.g. "a red apple") appears in the image."""

    def __init__(self, url="http://127.0.0.1:8765", threshold=0.5, timeout_s=30.0):
        self.url, self.threshold, self.timeout_s = url.rstrip("/"), threshold, timeout_s

    def is_visible(self, image, phrase, entity):
        from PIL import Image  # only needed when a VLM is used

        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="PNG")
        body = json.dumps({"image_png_b64": base64.b64encode(buffer.getvalue()).decode(), "phrase": phrase})
        request = urllib.request.Request(self.url + "/is_visible", data=body.encode(),
                                         headers={"Content-Type": "application/json"})
        start = time.perf_counter()
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            reply = json.loads(response.read())
        p_yes = float(reply["p_yes"])
        visible = p_yes >= self.threshold
        return Detection(visible, p_yes if visible else 1 - p_yes, "vlm", time.perf_counter() - start,
                         f"p_yes={p_yes:.3f} model={reply.get('model', '?')}")


class GroundTruthPerceiver:
    """Visible if the entity's centre is inside the camera's horizontal field of view and range.

    Ignores occlusion and lighting. `locate(entity)` returns (x, y) in the map frame or None;
    `camera_pose()` returns the camera's Pose2D. Used as the baseline the VLM is compared against.
    """

    def __init__(self, locate: Callable[[str], Optional[tuple]], camera_pose, hfov_rad=math.radians(70),
                 max_range_m=4.0):
        self.locate, self.camera_pose = locate, camera_pose
        self.half_fov, self.max_range = hfov_rad / 2, max_range_m

    def is_visible(self, image, phrase, entity):
        target = self.locate(entity)
        if target is None:
            return Detection(False, 1.0, "perception", detail="entity not in the world")
        pose = self.camera_pose()
        dx, dy = target[0] - pose.x, target[1] - pose.y
        bearing = (math.atan2(dy, dx) - pose.yaw + math.pi) % (2 * math.pi) - math.pi
        distance = math.hypot(dx, dy)
        visible = abs(bearing) <= self.half_fov and distance <= self.max_range
        return Detection(visible, 1.0, "perception",
                         detail=f"bearing={math.degrees(bearing):.0f}deg distance={distance:.2f}m")
