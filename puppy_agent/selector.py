"""Action Selector menu (architecture layer 7): the closed list of navigation primitives a selector may choose.

The selector (the VLM, or the rule-based baseline) looks at the camera image, reasons about it, and
names one primitive. Its output is untrusted: `validate()` accepts only a name from MENU, and
`motion_for()` turns a valid name into a bounded command for the navigator. The model never emits
velocities or poses itself, so a wrong answer costs one short, bounded move.

    forward   target ahead and far              -> goal STEP_M straight ahead
    left      target on the left of the image   -> goal along a left bearing (walk and turn)
    right     target on the right of the image  -> goal along a right bearing
    search    target not in view                -> turn in place, toward the side it was last seen
    back      something blocks the way          -> back up a little (Nav2 BackUp behaviour)
    arrived   target ahead and close            -> stop; the executor decides whether to accept it
"""
import math
from dataclasses import dataclass

from .navigation import Pose2D, wrap

MENU = ("forward", "left", "right", "search", "back", "arrived")

DESCRIPTIONS = {
    "forward": "the target is straight ahead but still far away: walk toward it",
    "left": "the target is on the left side of the image: walk forward while turning left",
    "right": "the target is on the right side of the image: walk forward while turning right",
    "search": "the target is not in the image: turn in place to look for it",
    "back": "something is blocking the way right in front of the robot: step back",
    "arrived": "the target is straight ahead and very close, filling a large part of the image: stop",
}


@dataclass
class MotionLimits:
    step_m: float = 0.4  # forward: distance of each goal
    veer_m: float = 0.3  # left/right: distance of each goal
    veer_rad: float = math.radians(30)  # left/right: bearing of each goal
    search_rad: float = math.radians(45)  # search: turn per decision
    back_m: float = 0.2  # back: distance


@dataclass
class Motion:
    kind: str  # "go_to", "spin", "back_up" or "stop"
    goal: Pose2D = None
    angle: float = 0.0
    distance: float = 0.0


def validate(action):
    """The menu entry for a selector answer, or None if it is not on the menu."""
    if not isinstance(action, str):
        return None
    action = action.strip().lower()
    return action if action in MENU else None


def last_seen_side(history):
    """'left' or 'right': the side the target was last reported on (default left)."""
    for action in reversed(history):
        if action in ("left", "right"):
            return action
    return "left"


def motion_for(action, pose, history, limits=None):
    """A bounded navigator command for a validated action."""
    limits = limits or MotionLimits()
    if action == "forward":
        return Motion("go_to", _ahead(pose, 0.0, limits.step_m))
    if action in ("left", "right"):
        bearing = limits.veer_rad if action == "left" else -limits.veer_rad
        return Motion("go_to", _ahead(pose, bearing, limits.veer_m))
    if action == "search":
        sign = 1.0 if last_seen_side(history) == "left" else -1.0
        return Motion("spin", angle=sign * limits.search_rad)
    if action == "back":
        return Motion("back_up", distance=limits.back_m)
    if action == "arrived":
        return Motion("stop")
    raise ValueError(f"not a menu action: {action!r}")


def _ahead(pose, bearing, distance):
    heading = wrap(pose.yaw + bearing)
    return Pose2D(pose.x + distance * math.cos(heading), pose.y + distance * math.sin(heading), heading)
