"""Turn the VLM's view answer ("the target is to the left") into a navigation goal.

The VLM decides *where* to go; the robot's navigation stack (Nav2 for ground robots, PX4 for a drone)
decides *how*. Only the adapter is robot-specific:

    bearing = perception.direction_bearing(direction.label, camera_hfov)   # same for every robot
    goal    = adapter.goal_toward(current_pose, bearing, step_m)           # per robot type
    navigator.go_to(goal)                                                  # Nav2 NavigateToPose / PX4 setpoint

A drone adapter (not implemented yet) would return a 3D setpoint holding altitude, for its PX4 navigator.
"""
import math
from typing import Protocol

from .navigation import Pose2D, wrap


class GoalAdapter(Protocol):
    def goal_toward(self, pose, bearing: float, step_m: float): ...


class GroundGoalAdapter:
    """Ground robots (PuppyPi, TurtleBot): a Pose2D step_m away along the bearing, facing that way.

    This is the goal a Nav2 NavigateToPose action takes (x, y, yaw in the map frame)."""

    def goal_toward(self, pose, bearing, step_m):
        heading = wrap(pose.yaw + bearing)
        return Pose2D(pose.x + step_m * math.cos(heading), pose.y + step_m * math.sin(heading), heading)
