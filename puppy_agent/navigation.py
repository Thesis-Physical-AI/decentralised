"""Navigation interface shaped like Nav2, plus a minimal built-in implementation.

The executor only talks to `Navigator`. Its methods mirror Nav2's action interfaces:
  go_to(pose)  ~ nav2_msgs/action/NavigateToPose   (goal pose in the map frame)
  spin(angle)  ~ nav2_msgs/action/Spin             (behavior server)
  feedback()   ~ action feedback (distance_remaining)
  result()     ~ action result: None while running, else SUCCEEDED / ABORTED / CANCELED
  cancel()     ~ cancel the active goal
and the robot is driven by body velocity commands, as Nav2's controller does on /cmd_vel.

`SimpleNavigator` is a stand-in for the prototype: it drives straight to the goal (rotate, then
drive) with no map, costmap or obstacle avoidance. A `Nav2Navigator` wrapping an rclpy action
client can replace it without changing the executor (see docs/PROTOTYPE.md).
"""
import math
from dataclasses import dataclass
from typing import Optional, Protocol

from .gait import MAX_VX, MAX_WZ

SUCCEEDED, ABORTED, CANCELED = "SUCCEEDED", "ABORTED", "CANCELED"


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: Optional[float] = None  # None: any final heading


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


class Robot(Protocol):
    """What a navigator and the executor need from a robot (simulated or real)."""

    def set_cmd_vel(self, vx: float, wz: float) -> None: ...
    def odom(self) -> Pose2D: ...  # pose in the map frame (sim: ground truth)
    def advance(self, seconds: float) -> None: ...  # let time pass (sim: step physics)
    def now(self) -> float: ...  # seconds (sim time)


class Navigator(Protocol):
    def go_to(self, goal: Pose2D) -> None: ...
    def spin(self, angle: float) -> None: ...
    def spin_once(self) -> None: ...  # do one control update; with Nav2: spin the rclpy node
    def feedback(self) -> dict: ...
    def result(self) -> Optional[str]: ...
    def cancel(self) -> None: ...


class SimpleNavigator:
    """Rotate toward the goal, drive to it, then turn to the goal heading. No obstacle avoidance."""

    def __init__(self, robot, pos_tol=0.08, yaw_tol=0.12, max_vx=0.8 * MAX_VX, max_wz=0.7 * MAX_WZ,
                 stuck_window_s=8.0, stuck_progress_m=0.05):
        self.robot = robot
        self.pos_tol, self.yaw_tol = pos_tol, yaw_tol
        self.max_vx, self.max_wz = max_vx, max_wz
        self.stuck_window_s, self.stuck_progress_m = stuck_window_s, stuck_progress_m
        self._mode = None
        self._result = None

    # -- goals -------------------------------------------------------------------------------
    def go_to(self, goal):
        self._mode, self._goal, self._result = "goto", goal, None
        self._best, self._best_t = math.inf, self.robot.now()

    def spin(self, angle):
        self._mode, self._result = "spin", None
        self._spin_left = angle
        self._last_yaw = self.robot.odom().yaw

    def cancel(self):
        if self._mode:
            self._finish(CANCELED)

    def result(self):
        return self._result

    def feedback(self):
        if self._mode == "goto":
            pose = self.robot.odom()
            return {"distance_remaining": math.hypot(self._goal.x - pose.x, self._goal.y - pose.y)}
        if self._mode == "spin":
            return {"angle_remaining": self._spin_left}
        return {}

    # -- control -----------------------------------------------------------------------------
    def spin_once(self):
        if self._mode == "goto":
            self._goto_step()
        elif self._mode == "spin":
            self._spin_step()

    def _finish(self, result):
        self.robot.set_cmd_vel(0.0, 0.0)
        self._mode, self._result = None, result

    def _turn_rate(self, error):
        return max(-self.max_wz, min(self.max_wz, 2.0 * error))

    def _goto_step(self):
        pose, goal = self.robot.odom(), self._goal
        dx, dy = goal.x - pose.x, goal.y - pose.y
        dist = math.hypot(dx, dy)
        if dist < self._best - self.stuck_progress_m:
            self._best, self._best_t = dist, self.robot.now()
        elif self.robot.now() - self._best_t > self.stuck_window_s:
            self._finish(ABORTED)  # no progress: blocked or unreachable
            return
        if dist > self.pos_tol:
            heading_error = wrap(math.atan2(dy, dx) - pose.yaw)
            if abs(heading_error) > 0.35:
                self.robot.set_cmd_vel(0.0, self._turn_rate(heading_error))
            else:
                self.robot.set_cmd_vel(min(self.max_vx, 0.6 * dist + 0.04), self._turn_rate(heading_error))
            return
        if goal.yaw is not None:
            yaw_error = wrap(goal.yaw - pose.yaw)
            if abs(yaw_error) > self.yaw_tol:
                self._best_t = self.robot.now()  # turning in place counts as progress
                self.robot.set_cmd_vel(0.0, self._turn_rate(yaw_error))
                return
        self._finish(SUCCEEDED)

    def _spin_step(self):
        yaw = self.robot.odom().yaw
        self._spin_left -= wrap(yaw - self._last_yaw)
        self._last_yaw = yaw
        if abs(self._spin_left) < self.yaw_tol:
            self._finish(SUCCEEDED)
        else:
            self.robot.set_cmd_vel(0.0, self._turn_rate(self._spin_left))
