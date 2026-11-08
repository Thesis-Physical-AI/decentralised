"""Test doubles: a kinematic robot that follows cmd_vel exactly, and scripted perceivers."""
import math

import numpy as np

from puppy_agent.navigation import Pose2D, wrap
from puppy_agent.perception import Detection


class KinematicRobot:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, dt=0.01):
        self.pose = Pose2D(x, y, yaw)
        self.vx = self.wz = 0.0
        self.t = 0.0
        self.dt = dt
        self.fallen = False

    def set_cmd_vel(self, vx, wz):
        self.vx, self.wz = vx, wz

    def odom(self):
        return Pose2D(self.pose.x, self.pose.y, self.pose.yaw)

    def now(self):
        return self.t

    def advance(self, seconds):
        for _ in range(max(1, round(seconds / self.dt))):
            self.pose.x += self.vx * math.cos(self.pose.yaw) * self.dt
            self.pose.y += self.vx * math.sin(self.pose.yaw) * self.dt
            self.pose.yaw = wrap(self.pose.yaw + self.wz * self.dt)
            self.t += self.dt

    def camera(self):
        return np.zeros((4, 4, 3), dtype=np.uint8)

    def safety(self):
        return "fell over" if self.fallen else ""


class StuckRobot(KinematicRobot):
    """Ignores velocity commands, like a robot pressed against a wall."""

    def set_cmd_vel(self, vx, wz):
        self.vx = self.wz = 0.0


class ScriptedPerceiver:
    """Answers from a list (True/False per query); answers False once the list runs out."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def is_visible(self, image, phrase, entity):
        self.calls.append(entity)
        visible = self.answers.pop(0) if self.answers else False
        return Detection(visible, 0.9, "vlm")
