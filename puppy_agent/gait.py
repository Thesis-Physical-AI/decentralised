"""PuppyPi leg kinematics and a velocity-driven trot gait.

The gait turns a body velocity command (vx forward m/s, wz yaw rad/s), the same quantity Nav2's
controller publishes on /cmd_vel, into a foot position for each leg. The simulator converts foot
positions to joint targets with leg_ik().
"""
import math

LEGS = ("lf", "rf", "lb", "rb")
THIGH = 0.0745  # hip (joint1) -> knee (joint2), from payloads/Physics/physics.usda
SHIN = 0.075  # knee (joint2) -> foot (joint3)
STAND_HEIGHT = 0.10  # foot depth below the hip when standing
LOW_HEIGHT = 0.05  # foot depth below the hip when crouched

# Measured in Isaac Sim 6.1 with hello_puppy.py (2026-11-08): a 0.04 m stride at a 0.5 s period walks
# 0.45 m in 3 s, and a +-0.03 m differential stride turns ~112 deg in 3 s. Hence:
STRIDE_PER_MPS = 1 / 3.75  # stride (m) per forward speed (m/s)
STRIDE_PER_RADPS = 0.03 / 0.65  # per-side stride difference (m) per yaw rate (rad/s)
MAX_STRIDE = 0.045
PERIOD = 0.5  # s per gait cycle
LIFT = 0.025  # swing foot height (m)

MAX_VX = MAX_STRIDE / STRIDE_PER_MPS  # ~0.17 m/s
MAX_WZ = MAX_STRIDE / STRIDE_PER_RADPS  # ~0.98 rad/s


def leg_ik(y, z):
    """Hip and knee angles (rad) that put the foot at (y, z) m from the hip; +y is backward, +z is up.

    At zero the thigh hangs straight down and the shin points forward, and a positive angle on either
    joint swings the foot backward. The knee bends backward, like the default zero pose.
    """
    d = min(max(math.hypot(y, z), 0.04), THIGH + SHIN - 1e-3)
    knee_bend = -math.acos((d * d - THIGH * THIGH - SHIN * SHIN) / (2 * THIGH * SHIN))
    hip = math.atan2(y, -z) - math.atan2(SHIN * math.sin(knee_bend), THIGH + SHIN * math.cos(knee_bend))
    return hip, knee_bend + math.pi / 2


def leg_fk(hip, knee):
    """Inverse of leg_ik: foot (y, z) from hip and knee joint angles."""
    bend = knee - math.pi / 2
    y = THIGH * math.sin(hip) + SHIN * math.sin(hip + bend)
    z = -(THIGH * math.cos(hip) + SHIN * math.cos(hip + bend))
    return y, z


def stand_feet(height=STAND_HEIGHT):
    return {leg: (0.0, -height) for leg in LEGS}


def side_strides(vx, wz):
    """Per-side stride lengths (left, right) for a body velocity command, clamped to the gait's range."""
    base = vx * STRIDE_PER_MPS
    diff = wz * STRIDE_PER_RADPS
    clamp = lambda s: max(-MAX_STRIDE, min(MAX_STRIDE, s))  # noqa: E731
    return clamp(base - diff), clamp(base + diff)


class TrotGait:
    """Diagonal trot (lf+rb, rf+lb) whose stride follows the latest velocity command.

    A positive stride pushes that side's feet backward during stance, which drives the body forward;
    equal strides walk straight, opposite strides spin in place. A zero command returns to standing.
    """

    PHASE_OFFSET = {"lf": 0.0, "rb": 0.0, "rf": 0.5, "lb": 0.5}

    def __init__(self, deadband=1e-3):
        self.deadband = deadband
        self.t = 0.0

    def update(self, dt, vx, wz):
        """Advance the gait by dt seconds and return {leg: (y, z)} foot targets."""
        if abs(vx) < self.deadband and abs(wz) < self.deadband:
            self.t = 0.0
            return stand_feet()
        left, right = side_strides(vx, wz)
        self.t += dt / PERIOD
        feet = {}
        for leg in LEGS:
            stride = left if leg.startswith("l") else right
            p = (self.t + self.PHASE_OFFSET[leg]) % 1.0
            if p < 0.5:  # swing: lift the foot and bring it forward
                y = 0.5 * stride * math.cos(2 * math.pi * p)
                z = -STAND_HEIGHT + LIFT * math.sin(2 * math.pi * p)
            else:  # stance: foot on the ground, sweeping backward
                y = -0.5 * stride + stride * (p - 0.5) / 0.5
                z = -STAND_HEIGHT
            feet[leg] = (y, z)
        return feet
