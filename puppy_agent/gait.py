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
# The rear hips are mounted 5.3 mm higher on the body than the front ones (joint1 localPos0 z in
# physics.usda), so equal leg heights pitch the nose up ~2 deg. Rear feet reach that much lower.
HEIGHT_OFFSET = {"lf": 0.0, "rf": 0.0, "lb": 0.0053, "rb": 0.0053}

MAX_STRIDE = 0.045  # m of foot travel per stance
PERIOD = 0.4  # s per gait cycle
DUTY = 0.65  # fraction of the cycle each foot is on the ground (> 0.5 overlaps the diagonal pairs)
LIFT = 0.015  # swing foot height (m); higher lifts rock the body side to side
STRIDE_RATE = 2 * MAX_STRIDE / PERIOD  # max change of a side's stride (m/s): no jumps when the command changes
# Calibration, measured headless in Isaac Sim 6.1 (2026-11-15). A stance foot sweeps its stride in DUTY * PERIOD s;
# the body covers SLIP of that (feet scuff), and skid-steer turning needs each side to move as if the
# feet were TURN_RADIUS from the centre (much wider than the 4.2 cm half track, since feet slip sideways).
SLIP = 0.97
TURN_RADIUS = 0.16  # m
# Attitude feedback: hip positions (x forward, y left; m) used to level the body from IMU roll/pitch.
HIP_XY = {"lf": (0.078, 0.042), "rf": (0.078, -0.042), "lb": (-0.078, 0.042), "rb": (-0.078, -0.042)}
LEVEL_GAIN = 1.0  # fraction of the measured tilt cancelled by leg height each step (0 = open loop)
MAX_LEVEL = 0.015  # m, cap on the height correction of one leg

MAX_VX = SLIP * MAX_STRIDE / (DUTY * PERIOD)
MAX_WZ = MAX_VX / TURN_RADIUS


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
    return {leg: (0.0, -height - HEIGHT_OFFSET[leg]) for leg in LEGS}


def side_strides(vx, wz, stance_time=DUTY * PERIOD):
    """Per-side stride lengths (left, right) for a body velocity command, clamped to the gait's range.

    When one side would exceed MAX_STRIDE both sides are scaled down together, so a saturated command
    keeps its turning radius instead of losing its yaw component.
    """
    base = vx * stance_time / SLIP
    diff = wz * TURN_RADIUS * stance_time / SLIP
    left, right = base - diff, base + diff
    peak = max(abs(left), abs(right))
    if peak > MAX_STRIDE:
        left, right = left * MAX_STRIDE / peak, right * MAX_STRIDE / peak
    return left, right


def _min_jerk(u):
    return u * u * u * (10 - 15 * u + 6 * u * u)


class TrotGait:
    """Diagonal trot (lf+rb, rf+lb) whose stride follows the latest velocity command.

    A positive stride pushes that side's feet backward during stance, which drives the body forward;
    equal strides walk straight, opposite strides spin in place.

    Foot positions are integrated rather than read off the phase, so they never jump: a stance foot
    moves at the speed its (rate-limited) stride asks for, and a swing foot travels from where it
    lifted off to the next touchdown point along a minimum-jerk path, lifting and landing with zero
    vertical speed. A zero command ramps the stride down and keeps stepping until every foot is back
    under its hip, then stands still.
    """

    PHASE_OFFSET = {"lf": 0.0, "rb": 0.0, "rf": 0.5, "lb": 0.5}

    def __init__(self, deadband=1e-3, period=PERIOD, duty=DUTY, lift=LIFT, height=STAND_HEIGHT,
                 stride_rate=STRIDE_RATE, level_gain=LEVEL_GAIN):
        self.deadband = deadband
        self.period, self.duty, self.lift, self.height = period, duty, lift, height
        self.stride_rate = stride_rate
        self.level_gain = level_gain
        self.t = 0.0
        self.walking = False
        self.stride = {"l": 0.0, "r": 0.0}
        self.y = {leg: 0.0 for leg in LEGS}
        self._liftoff_y = {leg: 0.0 for leg in LEGS}
        self._swinging = {leg: False for leg in LEGS}

    def _phase(self, leg):
        return (self.t + self.PHASE_OFFSET[leg]) % 1.0

    def _ground(self, leg):
        return -self.height - HEIGHT_OFFSET[leg]

    def update(self, dt, vx, wz, roll=0.0, pitch=0.0):
        """Advance the gait by dt seconds and return {leg: (y, z)} foot targets.

        roll/pitch (rad, from the IMU; +roll lifts the left side, +pitch lowers the nose) level the body
        when level_gain > 0: legs on the high side shorten and legs on the low side extend.
        """
        feet = self._feet(dt, vx, wz)
        if self.level_gain:
            for leg, (y, z) in feet.items():
                hx, hy = HIP_XY[leg]
                dz = self.level_gain * (math.sin(roll) * hy - math.sin(pitch) * hx)
                feet[leg] = (y, z + max(-MAX_LEVEL, min(MAX_LEVEL, dz)))
        return feet

    def _feet(self, dt, vx, wz):
        commanded = abs(vx) >= self.deadband or abs(wz) >= self.deadband
        target = dict(zip("lr", side_strides(vx, wz, self.duty * self.period) if commanded else (0.0, 0.0)))
        step = self.stride_rate * dt
        for side in "lr":
            self.stride[side] += max(-step, min(step, target[side] - self.stride[side]))

        if not self.walking:
            if not commanded:
                return stand_feet(self.height)
            self.walking = True

        self.t += dt / self.period
        stance_speed = 1.0 / (self.duty * self.period)  # stride lengths per second during stance
        stopping = not commanded and max(abs(s) for s in self.stride.values()) < 1e-4
        feet = {}
        for leg in LEGS:
            stride = self.stride[leg[0]]
            p = self._phase(leg)
            # While stopping, a foot already back under its hip stays planted instead of stepping again.
            planted = stopping and not self._swinging[leg] and abs(self.y[leg]) < 1e-3
            if p < self.duty or planted:  # stance: foot on the ground, sweeping backward
                self._swinging[leg] = False
                self.y[leg] += stride * stance_speed * dt
                z = self._ground(leg)
            else:  # swing: lift the foot and carry it to the next touchdown point
                if not self._swinging[leg]:
                    self._swinging[leg] = True
                    self._liftoff_y[leg] = self.y[leg]
                u = (p - self.duty) / (1.0 - self.duty)
                self.y[leg] = self._liftoff_y[leg] + (-0.5 * stride - self._liftoff_y[leg]) * _min_jerk(u)
                z = self._ground(leg) + self.lift * 0.5 * (1 - math.cos(2 * math.pi * u))
            limit = 0.75 * MAX_STRIDE
            self.y[leg] = max(-limit, min(limit, self.y[leg]))
            feet[leg] = (self.y[leg], z)

        if stopping and all(not self._swinging[leg] and abs(self.y[leg]) < 1e-3 for leg in LEGS):
            self.walking = False
            return stand_feet(self.height)
        return feet
