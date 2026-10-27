"""PuppyPi movements in Isaac Sim 6.x.

Run:  ~/Projects/isaacsim/_build/linux-x86_64/release/python.sh hello_puppy.py [--usd PATH] [--headless]
                                                                                [--moves MOVE ...] [--check]

Every movement the robot can make is its own function (see MOVES at the bottom). main() runs them in
order: by default the sequence in DEFAULT_SEQUENCE, or whatever is passed with --moves, e.g.
    hello_puppy.py --moves stand walk_forward turn_left sit --keep-open

--check additionally runs the original stage / gravity / joint sanity checks first.
"""

import argparse
import math
import os

MOVE_NAMES = [
    "stand", "sit", "lie_down", "bow", "push_up", "nod", "wave",
    "march", "walk_forward", "walk_backward", "turn_left", "turn_right",
]
DEFAULT_SEQUENCE = ["stand", "walk_forward", "turn_left", "walk_backward", "turn_right", "sit", "stand", "wave", "lie_down"]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--usd",
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "puppy_isaac", "puppy_isaac.usda"),
)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--keep-open", action="store_true", help="keep the window open after the movements")
parser.add_argument("--moves", nargs="+", choices=MOVE_NAMES, help="movements to run, in order")
parser.add_argument("--check", action="store_true", help="run the gravity and joint checks first")
args = parser.parse_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless})

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
from isaacsim.core.experimental.objects import DistantLight, GroundPlane
from isaacsim.core.experimental.prims import Articulation, XformPrim
from isaacsim.core.rendering_manager import RenderingManager
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.utils.viewports import set_camera_view
from pxr import Usd, UsdPhysics

ROBOT_PATH = "/World/Puppy"
SPAWN_Z = 0.25
PHYSICS_DT = 1.0 / 120.0

# Leg geometry, measured from the joint offsets in payloads/Physics/physics.usda.
LEGS = ("lf", "rf", "lb", "rb")
THIGH = 0.0745  # hip (joint1) -> knee (joint2)
SHIN = 0.075  # knee (joint2) -> foot (joint3)
STAND_HEIGHT = 0.10  # foot depth below the hip when standing
LOW_HEIGHT = 0.05  # foot depth below the hip when crouched


def step(n):
    for _ in range(n):
        SimulationManager.step()
        if not args.headless:
            RenderingManager.render()
        simulation_app.update()


def base_z(robot):
    positions, _ = robot.get_world_poses()
    return float(positions.numpy()[0, 2])


# ---------------------------------------------------------------- scene
def build_scene():
    stage_utils.create_new_stage()
    stage_utils.set_stage_units(meters_per_unit=1.0)
    GroundPlane("/World/GroundPlane", positions=[0, 0, 0])
    DistantLight("/World/DistantLight").set_intensities(300)
    set_camera_view(eye=[0.6, 0.6, 0.4], target=[0.0, 0.0, 0.1], camera_prim_path="/OmniverseKit_Persp")

    stage_utils.add_reference_to_stage(usd_path=args.usd, path=ROBOT_PATH)
    # The converter puts all physics behind a "Physics" variant; with nothing selected the robot is geometry only.
    physics_variant = stage_utils.get_current_stage().GetPrimAtPath(ROBOT_PATH).GetVariantSets().GetVariantSet("Physics")
    if physics_variant:
        physics_variant.SetVariantSelection("physx")
    XformPrim(ROBOT_PATH, reset_xform_op_properties=True).set_world_poses(positions=[0.0, 0.0, SPAWN_Z])


def check_stage():
    """Lists articulation roots and deactivates any joint pinning the robot to the world."""
    stage = stage_utils.get_current_stage()
    roots, pinned = [], []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(ROBOT_PATH)):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            roots.append(str(prim.GetPath()))
        joint = UsdPhysics.Joint(prim)
        if joint:
            targets = joint.GetBody0Rel().GetTargets()
            body0 = stage.GetPrimAtPath(targets[0]) if targets else None
            # A joint whose body0 is missing or not a rigid body is attached to the world.
            if body0 is None or not body0.HasAPI(UsdPhysics.RigidBodyAPI):
                pinned.append(str(prim.GetPath()))

    print(f"\n[stage] articulation roots: {roots}")
    if len(roots) != 1:
        print("    WARNING: expected exactly one articulation root")
    for path in pinned:
        print(f"    WARNING: {path} attaches the robot to the world -> deactivating it")
        stage.GetPrimAtPath(path).SetActive(False)
    if not pinned:
        print("    no world-pinning joints found")
    return len(roots) == 1


def check_gravity(robot):
    """Lets the robot settle for 2 s and confirms it fell and landed on the ground."""
    z0 = base_z(robot)
    step(240)
    z1 = base_z(robot)
    fell = z1 < z0 - 0.02
    above_ground = z1 > 0.0
    print(f"\n[gravity] base z: start {z0:.3f} m -> after 2 s {z1:.3f} m")
    print(f"    falls under gravity: {'PASS' if fell else 'FAIL'}")
    print(f"    stays above ground:  {'PASS' if above_ground else 'FAIL (falling through the floor: check colliders)'}")
    return fell and above_ground


def check_joints(robot):
    """Drives every revolute joint with a small sine wave and confirms they follow."""
    amp = 0.3
    errors = []
    for i in range(240):
        target = amp * math.sin(2 * math.pi * i / 120.0)
        robot.set_dof_position_targets(np.full(robot.num_dofs, target))
        step(1)
        errors.append(np.abs(robot.get_dof_positions().numpy()[0] - target).max())
    moved = float(np.max(np.abs(robot.get_dof_positions().numpy()))) > 1e-3 or max(errors) < amp
    print(f"\n[joints] max joint tracking error: {max(errors):.3f} rad (target amplitude {amp} rad)")
    print(f"    joints respond: {'PASS' if moved else 'FAIL'}")
    return moved


# ---------------------------------------------------------------- leg control
def leg_ik(y, z):
    """Hip and knee angles (rad) that put the foot at (y, z) m from the hip; +y is backward, +z is up.

    At zero the thigh hangs straight down and the shin points forward, and a positive angle on either
    joint swings the foot backward. The knee bends backward, like the default zero pose.
    """
    d = min(max(math.hypot(y, z), 0.04), THIGH + SHIN - 1e-3)
    knee_bend = -math.acos((d * d - THIGH * THIGH - SHIN * SHIN) / (2 * THIGH * SHIN))
    hip = math.atan2(y, -z) - math.atan2(SHIN * math.sin(knee_bend), THIGH + SHIN * math.cos(knee_bend))
    return hip, knee_bend + math.pi / 2


class Puppy:
    """Joint-target interface for the PuppyPi: movements describe where each foot goes, this does the rest."""

    def __init__(self, robot):
        self.robot = robot
        names = list(robot.dof_names)
        self.hip_idx = {leg: names.index(f"{leg}_joint1") for leg in LEGS}
        self.knee_idx = {leg: names.index(f"{leg}_joint2") for leg in LEGS}
        self.targets = np.zeros(robot.num_dofs)

    def feet_to_targets(self, feet):
        """feet: {leg: (y, z)} -> full joint-target array. Legs not listed keep their current target."""
        targets = self.targets.copy()
        for leg, (y, z) in feet.items():
            targets[self.hip_idx[leg]], targets[self.knee_idx[leg]] = leg_ik(y, z)
        return targets

    def set_feet(self, feet):
        """Sends foot positions as joint targets immediately (no interpolation)."""
        self.targets = self.feet_to_targets(feet)
        self.robot.set_dof_position_targets(self.targets)

    def move_feet(self, feet, seconds=0.5, hold=0.2):
        """Moves the feet smoothly to the given positions over `seconds`, then holds for `hold` seconds."""
        start, end = self.targets.copy(), self.feet_to_targets(feet)
        n = max(1, int(seconds / PHYSICS_DT))
        for i in range(1, n + 1):
            s = 0.5 - 0.5 * math.cos(math.pi * i / n)  # ease in/out
            self.targets = start + s * (end - start)
            self.robot.set_dof_position_targets(self.targets)
            step(1)
        step(int(hold / PHYSICS_DT))

    def pose(self, front=STAND_HEIGHT, back=STAND_HEIGHT, front_y=0.0, back_y=0.0, seconds=0.5, hold=0.2):
        """Moves front and back feet to the given depths below the hips and fore/aft offsets."""
        feet = {}
        for leg in LEGS:
            is_front = leg.endswith("f")
            feet[leg] = (front_y if is_front else back_y, -(front if is_front else back))
        self.move_feet(feet, seconds, hold)

    def gait(self, left_stride, right_stride, cycles=4, period=0.5, lift=0.025):
        """Trot: diagonal leg pairs (lf+rb, rf+lb) alternate swing and stance.

        A positive stride pushes that side's feet backward during stance, which drives the body forward.
        Equal strides walk straight; opposite strides spin in place.
        """
        self.pose(seconds=0.3, hold=0.0)
        phase_offset = {"lf": 0.0, "rb": 0.0, "rf": 0.5, "lb": 0.5}
        n = int(cycles * period / PHYSICS_DT)
        for i in range(n):
            t = i * PHYSICS_DT / period
            feet = {}
            for leg in LEGS:
                stride = left_stride if leg.startswith("l") else right_stride
                p = (t + phase_offset[leg]) % 1.0
                if p < 0.5:  # swing: lift the foot and bring it forward
                    y = 0.5 * stride * math.cos(2 * math.pi * p)
                    z = -STAND_HEIGHT + lift * math.sin(2 * math.pi * p)
                else:  # stance: foot on the ground, sweeping backward
                    y = -0.5 * stride + stride * (p - 0.5) / 0.5
                    z = -STAND_HEIGHT
                feet[leg] = (y, z)
            self.set_feet(feet)
            step(1)
        self.pose(seconds=0.3)


# ---------------------------------------------------------------- movements
def stand(puppy):
    """Stands up straight on all four legs."""
    puppy.pose()


def sit(puppy):
    """Sits: back legs folded, front legs straight."""
    puppy.pose(front=STAND_HEIGHT + 0.01, back=LOW_HEIGHT, front_y=-0.01, back_y=0.02, seconds=0.8, hold=1.0)


def lie_down(puppy):
    """Lies down with all four legs folded."""
    puppy.pose(front=LOW_HEIGHT, back=LOW_HEIGHT, seconds=1.0, hold=1.0)


def bow(puppy):
    """Play bow / stretch: front low and forward, back high."""
    puppy.pose(front=LOW_HEIGHT, back=STAND_HEIGHT + 0.01, front_y=-0.03, seconds=0.8, hold=1.0)
    puppy.pose(seconds=0.6)


def push_up(puppy, reps=3):
    """Lowers and raises the whole body."""
    for _ in range(reps):
        puppy.pose(front=LOW_HEIGHT + 0.01, back=LOW_HEIGHT + 0.01, seconds=0.6, hold=0.1)
        puppy.pose(seconds=0.6, hold=0.1)


def nod(puppy, reps=3):
    """Pitches the body nose-down and nose-up."""
    for _ in range(reps):
        puppy.pose(front=STAND_HEIGHT - 0.03, back=STAND_HEIGHT, seconds=0.4, hold=0.0)
        puppy.pose(front=STAND_HEIGHT, back=STAND_HEIGHT - 0.03, seconds=0.4, hold=0.0)
    puppy.pose(seconds=0.4)


def wave(puppy, reps=3):
    """Shifts weight back, lifts the front-right paw and waves it."""
    puppy.pose(front=STAND_HEIGHT, back=STAND_HEIGHT - 0.02, front_y=0.01, back_y=0.02, seconds=0.6, hold=0.2)
    for _ in range(reps):
        puppy.move_feet({"rf": (-0.07, -0.04)}, seconds=0.3, hold=0.0)
        puppy.move_feet({"rf": (-0.05, -0.07)}, seconds=0.3, hold=0.0)
    puppy.pose(seconds=0.6)


def march(puppy, cycles=4):
    """Trots in place."""
    puppy.gait(0.0, 0.0, cycles)


def walk_forward(puppy, cycles=6):
    puppy.gait(0.04, 0.04, cycles)


def walk_backward(puppy, cycles=6):
    puppy.gait(-0.03, -0.03, cycles)


def turn_left(puppy, cycles=6):
    """Spins counter-clockwise in place: right legs push forward, left legs push back."""
    puppy.gait(-0.03, 0.03, cycles)


def turn_right(puppy, cycles=6):
    """Spins clockwise in place: left legs push forward, right legs push back."""
    puppy.gait(0.03, -0.03, cycles)


MOVES = {
    "stand": stand,
    "sit": sit,
    "lie_down": lie_down,
    "bow": bow,
    "push_up": push_up,
    "nod": nod,
    "wave": wave,
    "march": march,
    "walk_forward": walk_forward,
    "walk_backward": walk_backward,
    "turn_left": turn_left,
    "turn_right": turn_right,
}
assert list(MOVES) == MOVE_NAMES


# ---------------------------------------------------------------- main
def main():
    build_scene()
    stage_ok = check_stage()

    robot = Articulation(ROBOT_PATH)
    SimulationManager.set_physics_dt(PHYSICS_DT)
    app_utils.play()
    simulation_app.update()
    print(f"    dofs ({robot.num_dofs}): {robot.dof_names}")

    # Hold the legs in place so the robot stands instead of collapsing.
    robot.set_dof_gains(stiffnesses=np.full(robot.num_dofs, 20.0), dampings=np.full(robot.num_dofs, 0.5))
    puppy = Puppy(robot)
    puppy.set_feet({leg: (0.0, -STAND_HEIGHT) for leg in LEGS})

    if args.check:
        ok = stage_ok and check_gravity(robot) and check_joints(robot)
        print("\nCHECKS:", "ALL PASS" if ok else "SEE FAILURES ABOVE")
        puppy.set_feet({leg: (0.0, -STAND_HEIGHT) for leg in LEGS})
    step(240)  # let the robot drop and settle

    # Choose the movements here (or pass --moves on the command line).
    sequence = args.moves or DEFAULT_SEQUENCE
    for name in sequence:
        print(f"\n[move] {name}")
        MOVES[name](puppy)

    if args.keep_open and not args.headless:
        while simulation_app.is_running():
            step(1)

    app_utils.stop()
    simulation_app.close()


main()
