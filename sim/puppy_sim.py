"""Isaac Sim side of the prototype: builds a world file's scene and drives the PuppyPi from velocity commands.

PuppySim implements the `Robot` interface of puppy_agent.navigation:
  set_cmd_vel(vx, wz)  body velocity in, as from Nav2's controller on /cmd_vel
  odom()               base pose in the map frame (ground truth here; real odometry later)
  advance(seconds)     step physics, running the trot gait
  now()                sim time
plus capture() for the front camera and safety() for fall detection.

Must be imported after SimulationApp has been created.
"""
import math

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
import omni.replicator.core as rep
from isaacsim.core.experimental.objects import DistantLight, GroundPlane
from isaacsim.core.experimental.prims import Articulation, XformPrim
from isaacsim.core.simulation_manager import SimulationManager
from pxr import Gf, UsdGeom, UsdLux, UsdPhysics

from puppy_agent.gait import LEGS, TrotGait, leg_ik, stand_feet
from puppy_agent.navigation import Pose2D

ROBOT_PATH = "/World/Puppy"
BASE_LINK = ROBOT_PATH + "/Geometry/base_link"
PHYSICS_DT = 1.0 / 120.0
SPAWN_Z = 0.25
CAMERA_OFFSET = (0.10, 0.0, 0.06)  # front camera position on the base (x forward, z up)
CAMERA_FOCAL_MM, CAMERA_APERTURE_MM = 12.0, 20.955
CAMERA_HFOV = 2 * math.atan(CAMERA_APERTURE_MM / (2 * CAMERA_FOCAL_MM))
FALL_HEIGHT_M, FALL_TILT_DEG = 0.04, 50.0


def _box(stage, path, center, size, color, collide=True):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.Xformable(cube)
    xform.AddTranslateOp().Set(Gf.Vec3d(*center))
    xform.AddScaleOp().Set(Gf.Vec3d(*size))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())


def _sphere(stage, path, center, radius, color):
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(radius)
    UsdGeom.Xformable(sphere).AddTranslateOp().Set(Gf.Vec3d(*center))
    sphere.CreateDisplayColorAttr([Gf.Vec3f(*color)])


def build_world(stage, world):
    """Ground, lights, walls and every entity with a shape that is present in the sim."""
    GroundPlane("/World/GroundPlane", positions=[0, 0, 0])
    DistantLight("/World/DistantLight").set_intensities(1000)
    UsdLux.DomeLight.Define(stage, "/World/Dome").CreateIntensityAttr(500)
    room = world.get("room")
    if room:
        (sx, sy), h, color, t = room["size"], room["wall_height"], room.get("wall_color", (0.75,) * 3), 0.05
        _box(stage, "/World/Walls/north", (0, sy / 2, h / 2), (sx, t, h), color)
        _box(stage, "/World/Walls/south", (0, -sy / 2, h / 2), (sx, t, h), color)
        _box(stage, "/World/Walls/east", (sx / 2, 0, h / 2), (t, sy, h), color)
        _box(stage, "/World/Walls/west", (-sx / 2, 0, h / 2), (t, sy, h), color)
    for name, spec in world["entities"].items():
        shape = spec.get("shape")
        if not shape or not spec.get("present_in_sim", True):
            continue
        path = f"/World/Entities/{name}"
        if shape["type"] == "box":
            _box(stage, path, spec["position"], shape["size"], shape["color"])
        elif shape["type"] == "sphere":
            _sphere(stage, path, spec["position"], shape["radius"], shape["color"])


class PuppySim:
    def __init__(self, simulation_app, world, usd_path, resolution=(320, 240), stiffness=20.0, damping=0.5):
        self.app = simulation_app
        self.world = world
        stage_utils.create_new_stage()
        stage_utils.set_stage_units(meters_per_unit=1.0)
        stage = stage_utils.get_current_stage()
        build_world(stage, world)

        stage_utils.add_reference_to_stage(usd_path=usd_path, path=ROBOT_PATH)
        # The URDF converter puts all physics behind a "Physics" variant; select PhysX.
        stage.GetPrimAtPath(ROBOT_PATH).GetVariantSets().GetVariantSet("Physics").SetVariantSelection("physx")
        x, y, yaw_deg = world["robot"]["start"]
        half = math.radians(yaw_deg) / 2
        XformPrim(ROBOT_PATH, reset_xform_op_properties=True).set_world_poses(
            positions=[x, y, SPAWN_Z], orientations=[math.cos(half), 0.0, 0.0, math.sin(half)])

        # Front camera. USD cameras look down -Z with +Y up; this rotation makes it look along base +X.
        camera = UsdGeom.Camera.Define(stage, BASE_LINK + "/FrontCamera")
        camera.CreateFocalLengthAttr(CAMERA_FOCAL_MM)
        camera.CreateHorizontalApertureAttr(CAMERA_APERTURE_MM)
        camera.CreateVerticalApertureAttr(CAMERA_APERTURE_MM * resolution[1] / resolution[0])
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 50.0))
        cx, cy, cz = CAMERA_OFFSET
        UsdGeom.Xformable(camera).AddTransformOp().Set(Gf.Matrix4d(0, -1, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, cx, cy, cz, 1))
        render_product = rep.create.render_product(str(camera.GetPath()), resolution)
        self._rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        self._rgb.attach([render_product])

        self.robot = Articulation(ROBOT_PATH)
        SimulationManager.set_physics_dt(PHYSICS_DT)
        app_utils.play()
        self.app.update()
        self.robot.set_dof_gains(stiffnesses=np.full(self.robot.num_dofs, stiffness),
                                 dampings=np.full(self.robot.num_dofs, damping))
        names = list(self.robot.dof_names)
        self._hip = {leg: names.index(f"{leg}_joint1") for leg in LEGS}
        self._knee = {leg: names.index(f"{leg}_joint2") for leg in LEGS}
        self._targets = np.zeros(self.robot.num_dofs)
        self.gait = TrotGait()
        self.vx = self.wz = 0.0
        self.t = 0.0
        self._set_feet(stand_feet())
        self.advance(2.0)  # drop and settle

    # -- Robot interface -------------------------------------------------------------------------
    def set_cmd_vel(self, vx, wz):
        self.vx, self.wz = float(vx), float(wz)

    def now(self):
        return self.t

    def odom(self):
        x, y, _, yaw, _, _ = self._base_state()
        return Pose2D(x, y, yaw)

    def advance(self, seconds):
        for _ in range(max(1, round(seconds / PHYSICS_DT))):
            self._set_feet(self.gait.update(PHYSICS_DT, self.vx, self.wz))
            SimulationManager.step()
            self.app.update()
            self.t += PHYSICS_DT

    # -- sensors and safety ----------------------------------------------------------------------
    def capture(self):
        rep.orchestrator.step(rt_subframes=4, pause_timeline=False)
        return np.asarray(self._rgb.get_data())[:, :, :3].copy()

    def camera_pose(self):
        x, y, _, yaw, _, _ = self._base_state()
        return Pose2D(x + CAMERA_OFFSET[0] * math.cos(yaw), y + CAMERA_OFFSET[0] * math.sin(yaw), yaw)

    def safety(self):
        _, _, z, _, roll, pitch = self._base_state()
        if z < FALL_HEIGHT_M:
            return f"base height {z:.3f} m below {FALL_HEIGHT_M} m"
        if max(abs(roll), abs(pitch)) > math.radians(FALL_TILT_DEG):
            return f"tilt roll={math.degrees(roll):.0f} pitch={math.degrees(pitch):.0f} deg"
        return ""

    def locate(self, name):
        """Ground-truth (x, y) of an entity actually present in the scene, else None."""
        spec = self.world["entities"].get(name)
        if not spec or "position" not in spec or not spec.get("present_in_sim", True):
            return None
        return tuple(spec["position"][:2])

    def close(self):
        app_utils.stop()

    # -- internals -------------------------------------------------------------------------------
    def _set_feet(self, feet):
        for leg, (y, z) in feet.items():
            self._targets[self._hip[leg]], self._targets[self._knee[leg]] = leg_ik(y, z)
        self.robot.set_dof_position_targets(self._targets)

    def _base_state(self):
        positions, orientations = self.robot.get_world_poses()
        x, y, z = (float(v) for v in positions.numpy()[0])
        w, qx, qy, qz = (float(v) for v in orientations.numpy()[0])
        yaw = math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        roll = math.atan2(2 * (w * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
        pitch = math.asin(max(-1.0, min(1.0, 2 * (w * qy - qz * qx))))
        return x, y, z, yaw, roll, pitch
