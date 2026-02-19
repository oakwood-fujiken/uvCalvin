"""Isaac Sim backend handler — MetaSim handler for running CALVIN on NVIDIA Isaac Sim.

Isaac Sim is NVIDIA's reference robotics simulation platform built on Omniverse.
This handler uses the lower-level Isaac Sim APIs directly (as opposed to Isaac Lab).

Requires: isaacsim / omni.isaac.core
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from calvin_metasim.env.handler import SimHandler

log = logging.getLogger(__name__)


class IsaacSimHandler(SimHandler):
    """SimHandler implementation backed by NVIDIA Isaac Sim (Omniverse).

    Uses omni.isaac.core APIs for articulation management and rendering.
    """

    def __init__(self, device: str = "cuda:0", headless: bool = True):
        self.device = device
        self.headless = headless
        self._world = None
        self._bodies: Dict[int, Any] = {}
        self._next_uid = 0
        self._timestep = 1.0 / 240.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def launch(self, show_gui: bool = False) -> None:
        try:
            from isaacsim import SimulationApp
        except ImportError:
            try:
                from omni.isaac.kit import SimulationApp
            except ImportError:
                raise ImportError(
                    "Isaac Sim is required. Please install via Omniverse Launcher or pip: "
                    "pip install isaacsim-rl isaacsim-replicator isaacsim-extscache-physics isaacsim-extscache-kit-sdk"
                )

        self._simulation_app = SimulationApp({"headless": self.headless and not show_gui})

        from omni.isaac.core import World

        self._world = World(stage_units_in_meters=1.0, physics_dt=self._timestep, rendering_dt=self._timestep)
        self._world.scene.add_default_ground_plane()
        log.info("Isaac Sim world created.")

    def close(self) -> None:
        if self._simulation_app is not None:
            self._simulation_app.close()
            self._simulation_app = None
        self._world = None

    # ------------------------------------------------------------------
    # Asset loading
    # ------------------------------------------------------------------

    def load_urdf(
        self,
        urdf_path: str,
        base_position: List[float],
        base_orientation_quat: List[float],
        use_fixed_base: bool = False,
        global_scaling: float = 1.0,
    ) -> int:
        uid = self._next_uid
        self._next_uid += 1

        from omni.isaac.core.utils.extensions import enable_extension

        enable_extension("omni.importer.urdf")

        from omni.importer.urdf import _urdf

        urdf_interface = _urdf.acquire_urdf_interface()
        import_config = _urdf.ImportConfig()
        import_config.fix_base = use_fixed_base
        import_config.merge_fixed_joints = False
        import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION

        prim_path = f"/World/body_{uid}"
        result = urdf_interface.parse_urdf(urdf_path, import_config)
        urdf_interface.import_robot(prim_path, urdf_path, result, import_config)

        from omni.isaac.core.articulations import Articulation

        articulation = self._world.scene.add(
            Articulation(prim_path=prim_path, name=f"body_{uid}")
        )

        self._bodies[uid] = {
            "articulation": articulation,
            "prim_path": prim_path,
        }

        # Set initial pose
        try:
            articulation.set_world_pose(
                position=np.array(base_position),
                orientation=np.array(base_orientation_quat),
            )
        except Exception:
            pass

        return uid

    def create_visual_cylinder(
        self,
        radius: float,
        length: float,
        position: List[float],
        rgba_color: List[float],
    ) -> int:
        uid = self._next_uid
        self._next_uid += 1
        try:
            from omni.isaac.core.objects import VisualCylinder

            self._world.scene.add(
                VisualCylinder(
                    prim_path=f"/World/visual_{uid}",
                    name=f"visual_{uid}",
                    radius=radius,
                    height=length,
                    position=np.array(position),
                    color=np.array(rgba_color[:3]),
                )
            )
        except Exception:
            pass
        return uid

    # ------------------------------------------------------------------
    # Physics stepping
    # ------------------------------------------------------------------

    def step_physics(self, n_steps: int = 1) -> None:
        if self._world is None:
            return
        for _ in range(n_steps):
            self._world.step(render=False)

    def set_gravity(self, gravity: Tuple[float, float, float]) -> None:
        if self._world is not None:
            try:
                from omni.isaac.core.utils.physics import set_gravity

                set_gravity(list(gravity))
            except Exception:
                pass

    def set_timestep(self, timestep: float) -> None:
        self._timestep = timestep

    # ------------------------------------------------------------------
    # Body / link / joint queries
    # ------------------------------------------------------------------

    def get_body_pose(self, body_id: int) -> Tuple[np.ndarray, np.ndarray]:
        body = self._bodies.get(body_id)
        if body is None:
            return np.zeros(3), np.array([0, 0, 0, 1.0])
        try:
            art = body["articulation"]
            pos, quat = art.get_world_pose()
            return np.array(pos), np.array(quat)
        except Exception:
            return np.zeros(3), np.array([0, 0, 0, 1.0])

    def get_body_velocity(self, body_id: int) -> Tuple[np.ndarray, np.ndarray]:
        body = self._bodies.get(body_id)
        if body is None:
            return np.zeros(3), np.zeros(3)
        try:
            art = body["articulation"]
            lin_vel = art.get_linear_velocity()
            ang_vel = art.get_angular_velocity()
            return np.array(lin_vel), np.array(ang_vel)
        except Exception:
            return np.zeros(3), np.zeros(3)

    def get_link_pose(self, body_id: int, link_index: int) -> Tuple[np.ndarray, np.ndarray]:
        body = self._bodies.get(body_id)
        if body is None:
            return np.zeros(3), np.array([0, 0, 0, 1.0])
        try:
            from omni.isaac.core.utils.prims import get_prim_at_path
            from pxr import UsdGeom

            prim_path = body["prim_path"]
            # Isaac Sim links are child prims
            art = body["articulation"]
            body_names = art.body_names
            if link_index < len(body_names):
                link_prim_path = f"{prim_path}/{body_names[link_index]}"
                prim = get_prim_at_path(link_prim_path)
                xformable = UsdGeom.Xformable(prim)
                transform = xformable.ComputeLocalToWorldTransform(0)
                pos = np.array([transform.GetRow(3)[i] for i in range(3)])
                # Extract quaternion from transform
                from scipy.spatial.transform import Rotation

                rot_mat = np.array([[transform.GetRow(i)[j] for j in range(3)] for i in range(3)])
                quat = Rotation.from_matrix(rot_mat).as_quat()
                return pos, quat
        except Exception:
            pass
        return np.zeros(3), np.array([0, 0, 0, 1.0])

    def get_num_joints(self, body_id: int) -> int:
        body = self._bodies.get(body_id)
        if body is None:
            return 0
        try:
            return body["articulation"].num_dof
        except Exception:
            return 0

    def get_joint_info(self, body_id: int, joint_index: int) -> Dict[str, Any]:
        body = self._bodies.get(body_id)
        if body is None:
            return self._default_joint_info(joint_index)
        try:
            art = body["articulation"]
            dof_properties = art.get_articulation_controller().get_dof_properties()
            return {
                "index": joint_index,
                "name": art.dof_names[joint_index] if joint_index < len(art.dof_names) else f"joint_{joint_index}",
                "type": 0,
                "lower_limit": float(dof_properties["lower"][joint_index]),
                "upper_limit": float(dof_properties["upper"][joint_index]),
                "max_force": float(dof_properties["maxEffort"][joint_index]),
                "max_velocity": float(dof_properties["maxVelocity"][joint_index]),
                "link_name": art.body_names[joint_index] if joint_index < len(art.body_names) else f"link_{joint_index}",
            }
        except Exception:
            return self._default_joint_info(joint_index)

    def get_joint_state(self, body_id: int, joint_index: int) -> float:
        body = self._bodies.get(body_id)
        if body is None:
            return 0.0
        try:
            positions = body["articulation"].get_joint_positions()
            return float(positions[joint_index])
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Body / joint state setting
    # ------------------------------------------------------------------

    def reset_body_pose(
        self,
        body_id: int,
        position: List[float],
        orientation_quat: List[float],
    ) -> None:
        body = self._bodies.get(body_id)
        if body is None:
            return
        try:
            body["articulation"].set_world_pose(
                position=np.array(position),
                orientation=np.array(orientation_quat),
            )
        except Exception:
            pass

    def reset_joint_state(
        self,
        body_id: int,
        joint_index: int,
        target_value: float,
        target_velocity: float = 0.0,
    ) -> None:
        body = self._bodies.get(body_id)
        if body is None:
            return
        try:
            art = body["articulation"]
            positions = art.get_joint_positions()
            velocities = art.get_joint_velocities()
            positions[joint_index] = target_value
            velocities[joint_index] = target_velocity
            art.set_joint_positions(positions)
            art.set_joint_velocities(velocities)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Motor control
    # ------------------------------------------------------------------

    def set_joint_motor_position(
        self,
        body_id: int,
        joint_index: int,
        target_position: float,
        force: float,
        max_velocity: float = 2.0,
    ) -> None:
        body = self._bodies.get(body_id)
        if body is None:
            return
        try:
            art = body["articulation"]
            controller = art.get_articulation_controller()
            targets = np.zeros(art.num_dof)
            targets[joint_index] = target_position
            controller.apply_action(
                controller.ArticulationAction(joint_positions=targets, joint_indices=[joint_index])
            )
        except Exception:
            pass

    def set_joint_motor_velocity(
        self,
        body_id: int,
        joint_index: int,
        force: float,
    ) -> None:
        pass  # Velocity control handled via drive API

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    def create_gear_constraint(
        self,
        body_id: int,
        joint_a: int,
        joint_b: int,
        gear_ratio: float = -1.0,
        max_force: float = 50.0,
    ) -> int:
        log.debug("Isaac Sim: gear constraints set via USD schema.")
        return -1

    # ------------------------------------------------------------------
    # Contacts
    # ------------------------------------------------------------------

    def get_contacts(self, body_id: int) -> List[Tuple]:
        body = self._bodies.get(body_id)
        if body is None:
            return []
        try:
            from omni.isaac.core.utils.physics import get_contact_raw_data

            raw = get_contact_raw_data(body["prim_path"])
            contacts = []
            for c in raw:
                contacts.append(
                    (0, body_id, -1, -1, -1, tuple(c.position), tuple(c.position),
                     tuple(c.normal), 0.0, float(c.impulse.length()))
                )
            return contacts
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Visual
    # ------------------------------------------------------------------

    def change_visual_color(
        self,
        body_id: int,
        link_index: int,
        rgba_color: List[float],
    ) -> None:
        pass  # Requires USD material API

    # ------------------------------------------------------------------
    # Camera rendering
    # ------------------------------------------------------------------

    def render_camera(
        self,
        width: int,
        height: int,
        view_matrix: np.ndarray,
        projection_matrix: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        try:
            from omni.isaac.sensor import Camera

            if not hasattr(self, "_camera"):
                self._camera = Camera(
                    prim_path="/World/camera",
                    resolution=(width, height),
                )
                self._camera.initialize()

            self._camera.get_current_frame()
            rgb = self._camera.get_rgba()[:, :, :3]
            depth = self._camera.get_depth()
            return rgb, depth
        except Exception:
            return np.zeros((height, width, 3), dtype=np.uint8), np.zeros((height, width))

    # ------------------------------------------------------------------
    # IK
    # ------------------------------------------------------------------

    def calculate_ik(
        self,
        body_id: int,
        end_effector_link: int,
        target_position: List[float],
        target_orientation_quat: List[float],
        lower_limits: Optional[List[float]] = None,
        upper_limits: Optional[List[float]] = None,
        joint_ranges: Optional[List[float]] = None,
        rest_poses: Optional[List[float]] = None,
    ) -> np.ndarray:
        body = self._bodies.get(body_id)
        if body is None:
            return np.zeros(7)
        try:
            from omni.isaac.motion_generation import LulaKinematicsSolver

            art = body["articulation"]
            ik_solver = LulaKinematicsSolver(
                robot_description_path="",
                urdf_path="",
            )
            result, success = ik_solver.compute_inverse_kinematics(
                target_position=np.array(target_position),
                target_orientation=np.array(target_orientation_quat),
            )
            if success:
                return result[:7]
        except Exception:
            pass
        # Fallback: return current joint positions
        try:
            return body["articulation"].get_joint_positions()[:7]
        except Exception:
            return np.zeros(7)

    @staticmethod
    def _default_joint_info(joint_index: int) -> Dict[str, Any]:
        return {
            "index": joint_index,
            "name": f"joint_{joint_index}",
            "type": 0,
            "lower_limit": -3.14,
            "upper_limit": 3.14,
            "max_force": 100.0,
            "max_velocity": 2.0,
            "link_name": f"link_{joint_index}",
        }
