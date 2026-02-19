"""SAPIEN backend handler — MetaSim handler for running CALVIN on SAPIEN.

SAPIEN is an open-source robotics simulation platform developed by
Hao Su's lab (UCSD), supporting PhysX-based rigid body simulation,
ray-tracing rendering, and manipulation-focused APIs.

Requires: sapien (pip install sapien)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from calvin_metasim.env.handler import SimHandler

log = logging.getLogger(__name__)


class SapienHandler(SimHandler):
    """SimHandler implementation backed by SAPIEN.

    SAPIEN provides PhysX5-based simulation with ray-traced rendering
    and rich articulation support.
    """

    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self._engine = None
        self._scene = None
        self._renderer = None
        self._bodies: Dict[int, Any] = {}
        self._next_uid = 0
        self._timestep = 1.0 / 240.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def launch(self, show_gui: bool = False) -> None:
        try:
            import sapien
        except ImportError:
            raise ImportError(
                "SAPIEN is required. Install via: pip install sapien"
            )

        self._sapien = sapien
        self._engine = sapien.Engine()

        if show_gui:
            self._renderer = sapien.SapienRenderer()
        else:
            self._renderer = sapien.SapienRenderer(offscreen_only=True)

        self._engine.set_renderer(self._renderer)

        self._scene = self._engine.create_scene()
        self._scene.set_timestep(self._timestep)
        self._scene.add_ground(altitude=0)

        # Ambient lighting
        self._scene.set_ambient_light([0.5, 0.5, 0.5])
        self._scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

        if show_gui:
            self._viewer = self._sapien.utils.Viewer(self._renderer)
            self._viewer.set_scene(self._scene)
            self._viewer.set_camera_xyz(x=2.0, y=0.0, z=2.5)
            self._viewer.set_camera_rpy(r=0, p=-0.5, y=3.14)
        else:
            self._viewer = None

        log.info("SAPIEN scene created.")

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        self._scene = None
        self._engine = None
        self._bodies.clear()

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

        loader = self._scene.create_urdf_loader()
        loader.fix_root_link = use_fixed_base
        loader.scale = global_scaling

        articulation = loader.load(urdf_path)
        if articulation is None:
            raise RuntimeError(f"Failed to load URDF: {urdf_path}")

        # Set initial pose
        pose = self._sapien.Pose(p=base_position, q=base_orientation_quat)
        articulation.set_root_pose(pose)

        self._bodies[uid] = {
            "articulation": articulation,
            "loader": loader,
        }
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
            builder = self._scene.create_actor_builder()
            builder.add_visual_from_mesh(
                self._sapien.Pose(p=position),
                # Use capsule as visual-only proxy
            )
            builder.add_capsule_visual(
                self._sapien.Pose(p=position),
                radius=radius,
                half_length=length / 2,
                color=rgba_color[:3],
            )
            actor = builder.build_static(name=f"visual_{uid}")
            self._bodies[uid] = {"actor": actor}
        except Exception:
            pass
        return uid

    # ------------------------------------------------------------------
    # Physics stepping
    # ------------------------------------------------------------------

    def step_physics(self, n_steps: int = 1) -> None:
        if self._scene is None:
            return
        for _ in range(n_steps):
            self._scene.step()
        if self._viewer is not None:
            self._scene.update_render()
            self._viewer.render()

    def set_gravity(self, gravity: Tuple[float, float, float]) -> None:
        if self._scene is not None:
            self._scene.set_gravity(list(gravity))

    def set_timestep(self, timestep: float) -> None:
        self._timestep = timestep
        if self._scene is not None:
            self._scene.set_timestep(timestep)

    # ------------------------------------------------------------------
    # Body / link / joint queries
    # ------------------------------------------------------------------

    def get_body_pose(self, body_id: int) -> Tuple[np.ndarray, np.ndarray]:
        body = self._bodies.get(body_id)
        if body is None:
            return np.zeros(3), np.array([0, 0, 0, 1.0])
        try:
            if "articulation" in body:
                pose = body["articulation"].get_root_pose()
            elif "actor" in body:
                pose = body["actor"].get_pose()
            else:
                return np.zeros(3), np.array([0, 0, 0, 1.0])
            return np.array(pose.p), np.array(pose.q)
        except Exception:
            return np.zeros(3), np.array([0, 0, 0, 1.0])

    def get_body_velocity(self, body_id: int) -> Tuple[np.ndarray, np.ndarray]:
        body = self._bodies.get(body_id)
        if body is None:
            return np.zeros(3), np.zeros(3)
        try:
            if "articulation" in body:
                root = body["articulation"].get_links()[0]
                return np.array(root.get_velocity()), np.array(root.get_angular_velocity())
        except Exception:
            pass
        return np.zeros(3), np.zeros(3)

    def get_link_pose(self, body_id: int, link_index: int) -> Tuple[np.ndarray, np.ndarray]:
        body = self._bodies.get(body_id)
        if body is None or "articulation" not in body:
            return np.zeros(3), np.array([0, 0, 0, 1.0])
        try:
            links = body["articulation"].get_links()
            if link_index < len(links):
                pose = links[link_index].get_pose()
                return np.array(pose.p), np.array(pose.q)
        except Exception:
            pass
        return np.zeros(3), np.array([0, 0, 0, 1.0])

    def get_num_joints(self, body_id: int) -> int:
        body = self._bodies.get(body_id)
        if body is None or "articulation" not in body:
            return 0
        try:
            return body["articulation"].dof
        except Exception:
            return 0

    def get_joint_info(self, body_id: int, joint_index: int) -> Dict[str, Any]:
        body = self._bodies.get(body_id)
        if body is None or "articulation" not in body:
            return self._default_joint_info(joint_index)
        try:
            art = body["articulation"]
            joints = art.get_active_joints()
            if joint_index < len(joints):
                joint = joints[joint_index]
                limits = joint.get_limits()
                links = art.get_links()
                link_name = links[joint_index].get_name() if joint_index < len(links) else f"link_{joint_index}"
                return {
                    "index": joint_index,
                    "name": joint.get_name(),
                    "type": 0,
                    "lower_limit": float(limits[0][0]),
                    "upper_limit": float(limits[0][1]),
                    "max_force": 100.0,
                    "max_velocity": 2.0,
                    "link_name": link_name,
                }
        except Exception:
            pass
        return self._default_joint_info(joint_index)

    def get_joint_state(self, body_id: int, joint_index: int) -> float:
        body = self._bodies.get(body_id)
        if body is None or "articulation" not in body:
            return 0.0
        try:
            qpos = body["articulation"].get_qpos()
            return float(qpos[joint_index])
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
            pose = self._sapien.Pose(p=position, q=orientation_quat)
            if "articulation" in body:
                body["articulation"].set_root_pose(pose)
            elif "actor" in body:
                body["actor"].set_pose(pose)
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
        if body is None or "articulation" not in body:
            return
        try:
            art = body["articulation"]
            qpos = art.get_qpos()
            qvel = art.get_qvel()
            qpos[joint_index] = target_value
            qvel[joint_index] = target_velocity
            art.set_qpos(qpos)
            art.set_qvel(qvel)
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
        if body is None or "articulation" not in body:
            return
        try:
            art = body["articulation"]
            joints = art.get_active_joints()
            if joint_index < len(joints):
                joint = joints[joint_index]
                joint.set_drive_property(stiffness=force * 10, damping=force)
                joint.set_drive_target(target_position)
                joint.set_drive_velocity_target(0)
        except Exception:
            pass

    def set_joint_motor_velocity(
        self,
        body_id: int,
        joint_index: int,
        force: float,
    ) -> None:
        body = self._bodies.get(body_id)
        if body is None or "articulation" not in body:
            return
        try:
            art = body["articulation"]
            joints = art.get_active_joints()
            if joint_index < len(joints):
                joint = joints[joint_index]
                joint.set_drive_property(stiffness=0, damping=force)
                joint.set_drive_velocity_target(0)
        except Exception:
            pass

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
        log.debug("SAPIEN: mimic joints handled via drive coupling. Skipping runtime creation.")
        return -1

    # ------------------------------------------------------------------
    # Contacts
    # ------------------------------------------------------------------

    def get_contacts(self, body_id: int) -> List[Tuple]:
        body = self._bodies.get(body_id)
        if body is None:
            return []
        try:
            contacts = self._scene.get_contacts()
            result = []
            art = body.get("articulation")
            if art is None:
                return []
            link_set = set(id(l) for l in art.get_links())
            for contact in contacts:
                actors = [contact.actor0, contact.actor1]
                if any(id(a) in link_set for a in actors):
                    for point in contact.points:
                        result.append(
                            (0, body_id, -1, -1, -1,
                             tuple(point.position), tuple(point.position),
                             tuple(point.normal), float(point.separation),
                             float(point.impulse.sum()))
                        )
            return result
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
        body = self._bodies.get(body_id)
        if body is None or "articulation" not in body:
            return
        try:
            links = body["articulation"].get_links()
            if link_index < len(links):
                for visual in links[link_index].get_visual_bodies():
                    for shape in visual.get_render_shapes():
                        mat = shape.material
                        mat.set_base_color(rgba_color)
                        shape.set_material(mat)
        except Exception:
            pass

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
        if self._scene is None:
            return np.zeros((height, width, 3), dtype=np.uint8), np.zeros((height, width))

        try:
            if not hasattr(self, "_camera"):
                self._camera = self._scene.add_camera(
                    name="metasim_camera",
                    width=width,
                    height=height,
                    fovy=np.deg2rad(45),
                    near=0.01,
                    far=10.0,
                )

            # Apply view matrix to camera
            view_4x4 = view_matrix.reshape(4, 4)
            self._camera.set_local_pose(
                self._sapien.Pose.from_transformation_matrix(np.linalg.inv(view_4x4))
            )

            self._scene.update_render()
            self._camera.take_picture()
            rgb = self._camera.get_color_rgba()[:, :, :3]
            rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
            depth = -self._camera.get_float_texture("Position")[:, :, 2]
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
        """IK using SAPIEN's built-in pinocchio-based IK solver."""
        body = self._bodies.get(body_id)
        if body is None or "articulation" not in body:
            return np.zeros(7)

        try:
            art = body["articulation"]
            model = art.create_pinocchio_model()
            target_pose = self._sapien.Pose(p=target_position, q=target_orientation_quat)

            result, success, error = model.compute_inverse_kinematics(
                link_index=end_effector_link,
                pose=target_pose,
                initial_qpos=art.get_qpos(),
                active_qmask=[1] * 7 + [0] * (art.dof - 7) if art.dof > 7 else [1] * art.dof,
            )
            if success:
                return np.array(result[:7])
        except Exception:
            pass

        # Fallback
        try:
            return body["articulation"].get_qpos()[:7]
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
