"""Genesis backend handler — MetaSim handler for running CALVIN on Genesis.

Genesis is an open-source, GPU-accelerated physics simulation platform
with differentiable physics support. It provides a Python API for
robot simulation with MPM, FEM, rigid body, and articulated body solvers.

Requires: genesis-world (pip install genesis-world)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from calvin_metasim.env.handler import SimHandler

log = logging.getLogger(__name__)


class GenesisHandler(SimHandler):
    """SimHandler implementation backed by Genesis.

    Genesis provides unified GPU-accelerated simulation with differentiable
    physics and native MJCF/URDF support.
    """

    def __init__(self, device: str = "cuda:0", backend: str = "gpu"):
        self.device = device
        self.backend_type = backend
        self._scene = None
        self._bodies: Dict[int, Any] = {}
        self._next_uid = 0
        self._timestep = 1.0 / 240.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def launch(self, show_gui: bool = False) -> None:
        try:
            import genesis as gs
        except ImportError:
            raise ImportError(
                "Genesis is required. Install via: pip install genesis-world"
            )

        gs.init(backend=gs.cpu if self.backend_type == "cpu" else gs.gpu)

        self._gs = gs
        self._scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self._timestep,
                gravity=(0.0, 0.0, -9.8),
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(2.5, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.0),
                camera_fov=40,
            ),
            show_viewer=show_gui,
            rigid_options=gs.options.RigidOptions(
                enable_collision=True,
                enable_joint_limit=True,
            ),
        )
        self._scene.add_entity(gs.morphs.Plane())
        log.info("Genesis scene created.")

    def close(self) -> None:
        if self._scene is not None:
            self._scene = None
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

        # Convert quaternion (xyzw) to euler for Genesis morph
        euler = self.quat_to_euler(base_orientation_quat).tolist()

        entity = self._scene.add_entity(
            self._gs.morphs.URDF(
                file=urdf_path,
                pos=tuple(base_position),
                euler=tuple(euler),
                fixed=use_fixed_base,
                scale=global_scaling,
            ),
        )

        self._bodies[uid] = {
            "entity": entity,
            "urdf_path": urdf_path,
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
            entity = self._scene.add_entity(
                self._gs.morphs.Cylinder(
                    pos=tuple(position),
                    radius=radius,
                    height=length,
                    visualization=True,
                    collision=False,
                ),
            )
            self._bodies[uid] = {"entity": entity}
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

    def set_gravity(self, gravity: Tuple[float, float, float]) -> None:
        # Genesis gravity is set at scene creation; runtime change via options
        pass

    def set_timestep(self, timestep: float) -> None:
        self._timestep = timestep

    # ------------------------------------------------------------------
    # Body / link / joint queries
    # ------------------------------------------------------------------

    def get_body_pose(self, body_id: int) -> Tuple[np.ndarray, np.ndarray]:
        body = self._bodies.get(body_id)
        if body is None or "entity" not in body:
            return np.zeros(3), np.array([0, 0, 0, 1.0])
        try:
            entity = body["entity"]
            pos = entity.get_pos().cpu().numpy()
            quat = entity.get_quat().cpu().numpy()
            return pos, quat
        except Exception:
            return np.zeros(3), np.array([0, 0, 0, 1.0])

    def get_body_velocity(self, body_id: int) -> Tuple[np.ndarray, np.ndarray]:
        body = self._bodies.get(body_id)
        if body is None or "entity" not in body:
            return np.zeros(3), np.zeros(3)
        try:
            entity = body["entity"]
            vel = entity.get_vel().cpu().numpy()
            ang = entity.get_ang().cpu().numpy()
            return vel, ang
        except Exception:
            return np.zeros(3), np.zeros(3)

    def get_link_pose(self, body_id: int, link_index: int) -> Tuple[np.ndarray, np.ndarray]:
        body = self._bodies.get(body_id)
        if body is None or "entity" not in body:
            return np.zeros(3), np.array([0, 0, 0, 1.0])
        try:
            entity = body["entity"]
            links = entity.get_links()
            if link_index < len(links):
                link = links[link_index]
                pos = link.get_pos().cpu().numpy()
                quat = link.get_quat().cpu().numpy()
                return pos, quat
        except Exception:
            pass
        return np.zeros(3), np.array([0, 0, 0, 1.0])

    def get_num_joints(self, body_id: int) -> int:
        body = self._bodies.get(body_id)
        if body is None or "entity" not in body:
            return 0
        try:
            return body["entity"].n_dofs
        except Exception:
            return 0

    def get_joint_info(self, body_id: int, joint_index: int) -> Dict[str, Any]:
        body = self._bodies.get(body_id)
        if body is None or "entity" not in body:
            return self._default_joint_info(joint_index)
        try:
            entity = body["entity"]
            joints = entity.get_joints()
            if joint_index < len(joints):
                joint = joints[joint_index]
                return {
                    "index": joint_index,
                    "name": joint.name,
                    "type": 0,
                    "lower_limit": float(joint.dof_lower_limit) if hasattr(joint, "dof_lower_limit") else -3.14,
                    "upper_limit": float(joint.dof_upper_limit) if hasattr(joint, "dof_upper_limit") else 3.14,
                    "max_force": 100.0,
                    "max_velocity": 2.0,
                    "link_name": joint.name.replace("_joint", "_link"),
                }
        except Exception:
            pass
        return self._default_joint_info(joint_index)

    def get_joint_state(self, body_id: int, joint_index: int) -> float:
        body = self._bodies.get(body_id)
        if body is None or "entity" not in body:
            return 0.0
        try:
            dofs_pos = body["entity"].get_dofs_position().cpu().numpy()
            return float(dofs_pos[joint_index])
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
        if body is None or "entity" not in body:
            return
        try:
            import torch

            entity = body["entity"]
            entity.set_pos(torch.tensor(position, dtype=torch.float32, device=self.device))
            entity.set_quat(torch.tensor(orientation_quat, dtype=torch.float32, device=self.device))
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
        if body is None or "entity" not in body:
            return
        try:
            import torch

            entity = body["entity"]
            positions = entity.get_dofs_position()
            velocities = entity.get_dofs_velocity()
            positions[joint_index] = target_value
            velocities[joint_index] = target_velocity
            entity.set_dofs_position(positions)
            entity.set_dofs_velocity(velocities)
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
        if body is None or "entity" not in body:
            return
        try:
            import torch

            entity = body["entity"]
            entity.control_dofs_position(
                torch.tensor([target_position], dtype=torch.float32, device=self.device),
                dofs_idx_local=torch.tensor([joint_index], dtype=torch.long, device=self.device),
            )
        except Exception:
            pass

    def set_joint_motor_velocity(
        self,
        body_id: int,
        joint_index: int,
        force: float,
    ) -> None:
        body = self._bodies.get(body_id)
        if body is None or "entity" not in body:
            return
        try:
            import torch

            entity = body["entity"]
            entity.control_dofs_velocity(
                torch.tensor([0.0], dtype=torch.float32, device=self.device),
                dofs_idx_local=torch.tensor([joint_index], dtype=torch.long, device=self.device),
            )
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
        log.debug("Genesis: gear constraints defined in URDF/MJCF. Skipping runtime creation.")
        return -1

    # ------------------------------------------------------------------
    # Contacts
    # ------------------------------------------------------------------

    def get_contacts(self, body_id: int) -> List[Tuple]:
        body = self._bodies.get(body_id)
        if body is None or "entity" not in body:
            return []
        try:
            entity = body["entity"]
            links = entity.get_links()
            contacts = []
            for link_idx, link in enumerate(links):
                if link.is_in_contact():
                    contact_forces = link.get_contact_forces().cpu().numpy()
                    force_mag = np.linalg.norm(contact_forces)
                    if force_mag > 0.01:
                        contacts.append(
                            (0, body_id, -1, link_idx, -1,
                             (0, 0, 0), (0, 0, 0), (0, 0, 1), 0.0, force_mag)
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
        # Genesis visual changes via material API
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
                    res=(width, height),
                    pos=(2.5, 0.0, 2.5),
                    lookat=(0.0, 0.0, 0.0),
                    fov=40,
                )

            self._camera.render()
            rgb = self._camera.get_color().cpu().numpy()
            depth = self._camera.get_depth().cpu().numpy()
            return rgb[:, :, :3], depth
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
        if body is None or "entity" not in body:
            return np.zeros(7)
        try:
            import torch

            entity = body["entity"]
            result = entity.inverse_kinematics(
                link=entity.get_links()[end_effector_link],
                pos=torch.tensor(target_position, dtype=torch.float32, device=self.device),
                quat=torch.tensor(target_orientation_quat, dtype=torch.float32, device=self.device),
            )
            return result.cpu().numpy()[:7]
        except Exception:
            pass
        # Fallback
        try:
            return body["entity"].get_dofs_position().cpu().numpy()[:7]
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
