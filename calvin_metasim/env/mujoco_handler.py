"""MuJoCo backend handler — MetaSim handler for running CALVIN in MuJoCo.

This allows CALVIN tasks to run in MuJoCo instead of PyBullet, which enables
GPU-accelerated rendering and compatibility with MuJoCo-based RL frameworks.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from calvin_metasim.env.handler import SimHandler

log = logging.getLogger(__name__)


class MuJoCoHandler(SimHandler):
    """SimHandler implementation backed by MuJoCo (dm_control / mujoco bindings).

    This handler converts CALVIN's URDF assets to MJCF on the fly and
    manages a MuJoCo simulation instance.
    """

    def __init__(self, use_egl: bool = True):
        self.use_egl = use_egl
        self._model = None
        self._data = None
        self._renderer = None
        self._body_uids: Dict[int, str] = {}  # uid -> body name in MuJoCo model
        self._next_uid = 0
        self._loaded_xmls: List[str] = []
        self._launched = False
        self._timestep = 1.0 / 240.0

        # Track loaded bodies for URDF->MJCF mapping
        self._uid_to_body_name: Dict[int, str] = {}
        self._uid_to_joint_offset: Dict[int, int] = {}  # first joint index for this body
        self._uid_to_njoint: Dict[int, int] = {}
        self._uid_to_nlink: Dict[int, int] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def launch(self, show_gui: bool = False) -> None:
        try:
            import mujoco
        except ImportError:
            raise ImportError(
                "MuJoCo Python bindings required. Install via: pip install mujoco"
            )

        if self.use_egl and not show_gui:
            os.environ.setdefault("MUJOCO_GL", "egl")

        self._mj = mujoco
        self._launched = True
        log.info("MuJoCo handler initialized (lazy model loading on first URDF).")

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._model = None
        self._data = None
        self._launched = False

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
        from calvin_metasim.assets.asset_utils import urdf_to_mjcf

        mjcf_path = urdf_to_mjcf(urdf_path)
        uid = self._next_uid
        self._next_uid += 1

        # Load or merge into existing model
        if self._model is None:
            self._model = self._mj.MjModel.from_xml_path(mjcf_path)
            self._data = self._mj.MjData(self._model)
            body_name = self._model.body(1).name if self._model.nbody > 1 else f"body_{uid}"
            n_joints = self._model.njnt
            n_links = self._model.nbody - 1
        else:
            # For multi-body scenes, we use mujoco's attach/merge functionality
            # This is a simplified approach — full scene merging would require
            # building a composite MJCF
            log.warning(
                f"MuJoCo handler: multi-URDF loading (uid={uid}) requires scene "
                f"pre-compilation. Using placeholder for {urdf_path}."
            )
            body_name = f"body_{uid}"
            n_joints = 0
            n_links = 0

        self._uid_to_body_name[uid] = body_name
        self._uid_to_njoint[uid] = n_joints
        self._uid_to_nlink[uid] = n_links

        # Set initial pose
        if self._data is not None and uid == 0:
            self._data.qpos[:3] = base_position
            self._data.qpos[3:7] = base_orientation_quat
            self._mj.mj_forward(self._model, self._data)

        return uid

    def create_visual_cylinder(
        self,
        radius: float,
        length: float,
        position: List[float],
        rgba_color: List[float],
    ) -> int:
        # MuJoCo visual-only geoms are created differently.
        # For the cosmetic robot base, we skip this in MuJoCo.
        uid = self._next_uid
        self._next_uid += 1
        return uid

    # ------------------------------------------------------------------
    # Physics stepping
    # ------------------------------------------------------------------

    def step_physics(self, n_steps: int = 1) -> None:
        if self._model is None or self._data is None:
            return
        for _ in range(n_steps):
            self._mj.mj_step(self._model, self._data)

    def set_gravity(self, gravity: Tuple[float, float, float]) -> None:
        if self._model is not None:
            self._model.opt.gravity[:] = gravity

    def set_timestep(self, timestep: float) -> None:
        self._timestep = timestep
        if self._model is not None:
            self._model.opt.timestep = timestep

    # ------------------------------------------------------------------
    # Body / link / joint queries
    # ------------------------------------------------------------------

    def get_body_pose(self, body_id: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._data is None:
            return np.zeros(3), np.array([0, 0, 0, 1.0])
        name = self._uid_to_body_name.get(body_id, f"body_{body_id}")
        try:
            body = self._data.body(name)
            return body.xpos.copy(), body.xquat.copy()
        except Exception:
            return np.zeros(3), np.array([0, 0, 0, 1.0])

    def get_body_velocity(self, body_id: int) -> Tuple[np.ndarray, np.ndarray]:
        # MuJoCo stores velocities in self._data.cvel for bodies
        return np.zeros(3), np.zeros(3)

    def get_link_pose(self, body_id: int, link_index: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._data is None:
            return np.zeros(3), np.array([0, 0, 0, 1.0])
        # In MuJoCo, links correspond to bodies (child bodies of the root)
        try:
            # link_index maps to body index offset
            body_idx = link_index + 1  # offset for world body
            pos = self._data.xpos[body_idx].copy()
            quat = self._data.xquat[body_idx].copy()
            return pos, quat
        except (IndexError, KeyError):
            return np.zeros(3), np.array([0, 0, 0, 1.0])

    def get_num_joints(self, body_id: int) -> int:
        return self._uid_to_njoint.get(body_id, 0)

    def get_joint_info(self, body_id: int, joint_index: int) -> Dict[str, Any]:
        if self._model is None:
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
        try:
            joint = self._model.jnt(joint_index)
            return {
                "index": joint_index,
                "name": joint.name,
                "type": int(joint.type[0]) if hasattr(joint.type, '__len__') else int(joint.type),
                "lower_limit": float(self._model.jnt_range[joint_index, 0]),
                "upper_limit": float(self._model.jnt_range[joint_index, 1]),
                "max_force": 100.0,
                "max_velocity": 2.0,
                "link_name": self._model.body(joint.bodyid[0]).name if hasattr(joint, 'bodyid') else f"link_{joint_index}",
            }
        except Exception:
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

    def get_joint_state(self, body_id: int, joint_index: int) -> float:
        if self._data is None:
            return 0.0
        try:
            return float(self._data.qpos[joint_index])
        except (IndexError, KeyError):
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
        if self._data is None:
            return
        name = self._uid_to_body_name.get(body_id, f"body_{body_id}")
        try:
            joint_id = self._model.body(name).jntadr[0]
            qpos_adr = self._model.jnt_qposadr[joint_id]
            self._data.qpos[qpos_adr : qpos_adr + 3] = position
            self._data.qpos[qpos_adr + 3 : qpos_adr + 7] = orientation_quat
            self._mj.mj_forward(self._model, self._data)
        except Exception:
            pass

    def reset_joint_state(
        self,
        body_id: int,
        joint_index: int,
        target_value: float,
        target_velocity: float = 0.0,
    ) -> None:
        if self._data is None:
            return
        try:
            self._data.qpos[joint_index] = target_value
            self._data.qvel[joint_index] = target_velocity
        except (IndexError, KeyError):
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
        if self._data is None:
            return
        try:
            self._data.ctrl[joint_index] = target_position
        except (IndexError, KeyError):
            pass

    def set_joint_motor_velocity(
        self,
        body_id: int,
        joint_index: int,
        force: float,
    ) -> None:
        if self._data is None:
            return
        try:
            self._data.ctrl[joint_index] = 0.0
        except (IndexError, KeyError):
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
        # MuJoCo handles gear constraints via equality constraints in MJCF.
        # This would need to be baked into the MJCF during conversion.
        log.debug("Gear constraints should be defined in MJCF. Skipping runtime creation.")
        return -1

    # ------------------------------------------------------------------
    # Contacts
    # ------------------------------------------------------------------

    def get_contacts(self, body_id: int) -> List[Tuple]:
        if self._data is None:
            return []
        contacts = []
        for i in range(self._data.ncon):
            contact = self._data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2
            body1 = self._model.geom_bodyid[geom1]
            body2 = self._model.geom_bodyid[geom2]
            if body1 == body_id or body2 == body_id:
                # Return in PyBullet-compatible format
                contacts.append(
                    (0, body1, body2, -1, -1, contact.pos, contact.pos, contact.frame[:3], contact.dist, 0.0)
                )
        return contacts

    # ------------------------------------------------------------------
    # Visual
    # ------------------------------------------------------------------

    def change_visual_color(
        self,
        body_id: int,
        link_index: int,
        rgba_color: List[float],
    ) -> None:
        if self._model is None:
            return
        try:
            geom_id = self._model.body(link_index + 1).geomadr[0]
            self._model.geom_rgba[geom_id] = rgba_color
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
        if self._model is None or self._data is None:
            return np.zeros((height, width, 3), dtype=np.uint8), np.zeros((height, width))

        if self._renderer is None:
            self._renderer = self._mj.Renderer(self._model, height=height, width=width)

        self._renderer.update_scene(self._data)
        rgb = self._renderer.render()
        # Depth rendering
        self._renderer.enable_depth_rendering()
        depth = self._renderer.render()
        self._renderer.disable_depth_rendering()

        return rgb, depth

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
        """IK via MuJoCo's damped least-squares Jacobian method."""
        if self._model is None or self._data is None:
            return np.zeros(7)

        target_pos = np.array(target_position)
        target_quat = np.array(target_orientation_quat)

        # Simple iterative IK
        max_iterations = 100
        tolerance = 1e-4
        damping = 1e-4

        data_copy = self._mj.MjData(self._model)
        data_copy.qpos[:] = self._data.qpos[:]
        self._mj.mj_forward(self._model, data_copy)

        site_id = end_effector_link
        jacp = np.zeros((3, self._model.nv))
        jacr = np.zeros((3, self._model.nv))

        for _ in range(max_iterations):
            self._mj.mj_forward(self._model, data_copy)
            current_pos = data_copy.xpos[site_id + 1].copy()
            pos_error = target_pos - current_pos

            if np.linalg.norm(pos_error) < tolerance:
                break

            self._mj.mj_jac(self._model, data_copy, jacp, jacr, target_pos, site_id + 1)
            J = jacp
            JtJ = J @ J.T + damping * np.eye(3)
            delta_q = J.T @ np.linalg.solve(JtJ, pos_error)
            data_copy.qpos[:len(delta_q)] += delta_q
            # Clamp to joint limits if provided
            if lower_limits is not None:
                n = min(len(lower_limits), len(delta_q))
                data_copy.qpos[:n] = np.clip(
                    data_copy.qpos[:n],
                    lower_limits[:n],
                    upper_limits[:n],
                )

        return data_copy.qpos[:7].copy()
