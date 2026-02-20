"""Simulator handler interface — the core abstraction layer.

Each simulator backend implements this interface. The CalvinMetaSimEnv
delegates all physics, rendering, and state queries to the handler.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class SimHandler(abc.ABC):
    """Abstract base for simulator backends.

    A handler manages the lifetime of a single simulation instance and
    provides a uniform API for:
      - loading URDFs / creating objects
      - stepping physics
      - querying body/joint/link states
      - rendering cameras
      - setting body/joint states (for reset)
      - querying contacts
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def launch(self, show_gui: bool = False) -> None:
        """Start the simulator."""

    @abc.abstractmethod
    def close(self) -> None:
        """Shut down the simulator and release resources."""

    # ------------------------------------------------------------------
    # Asset loading
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def load_urdf(
        self,
        urdf_path: str,
        base_position: List[float],
        base_orientation_quat: List[float],
        use_fixed_base: bool = False,
        global_scaling: float = 1.0,
    ) -> int:
        """Load a URDF and return a unique body id."""

    @abc.abstractmethod
    def create_visual_cylinder(
        self,
        radius: float,
        length: float,
        position: List[float],
        rgba_color: List[float],
    ) -> int:
        """Create a visual-only cylinder (used for the robot base cosmetic)."""

    # ------------------------------------------------------------------
    # Physics stepping
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def step_physics(self, n_steps: int = 1) -> None:
        """Advance the simulation by n_steps timesteps."""

    @abc.abstractmethod
    def set_gravity(self, gravity: Tuple[float, float, float]) -> None:
        """Set the gravity vector."""

    @abc.abstractmethod
    def set_timestep(self, timestep: float) -> None:
        """Set the physics timestep."""

    # ------------------------------------------------------------------
    # Body / link / joint queries
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def get_body_pose(self, body_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (position[3], orientation_quat[4]) of a body's base."""

    @abc.abstractmethod
    def get_body_velocity(self, body_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (linear_vel[3], angular_vel[3]) of a body's base."""

    @abc.abstractmethod
    def get_link_pose(self, body_id: int, link_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (position[3], orientation_quat[4]) of a specific link."""

    @abc.abstractmethod
    def get_num_joints(self, body_id: int) -> int:
        """Return the number of joints for a body."""

    @abc.abstractmethod
    def get_joint_info(self, body_id: int, joint_index: int) -> Dict[str, Any]:
        """Return joint info dict with at least: 'name', 'lower_limit', 'upper_limit', 'link_name'."""

    @abc.abstractmethod
    def get_joint_state(self, body_id: int, joint_index: int) -> float:
        """Return the current position of a joint."""

    # ------------------------------------------------------------------
    # Body / joint state setting (for reset)
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def reset_body_pose(
        self,
        body_id: int,
        position: List[float],
        orientation_quat: List[float],
    ) -> None:
        """Teleport a body to a given pose."""

    @abc.abstractmethod
    def reset_joint_state(
        self,
        body_id: int,
        joint_index: int,
        target_value: float,
        target_velocity: float = 0.0,
    ) -> None:
        """Set a joint to a specific position and velocity."""

    # ------------------------------------------------------------------
    # Motor control
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def set_joint_motor_position(
        self,
        body_id: int,
        joint_index: int,
        target_position: float,
        force: float,
        max_velocity: float = 2.0,
    ) -> None:
        """Set position control on a joint."""

    @abc.abstractmethod
    def set_joint_motor_velocity(
        self,
        body_id: int,
        joint_index: int,
        force: float,
    ) -> None:
        """Set velocity control (friction-like) on a joint."""

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def create_gear_constraint(
        self,
        body_id: int,
        joint_a: int,
        joint_b: int,
        gear_ratio: float = -1.0,
        max_force: float = 50.0,
    ) -> int:
        """Create a gear constraint between two joints of the same body."""

    # ------------------------------------------------------------------
    # Contacts
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def get_contacts(self, body_id: int) -> List[Tuple]:
        """Return contact points for a body.

        Each contact is a tuple in PyBullet-compatible format:
        (contactFlag, bodyA, bodyB, linkA, linkB, posOnA, posOnB,
         contactNormal, contactDistance, normalForce, ...)

        The key fields used by CALVIN's Tasks class:
          c[2] -> bodyB uid
          c[4] -> linkB index
        """

    # ------------------------------------------------------------------
    # Visual
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def change_visual_color(
        self,
        body_id: int,
        link_index: int,
        rgba_color: List[float],
    ) -> None:
        """Change the visual color of a body's link."""

    # ------------------------------------------------------------------
    # Camera rendering
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def render_camera(
        self,
        width: int,
        height: int,
        view_matrix: np.ndarray,
        projection_matrix: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Render an image and return (rgb[H,W,3], depth[H,W])."""

    # ------------------------------------------------------------------
    # IK
    # ------------------------------------------------------------------

    @abc.abstractmethod
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
        """Compute inverse kinematics and return joint positions."""

    # ------------------------------------------------------------------
    # Utility helpers built on top of the abstract methods
    # ------------------------------------------------------------------

    def get_joint_index_by_name(self, body_id: int, joint_name: str) -> int:
        """Find joint index by name string."""
        for i in range(self.get_num_joints(body_id)):
            info = self.get_joint_info(body_id, i)
            if info["name"] == joint_name:
                return i
        raise ValueError(f"Joint '{joint_name}' not found on body {body_id}")

    def get_link_index_by_name(self, body_id: int, link_name: str) -> int:
        """Find link index by name string."""
        for i in range(self.get_num_joints(body_id)):
            info = self.get_joint_info(body_id, i)
            if info["link_name"] == link_name:
                return i
        raise ValueError(f"Link '{link_name}' not found on body {body_id}")

    def get_link_map(self, body_id: int) -> Dict[str, int]:
        """Return a dict mapping link names to link indices."""
        links = {}
        for i in range(self.get_num_joints(body_id)):
            info = self.get_joint_info(body_id, i)
            links[info["link_name"]] = i
        links["base_link"] = -1
        return links

    @staticmethod
    def euler_to_quat(euler: List[float]) -> np.ndarray:
        """Convert euler angles (xyz) to quaternion (xyzw)."""
        from scipy.spatial.transform import Rotation

        return Rotation.from_euler("xyz", euler).as_quat()

    @staticmethod
    def quat_to_euler(quat: List[float]) -> np.ndarray:
        """Convert quaternion (xyzw) to euler angles (xyz)."""
        from scipy.spatial.transform import Rotation

        return Rotation.from_quat(quat).as_euler("xyz")

    @staticmethod
    def compute_view_matrix(eye: List[float], target: List[float], up: List[float]) -> np.ndarray:
        """Compute a 4x4 OpenGL-style view matrix."""
        eye = np.array(eye, dtype=np.float64)
        target = np.array(target, dtype=np.float64)
        up = np.array(up, dtype=np.float64)

        forward = target - eye
        forward = forward / np.linalg.norm(forward)

        side = np.cross(forward, up)
        side = side / np.linalg.norm(side)

        up_corrected = np.cross(side, forward)

        view = np.eye(4, dtype=np.float64)
        view[0, :3] = side
        view[1, :3] = up_corrected
        view[2, :3] = -forward
        view[0, 3] = -np.dot(side, eye)
        view[1, 3] = -np.dot(up_corrected, eye)
        view[2, 3] = np.dot(forward, eye)
        return view

    @staticmethod
    def compute_projection_matrix_fov(
        fov: float, aspect: float, near: float, far: float
    ) -> np.ndarray:
        """Compute a perspective projection matrix from FOV (degrees)."""
        fov_rad = np.deg2rad(fov)
        f = 1.0 / np.tan(fov_rad / 2.0)
        proj = np.zeros((4, 4), dtype=np.float64)
        proj[0, 0] = f / aspect
        proj[1, 1] = f
        proj[2, 2] = (far + near) / (near - far)
        proj[2, 3] = (2.0 * far * near) / (near - far)
        proj[3, 2] = -1.0
        return proj
