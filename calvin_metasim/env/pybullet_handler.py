"""PyBullet backend handler — reference implementation using CALVIN's native simulator.

This handler wraps PyBullet calls behind the SimHandler interface, providing
a verified baseline that should produce identical results to the original
PlayTableSimEnv.
"""

from __future__ import annotations

import logging
import os
import pkgutil
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from calvin_metasim.env.handler import SimHandler

log = logging.getLogger(__name__)


class PyBulletHandler(SimHandler):
    """SimHandler implementation backed by PyBullet."""

    def __init__(self, use_egl: bool = True):
        self.use_egl = use_egl
        self.p = None  # pybullet module or BulletClient
        self.cid = -1
        self._owns_client = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def launch(self, show_gui: bool = False) -> None:
        import pybullet as p
        import pybullet_utils.bullet_client as bc

        if show_gui:
            self.p = bc.BulletClient(connection_mode=p.GUI)
            self.cid = self.p._client
        elif self.use_egl:
            self.p = p
            self.cid = p.connect(p.DIRECT)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.cid)
            p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0, physicsClientId=self.cid)
            p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0, physicsClientId=self.cid)
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0, physicsClientId=self.cid)
            egl = pkgutil.get_loader("eglRenderer")
            if egl:
                plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
            else:
                plugin = p.loadPlugin("eglRendererPlugin")
            if plugin < 0:
                log.warning("EGL plugin failed to load, falling back to CPU rendering.")
            os.environ["PYOPENGL_PLATFORM"] = "egl"
        else:
            self.p = bc.BulletClient(connection_mode=p.DIRECT)
            self.cid = self.p._client

        self._owns_client = True
        self.p.resetSimulation(physicsClientId=self.cid)
        self.p.setPhysicsEngineParameter(deterministicOverlappingPairs=1, physicsClientId=self.cid)
        self.p.configureDebugVisualizer(self.p.COV_ENABLE_GUI, 0, physicsClientId=self.cid)
        log.info(f"PyBullet connected with client id: {self.cid}")

    def close(self) -> None:
        if self._owns_client and self.cid >= 0 and self.p is not None:
            try:
                self.p.disconnect(physicsClientId=self.cid)
            except Exception:
                pass
            self.cid = -1

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
        # Set search path to parent directory of the URDF
        from pathlib import Path

        parent_dir = str(Path(urdf_path).parent.parent)
        self.p.setAdditionalSearchPath(parent_dir, physicsClientId=self.cid)
        return self.p.loadURDF(
            urdf_path,
            base_position,
            base_orientation_quat,
            useFixedBase=use_fixed_base,
            globalScaling=global_scaling,
            physicsClientId=self.cid,
        )

    def create_visual_cylinder(
        self,
        radius: float,
        length: float,
        position: List[float],
        rgba_color: List[float],
    ) -> int:
        shape = self.p.createVisualShape(
            shapeType=self.p.GEOM_CYLINDER,
            rgbaColor=rgba_color,
            radius=radius,
            length=length,
            visualFramePosition=position,
            physicsClientId=self.cid,
        )
        return self.p.createMultiBody(baseVisualShapeIndex=shape, physicsClientId=self.cid)

    # ------------------------------------------------------------------
    # Physics stepping
    # ------------------------------------------------------------------

    def step_physics(self, n_steps: int = 1) -> None:
        for _ in range(n_steps):
            self.p.stepSimulation(physicsClientId=self.cid)

    def set_gravity(self, gravity: Tuple[float, float, float]) -> None:
        self.p.setGravity(*gravity, physicsClientId=self.cid)

    def set_timestep(self, timestep: float) -> None:
        self.p.setTimeStep(timestep, physicsClientId=self.cid)

    # ------------------------------------------------------------------
    # Body / link / joint queries
    # ------------------------------------------------------------------

    def get_body_pose(self, body_id: int) -> Tuple[np.ndarray, np.ndarray]:
        pos, orn = self.p.getBasePositionAndOrientation(body_id, physicsClientId=self.cid)
        return np.array(pos), np.array(orn)

    def get_body_velocity(self, body_id: int) -> Tuple[np.ndarray, np.ndarray]:
        lin, ang = self.p.getBaseVelocity(body_id, physicsClientId=self.cid)
        return np.array(lin), np.array(ang)

    def get_link_pose(self, body_id: int, link_index: int) -> Tuple[np.ndarray, np.ndarray]:
        state = self.p.getLinkState(body_id, link_index, physicsClientId=self.cid)
        return np.array(state[0]), np.array(state[1])

    def get_num_joints(self, body_id: int) -> int:
        return self.p.getNumJoints(body_id, physicsClientId=self.cid)

    def get_joint_info(self, body_id: int, joint_index: int) -> Dict[str, Any]:
        info = self.p.getJointInfo(body_id, joint_index, physicsClientId=self.cid)
        return {
            "index": info[0],
            "name": info[1].decode("utf-8"),
            "type": info[2],
            "lower_limit": info[8],
            "upper_limit": info[9],
            "max_force": info[10],
            "max_velocity": info[11],
            "link_name": info[12].decode("utf-8"),
        }

    def get_joint_state(self, body_id: int, joint_index: int) -> float:
        return float(self.p.getJointState(body_id, joint_index, physicsClientId=self.cid)[0])

    # ------------------------------------------------------------------
    # Body / joint state setting
    # ------------------------------------------------------------------

    def reset_body_pose(
        self,
        body_id: int,
        position: List[float],
        orientation_quat: List[float],
    ) -> None:
        self.p.resetBasePositionAndOrientation(body_id, position, orientation_quat, physicsClientId=self.cid)

    def reset_joint_state(
        self,
        body_id: int,
        joint_index: int,
        target_value: float,
        target_velocity: float = 0.0,
    ) -> None:
        self.p.resetJointState(
            body_id,
            joint_index,
            targetValue=target_value,
            targetVelocity=target_velocity,
            physicsClientId=self.cid,
        )

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
        self.p.setJointMotorControl2(
            bodyIndex=body_id,
            jointIndex=joint_index,
            controlMode=self.p.POSITION_CONTROL,
            targetPosition=target_position,
            force=force,
            maxVelocity=max_velocity,
            physicsClientId=self.cid,
        )

    def set_joint_motor_velocity(
        self,
        body_id: int,
        joint_index: int,
        force: float,
    ) -> None:
        self.p.setJointMotorControl2(
            bodyIndex=body_id,
            jointIndex=joint_index,
            controlMode=self.p.VELOCITY_CONTROL,
            force=force,
            physicsClientId=self.cid,
        )

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
        c = self.p.createConstraint(
            body_id,
            joint_a,
            body_id,
            joint_b,
            jointType=self.p.JOINT_GEAR,
            jointAxis=[1, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
            physicsClientId=self.cid,
        )
        self.p.changeConstraint(c, gearRatio=gear_ratio, erp=0.1, maxForce=max_force, physicsClientId=self.cid)
        return c

    # ------------------------------------------------------------------
    # Contacts
    # ------------------------------------------------------------------

    def get_contacts(self, body_id: int) -> List[Tuple]:
        return list(self.p.getContactPoints(bodyA=body_id, physicsClientId=self.cid))

    # ------------------------------------------------------------------
    # Visual
    # ------------------------------------------------------------------

    def change_visual_color(
        self,
        body_id: int,
        link_index: int,
        rgba_color: List[float],
    ) -> None:
        self.p.changeVisualShape(body_id, link_index, rgbaColor=rgba_color, physicsClientId=self.cid)

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
        # PyBullet expects flat 16-element lists
        view_flat = view_matrix.flatten().tolist()
        proj_flat = projection_matrix.flatten().tolist()
        image = self.p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_flat,
            projectionMatrix=proj_flat,
            physicsClientId=self.cid,
        )
        w, h, rgb_pixels, depth_pixels, _ = image
        rgb = np.reshape(rgb_pixels, (h, w, 4))[:, :, :3]
        depth_buffer = np.reshape(depth_pixels, (h, w))
        return rgb, depth_buffer

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
        kwargs = {
            "bodyUniqueId": body_id,
            "endEffectorLinkIndex": end_effector_link,
            "targetPosition": target_position,
            "targetOrientation": target_orientation_quat,
            "physicsClientId": self.cid,
        }
        if lower_limits is not None:
            kwargs["lowerLimits"] = lower_limits
            kwargs["upperLimits"] = upper_limits
            kwargs["jointRanges"] = joint_ranges
            kwargs["restPoses"] = rest_poses
        result = self.p.calculateInverseKinematics(**kwargs)
        return np.array(result)
