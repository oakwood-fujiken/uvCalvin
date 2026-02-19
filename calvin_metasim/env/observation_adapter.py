"""Observation adapter — translates MetaSim state into CALVIN's exact observation format.

CALVIN models depend on specific observation dictionary structures with exact keys
and shapes. This adapter queries the SimHandler and assembles observations that are
byte-for-byte compatible with the original PlayTableSimEnv.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from calvin_metasim.cfg.scenario_cfg import CalvinScenarioCfg, CameraCfg
    from calvin_metasim.env.handler import SimHandler
    from calvin_metasim.scene.interactive_objects import InteractiveObjectManager


class ObservationAdapter:
    """Assembles CALVIN-format observations from SimHandler state queries.

    Produces:
        robot_obs: np.ndarray(15,)
            [tcp_pos(3), tcp_orn_euler(3), gripper_width(1), arm_joints(7), gripper_action(1)]
        scene_obs: np.ndarray(24,)
            [slide(1), drawer(1), button(1), switch(1), lightbulb(1), led(1),
             block_red_pos(3)+orn(3), block_blue_pos(3)+orn(3), block_pink_pos(3)+orn(3)]
        rgb_obs: dict of name -> np.ndarray(H,W,3)
        depth_obs: dict of name -> np.ndarray(H,W)
        info: dict compatible with Tasks.get_task_info()
    """

    def __init__(
        self,
        scenario_cfg: CalvinScenarioCfg,
        handler: SimHandler,
        interactive_mgr: InteractiveObjectManager,
    ):
        self.cfg = scenario_cfg
        self.handler = handler
        self.interactive_mgr = interactive_mgr

        # These will be set after scene loading
        self.robot_id: Optional[int] = None
        self.fixed_object_ids: Dict[str, int] = {}  # name -> body_id
        self.movable_object_ids: Dict[str, int] = {}  # name -> body_id (ordered)
        self.link_maps: Dict[str, Dict[str, int]] = {}  # fixed_obj_name -> {link_name: link_id}

        # Camera state (precomputed matrices)
        self._camera_states: List[Dict] = []

        # Track gripper action
        self.gripper_action: int = 1

    def set_ids(
        self,
        robot_id: int,
        fixed_object_ids: Dict[str, int],
        movable_object_ids: Dict[str, int],
    ):
        """Called after scene loading to register body IDs."""
        self.robot_id = robot_id
        self.fixed_object_ids = fixed_object_ids
        self.movable_object_ids = movable_object_ids

        # Build link maps for fixed objects
        for name, body_id in fixed_object_ids.items():
            n_joints = self.handler.get_num_joints(body_id)
            if n_joints > 0:
                self.link_maps[name] = self.handler.get_link_map(body_id)

    def setup_cameras(self):
        """Pre-compute camera view/projection matrices."""
        self._camera_states = []
        for cam_cfg in self.cfg.cameras:
            if cam_cfg.camera_type == "static":
                view_mat = self.handler.compute_view_matrix(
                    cam_cfg.look_from, cam_cfg.look_at, cam_cfg.up_vector
                )
                proj_mat = self.handler.compute_projection_matrix_fov(
                    cam_cfg.fov, cam_cfg.aspect, cam_cfg.nearval, cam_cfg.farval
                )
                self._camera_states.append(
                    {
                        "cfg": cam_cfg,
                        "view_matrix": view_mat,
                        "projection_matrix": proj_mat,
                        "type": "static",
                    }
                )
            elif cam_cfg.camera_type == "attached":
                proj_mat = self.handler.compute_projection_matrix_fov(
                    cam_cfg.fov, cam_cfg.aspect, cam_cfg.nearval, cam_cfg.farval
                )
                self._camera_states.append(
                    {
                        "cfg": cam_cfg,
                        "projection_matrix": proj_mat,
                        "type": "attached",
                        "link_index": self.handler.get_link_index_by_name(
                            self.robot_id, cam_cfg.attached_link_name
                        ),
                    }
                )

    # ------------------------------------------------------------------
    # Main observation assembly
    # ------------------------------------------------------------------

    def get_obs(self) -> Dict:
        """Return full observation dict matching PlayTableSimEnv.get_obs()."""
        rgb_obs, depth_obs = self.get_camera_obs()
        obs = {"rgb_obs": rgb_obs, "depth_obs": depth_obs}
        obs.update(self.get_state_obs())
        return obs

    def get_state_obs(self) -> Dict:
        """Return robot_obs and scene_obs."""
        robot_obs = self._build_robot_obs()
        scene_obs = self._build_scene_obs()
        return {"robot_obs": robot_obs, "scene_obs": scene_obs}

    def get_camera_obs(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Render all cameras."""
        rgb_obs = {}
        depth_obs = {}
        for cam_state in self._camera_states:
            cfg = cam_state["cfg"]
            if cam_state["type"] == "static":
                view_mat = cam_state["view_matrix"]
            else:
                view_mat = self._compute_gripper_view_matrix(cam_state["link_index"])
            rgb, depth = self.handler.render_camera(
                cfg.width, cfg.height, view_mat, cam_state["projection_matrix"]
            )
            rgb_obs[f"rgb_{cfg.name}"] = rgb
            depth_obs[f"depth_{cfg.name}"] = depth
        return rgb_obs, depth_obs

    def get_info(self, use_scene_info: bool = True) -> Dict:
        """Return info dict compatible with Tasks.get_task_info().

        Structure:
            robot_info:
                tcp_pos, tcp_orn, gripper_opening_width, arm_joint_states,
                gripper_action, uid, contacts
            scene_info:
                fixed_objects: {name: {uid, links, contacts}}
                movable_objects: {name: {uid, current_pos, current_orn, current_lin_vel, current_ang_vel, contacts}}
                doors: {name: {current_state}}
                buttons: {name: {joint_state, logical_state}}
                switches: {name: {joint_state, logical_state}}
                lights: {name: {logical_state}}
        """
        robot_obs, robot_info = self._build_robot_info()
        info = {"robot_info": robot_info}
        if use_scene_info:
            info["scene_info"] = self._build_scene_info()
        return info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_robot_obs(self) -> np.ndarray:
        """Build 15D robot observation (euler mode)."""
        agent = self.cfg.agent
        tcp_pos, tcp_orn_quat = self.handler.get_link_pose(self.robot_id, agent.tcp_link_id)
        tcp_orn_euler = self.handler.quat_to_euler(tcp_orn_quat)

        gripper_width = sum(
            self.handler.get_joint_state(self.robot_id, jid) for jid in agent.gripper_joint_ids
        )

        arm_joints = [self.handler.get_joint_state(self.robot_id, jid) for jid in agent.arm_joint_ids]

        return np.array(
            [*tcp_pos, *tcp_orn_euler, gripper_width, *arm_joints, self.gripper_action],
            dtype=np.float64,
        )

    def _build_scene_obs(self) -> np.ndarray:
        """Build 24D scene observation (euler mode)."""
        # Interactive object states (doors, buttons, switches, lights)
        interactive_states = self.interactive_mgr.get_obs()

        # Movable object poses
        object_poses = []
        for name in self.movable_object_ids:
            body_id = self.movable_object_ids[name]
            pos, orn_quat = self.handler.get_body_pose(body_id)
            orn_euler = self.handler.quat_to_euler(orn_quat)
            object_poses.extend([*pos, *orn_euler])

        return np.concatenate([interactive_states, object_poses])

    def _build_robot_info(self) -> Tuple[np.ndarray, Dict]:
        """Build robot observation and info dict."""
        agent = self.cfg.agent
        tcp_pos, tcp_orn_quat = self.handler.get_link_pose(self.robot_id, agent.tcp_link_id)
        tcp_orn_euler = self.handler.quat_to_euler(tcp_orn_quat)

        gripper_width = sum(
            self.handler.get_joint_state(self.robot_id, jid) for jid in agent.gripper_joint_ids
        )
        arm_joints = [self.handler.get_joint_state(self.robot_id, jid) for jid in agent.arm_joint_ids]

        robot_obs = np.array(
            [*tcp_pos, *tcp_orn_euler, gripper_width, *arm_joints, self.gripper_action],
            dtype=np.float64,
        )

        robot_info = {
            "tcp_pos": tuple(tcp_pos),
            "tcp_orn": tuple(tcp_orn_euler),
            "gripper_opening_width": gripper_width,
            "arm_joint_states": arm_joints,
            "gripper_action": self.gripper_action,
            "uid": self.robot_id,
            "contacts": self.handler.get_contacts(self.robot_id),
        }
        return robot_obs, robot_info

    def _build_scene_info(self) -> Dict:
        """Build scene info dict compatible with Tasks class."""
        info = {
            "fixed_objects": {},
            "movable_objects": {},
        }

        for name, body_id in self.fixed_object_ids.items():
            obj_info = {"uid": body_id, "contacts": self.handler.get_contacts(body_id)}
            if name in self.link_maps:
                obj_info["links"] = self.link_maps[name]
            info["fixed_objects"][name] = obj_info

        for name, body_id in self.movable_object_ids.items():
            pos, orn = self.handler.get_body_pose(body_id)
            lin_vel, ang_vel = self.handler.get_body_velocity(body_id)
            info["movable_objects"][name] = {
                "uid": body_id,
                "current_pos": tuple(pos),
                "current_orn": tuple(orn),  # quaternion for Tasks class
                "current_lin_vel": tuple(lin_vel),
                "current_ang_vel": tuple(ang_vel),
                "contacts": self.handler.get_contacts(body_id),
            }

        # Merge interactive object info
        interactive_info = self.interactive_mgr.get_info()
        info.update(interactive_info)
        return info

    def _compute_gripper_view_matrix(self, link_index: int) -> np.ndarray:
        """Compute view matrix for gripper-attached camera."""
        cam_pos, cam_orn_quat = self.handler.get_link_pose(self.robot_id, link_index)
        # Convert quaternion to rotation matrix
        from scipy.spatial.transform import Rotation

        cam_rot = Rotation.from_quat(cam_orn_quat).as_matrix()
        cam_rot_y = cam_rot[:, 1]
        cam_rot_z = cam_rot[:, 2]
        # Camera looks along Y axis, up is -Z (matching CALVIN's gripper camera convention)
        target = np.array(cam_pos) + cam_rot_y
        up = -cam_rot_z
        return self.handler.compute_view_matrix(cam_pos.tolist(), target.tolist(), up.tolist())
