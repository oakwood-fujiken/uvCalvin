"""Action adapter — converts CALVIN action formats to MetaSim handler commands.

Handles both relative (7D) and absolute (3-tuple) action modes, including
the relative-to-absolute accumulation logic from CALVIN's Robot class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from calvin_metasim.cfg.scenario_cfg import AgentCfg
    from calvin_metasim.env.handler import SimHandler


class ActionAdapter:
    """Translates CALVIN actions into joint-level commands via the handler's IK.

    CALVIN actions:
        Relative (7D): [delta_pos(3), delta_orn_euler(3), gripper(1)]
            Scaled by max_rel_pos and max_rel_orn, accumulated onto target_pos/target_orn.
        Absolute (tuple of 3): (target_pos[3], target_orn_euler_or_quat[3or4], gripper_int)
    """

    def __init__(self, agent_cfg: AgentCfg, handler: SimHandler, robot_id: int):
        self.cfg = agent_cfg
        self.handler = handler
        self.robot_id = robot_id

        # Target pose accumulators for relative mode
        self.target_pos: Optional[np.ndarray] = None
        self.target_orn: Optional[np.ndarray] = None

        # IK configuration
        num_dof = handler.get_num_joints(robot_id)
        self.ll = [-7.0] * num_dof
        self.ul = [7.0] * num_dof
        self.jr = [7.0] * num_dof
        self.rp = list(agent_cfg.initial_joint_positions) + [agent_cfg.gripper_joint_limits[1]] * 2

    def reset(self, tcp_pos: np.ndarray, tcp_orn_euler: np.ndarray):
        """Reset target pose accumulators (called on env reset)."""
        self.target_pos = np.array(tcp_pos, dtype=np.float64)
        self.target_orn = np.array(tcp_orn_euler, dtype=np.float64)

    def apply_action(self, action: Union[np.ndarray, tuple]) -> int:
        """Convert a CALVIN action and apply it to the robot.

        Args:
            action: Either a 7D numpy array (relative) or a tuple of 3 (absolute).

        Returns:
            gripper_action: 1 (open) or -1 (close)
        """
        if not isinstance(action, tuple) or (isinstance(action, tuple) and len(action) != 3):
            # Relative action: 7D array
            action = np.asarray(action, dtype=np.float64)
            target_ee_pos, target_ee_orn, gripper_action = self._relative_to_absolute(action)
        else:
            # Absolute action: (pos, orn, gripper)
            target_ee_pos = np.array(action[0], dtype=np.float64)
            target_ee_orn = np.array(action[1], dtype=np.float64)
            gripper_action = action[2]

        assert len(target_ee_pos) == 3
        assert len(target_ee_orn) in (3, 4)

        # Convert euler to quaternion if needed
        if len(target_ee_orn) == 3:
            target_ee_orn_quat = self.handler.euler_to_quat(target_ee_orn.tolist())
        else:
            target_ee_orn_quat = target_ee_orn

        if not isinstance(gripper_action, (int, np.integer)):
            if hasattr(gripper_action, "__len__") and len(gripper_action) == 1:
                gripper_action = gripper_action[0]
        gripper_action = 1 if gripper_action > 0 else -1

        # Compute IK
        joint_positions = self.handler.calculate_ik(
            self.robot_id,
            self.cfg.tcp_link_id,
            target_ee_pos.tolist(),
            target_ee_orn_quat.tolist(),
            lower_limits=self.ll,
            upper_limits=self.ul,
            joint_ranges=self.jr,
            rest_poses=self.rp,
        )

        # Apply joint position control to arm joints
        for i in range(self.cfg.end_effector_link_id):
            self.handler.set_joint_motor_position(
                self.robot_id,
                i,
                target_position=float(joint_positions[i]),
                force=self.cfg.max_joint_force,
                max_velocity=self.cfg.max_velocity,
            )

        # Apply gripper control
        self._control_gripper(gripper_action)

        return gripper_action

    def _relative_to_absolute(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Convert 7D relative action to absolute target pose.

        Mirrors Robot.relative_to_absolute() from calvin_env.
        """
        assert len(action) == 7
        rel_pos, rel_orn, gripper = np.split(action, [3, 6])
        rel_pos = rel_pos * self.cfg.max_rel_pos * self.cfg.magic_scaling_factor_pos
        rel_orn = rel_orn * self.cfg.max_rel_orn * self.cfg.magic_scaling_factor_orn

        if self.cfg.use_target_pose:
            self.target_pos += rel_pos
            self.target_orn += rel_orn
            return self.target_pos.copy(), self.target_orn.copy(), gripper[0]
        else:
            tcp_pos, tcp_orn_quat = self.handler.get_link_pose(self.robot_id, self.cfg.tcp_link_id)
            tcp_orn_euler = self.handler.quat_to_euler(tcp_orn_quat)
            abs_pos = np.array(tcp_pos) + rel_pos
            abs_orn = np.array(tcp_orn_euler) + rel_orn
            return abs_pos, abs_orn, gripper[0]

    def _control_gripper(self, gripper_action: int):
        """Apply gripper control (open/close)."""
        if gripper_action == 1:
            finger_pos = self.cfg.gripper_joint_limits[1]
            force = self.cfg.gripper_force / 100.0
        else:
            finger_pos = self.cfg.gripper_joint_limits[0]
            force = self.cfg.gripper_force

        for jid in self.cfg.gripper_joint_ids:
            self.handler.set_joint_motor_position(
                self.robot_id,
                jid,
                target_position=finger_pos,
                force=force,
                max_velocity=1.0,
            )
