"""Scene builder — constructs the CALVIN scene via the SimHandler interface.

Loads all URDFs, creates interactive objects, and handles random object placement.
"""

from __future__ import annotations

import itertools
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from calvin_metasim.scene.interactive_objects import (
    InteractiveObjectManager,
    SimButton,
    SimDoor,
    SimLight,
    SimSwitch,
)

if TYPE_CHECKING:
    from calvin_metasim.cfg.scenario_cfg import CalvinScenarioCfg
    from calvin_metasim.env.handler import SimHandler

log = logging.getLogger(__name__)


class CalvinSceneBuilder:
    """Builds the complete CALVIN scene from a CalvinScenarioCfg."""

    def __init__(self, scenario_cfg: CalvinScenarioCfg, handler: SimHandler, np_random: np.random.RandomState):
        self.cfg = scenario_cfg
        self.handler = handler
        self.np_random = np_random
        self.data_path = Path(scenario_cfg.data_path)

        # Populated during build
        self.robot_id: Optional[int] = None
        self.fixed_object_ids: Dict[str, int] = {}
        self.movable_object_ids: Dict[str, int] = {}  # ordered dict (insertion order)
        self.interactive_mgr = InteractiveObjectManager()

    def build(self) -> Tuple[int, Dict[str, int], Dict[str, int], InteractiveObjectManager]:
        """Build the full scene and return IDs.

        Returns:
            (robot_id, fixed_object_ids, movable_object_ids, interactive_mgr)
        """
        self.handler.set_gravity(self.cfg.gravity)

        # Load ground plane
        plane_path = self.data_path / "plane" / "plane.urdf"
        if plane_path.exists():
            self.handler.load_urdf(
                str(plane_path),
                base_position=[0, 0, 0],
                base_orientation_quat=[0, 0, 0, 1],
                use_fixed_base=False,
                global_scaling=1.0,
            )

        # Load robot
        self.robot_id = self._load_robot()

        # Load fixed objects (table, etc.)
        for obj_cfg in self.cfg.fixed_objects:
            body_id = self._load_fixed_object(obj_cfg)
            self.fixed_object_ids[obj_cfg.name] = body_id

        # Create interactive objects (doors, buttons, switches, lights)
        self._create_interactive_objects()

        # Load movable objects (blocks)
        for obj_cfg in self.cfg.movable_objects:
            body_id = self._load_movable_object(obj_cfg)
            self.movable_object_ids[obj_cfg.name] = body_id

        return self.robot_id, self.fixed_object_ids, self.movable_object_ids, self.interactive_mgr

    def _load_robot(self) -> int:
        """Load the Franka Panda robot."""
        agent = self.cfg.agent
        urdf_path = str(self.data_path / agent.urdf_path)
        base_orn_quat = self.handler.euler_to_quat(agent.base_orientation).tolist()

        robot_id = self.handler.load_urdf(
            urdf_path,
            base_position=agent.base_position,
            base_orientation_quat=base_orn_quat,
            use_fixed_base=True,
            global_scaling=1.0,
        )

        # Create cosmetic base cylinder
        pos = list(agent.base_position)
        pos[2] /= 2
        angle = agent.base_orientation[2]  # yaw
        pos[0] -= np.cos(angle) * 0.05
        pos[1] -= np.sin(angle) * 0.05
        self.handler.create_visual_cylinder(
            radius=0.13,
            length=agent.base_position[2],
            position=pos,
            rgba_color=[1, 1, 1, 1],
        )

        # Create gripper gear constraint
        self.handler.create_gear_constraint(
            robot_id,
            agent.gripper_joint_ids[0],
            agent.gripper_joint_ids[1],
            gear_ratio=-1.0,
            max_force=50.0,
        )

        # Reset to initial configuration
        self._reset_robot(robot_id)

        return robot_id

    def _reset_robot(self, robot_id: int, robot_obs: Optional[np.ndarray] = None):
        """Reset robot joints to initial or specified state."""
        agent = self.cfg.agent
        if robot_obs is None:
            joint_states = agent.initial_joint_positions
            gripper_state = agent.gripper_joint_limits[1]
        else:
            # Parse robot_obs: [tcp_pos(3), tcp_orn(3), gripper_width(1), joints(7), gripper_action(1)]
            gripper_state = robot_obs[6] / 2.0
            joint_states = robot_obs[7:14]

        for i, jid in enumerate(agent.arm_joint_ids):
            self.handler.reset_joint_state(robot_id, jid, float(joint_states[i]))
            self.handler.set_joint_motor_position(
                robot_id,
                jid,
                target_position=float(joint_states[i]),
                force=agent.max_joint_force,
                max_velocity=agent.max_velocity,
            )
        for jid in agent.gripper_joint_ids:
            self.handler.reset_joint_state(robot_id, jid, gripper_state)
            self.handler.set_joint_motor_position(
                robot_id,
                jid,
                target_position=gripper_state,
                force=agent.gripper_force,
                max_velocity=1.0,
            )

    def _load_fixed_object(self, obj_cfg) -> int:
        """Load a fixed object URDF."""
        urdf_path = str(self.data_path / obj_cfg.urdf_file)
        base_orn_quat = self.handler.euler_to_quat(obj_cfg.initial_orn).tolist()
        return self.handler.load_urdf(
            urdf_path,
            base_position=obj_cfg.initial_pos,
            base_orientation_quat=base_orn_quat,
            use_fixed_base=False,
            global_scaling=self.cfg.global_scaling,
        )

    def _load_movable_object(self, obj_cfg) -> int:
        """Load a movable object URDF with random placement."""
        urdf_path = str(self.data_path / obj_cfg.urdf_file)
        initial_pos, initial_orn_quat = self._sample_initial_pose(obj_cfg)
        return self.handler.load_urdf(
            urdf_path,
            base_position=initial_pos.tolist(),
            base_orientation_quat=initial_orn_quat.tolist(),
            use_fixed_base=False,
            global_scaling=self.cfg.global_scaling,
        )

    def _sample_initial_pose(self, obj_cfg) -> Tuple[np.ndarray, np.ndarray]:
        """Sample initial pose for a movable object (mirrors MovableObject.sample_initial_pose)."""
        # Position
        if isinstance(obj_cfg.initial_pos, str):
            if obj_cfg.initial_pos == "any":
                surface_name = self.np_random.choice(list(self.cfg.surfaces.keys()))
                sampling_range = np.array(self.cfg.surfaces[surface_name])
            else:
                sampling_range = np.array(self.cfg.surfaces[obj_cfg.initial_pos])
            initial_pos = self.np_random.uniform(sampling_range[0], sampling_range[1])
        else:
            initial_pos = np.array(obj_cfg.initial_pos)

        # Orientation
        if isinstance(obj_cfg.initial_orn, str):
            if obj_cfg.initial_orn == "any":
                euler = self.np_random.uniform([0, 0, -np.pi], [0, 0, np.pi])
                initial_orn = self.handler.euler_to_quat(euler.tolist())
            else:
                raise ValueError("Only 'any' keyword supported for random orientation")
        else:
            initial_orn = self.handler.euler_to_quat(obj_cfg.initial_orn)

        return initial_pos, initial_orn

    def _create_interactive_objects(self):
        """Create all doors, buttons, switches, and lights."""
        mgr = self.interactive_mgr

        for door_cfg in self.cfg.doors:
            parent_id = self.fixed_object_ids[door_cfg.parent_object]
            mgr.doors.append(SimDoor(door_cfg.name, door_cfg.initial_state, parent_id, self.handler))

        for button_cfg in self.cfg.buttons:
            parent_id = self.fixed_object_ids[button_cfg.parent_object]
            mgr.buttons.append(
                SimButton(button_cfg.name, button_cfg.initial_state, button_cfg.effect, parent_id, self.handler)
            )

        for switch_cfg in self.cfg.switches:
            parent_id = self.fixed_object_ids[switch_cfg.parent_object]
            mgr.switches.append(
                SimSwitch(switch_cfg.name, switch_cfg.initial_state, switch_cfg.effect, parent_id, self.handler)
            )

        for light_cfg in self.cfg.lights:
            parent_id = self.fixed_object_ids[light_cfg.parent_object]
            mgr.lights.append(
                SimLight(light_cfg.name, light_cfg.link, light_cfg.color_on, parent_id, self.handler)
            )

        mgr.link_effects()

    def reset_scene(self, scene_obs: Optional[np.ndarray] = None):
        """Reset the scene to initial or specified state."""
        if scene_obs is None:
            self.interactive_mgr.reset()
            self._reset_movable_objects()
        else:
            door_info, button_info, switch_info, light_info, obj_info = self._parse_scene_obs(scene_obs)
            self.interactive_mgr.reset((door_info, button_info, switch_info, light_info))
            for obj_name, state in zip(self.movable_object_ids.keys(), obj_info):
                body_id = self.movable_object_ids[obj_name]
                pos, orn = np.split(state, [3])
                if len(orn) == 3:
                    orn = self.handler.euler_to_quat(orn.tolist())
                self.handler.reset_body_pose(body_id, pos.tolist(), orn.tolist())

    def reset_robot(self, robot_obs: Optional[np.ndarray] = None):
        """Reset the robot to initial or specified state."""
        self._reset_robot(self.robot_id, robot_obs)

    def _reset_movable_objects(self):
        """Reset movable objects with collision-free random placement."""
        max_iterations = 1000
        obj_list = list(self.movable_object_ids.items())
        obj_cfgs = {cfg.name: cfg for cfg in self.cfg.movable_objects}

        for _ in range(max_iterations):
            for name, body_id in obj_list:
                pos, orn = self._sample_initial_pose(obj_cfgs[name])
                self.handler.reset_body_pose(body_id, pos.tolist(), orn.tolist())
            self.handler.step_physics(1)

            # Check pairwise contacts
            contact = False
            for (name_a, id_a), (name_b, id_b) in itertools.combinations(obj_list, 2):
                contacts_a = self.handler.get_contacts(id_a)
                if any(c[2] == id_b for c in contacts_a):
                    contact = True
                    break
            if not contact:
                return

        log.error(f"Could not place objects in {max_iterations} iterations without contacts")

    def _parse_scene_obs(self, scene_obs: np.ndarray):
        """Parse flat scene_obs into per-component arrays (mirrors PlayTableScene.parse_scene_obs)."""
        n_doors = len(self.interactive_mgr.doors)
        n_buttons = len(self.interactive_mgr.buttons)
        n_switches = len(self.interactive_mgr.switches)
        n_lights = len(self.interactive_mgr.lights)
        n_obj = len(self.movable_object_ids)

        split_ids = np.cumsum([n_doors, n_buttons, n_switches, n_lights])
        door_info, button_info, switch_info, light_info, obj_info = np.split(scene_obs, split_ids)
        obj_info = np.split(obj_info, n_obj)
        return door_info, button_info, switch_info, light_info, obj_info
