"""Configuration dataclasses mapping CALVIN Hydra configs to MetaSim ScenarioCfg format."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from omegaconf import DictConfig, OmegaConf


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


@dataclass
class AgentCfg:
    """Franka Panda robot configuration (from panda.yaml + scene.yaml)."""

    urdf_path: str = "franka_panda/panda.urdf"
    base_position: List[float] = field(default_factory=lambda: [-0.34, -0.46, 0.24])
    base_orientation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    initial_joint_positions: List[float] = field(
        default_factory=lambda: [-1.21779206, 1.03987646, 2.11978261, -2.34205014, -0.87015947, 1.64119353, 0.55344866]
    )
    arm_joint_ids: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6])
    gripper_joint_ids: List[int] = field(default_factory=lambda: [9, 10])
    gripper_joint_limits: Tuple[float, float] = (0.0, 0.04)
    tcp_link_id: int = 13
    end_effector_link_id: int = 7
    gripper_cam_link: int = 12
    max_joint_force: float = 200.0
    gripper_force: float = 200.0
    max_velocity: float = 2.0
    lower_joint_limits: List[float] = field(
        default_factory=lambda: [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
    )
    upper_joint_limits: List[float] = field(
        default_factory=lambda: [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
    )
    use_nullspace: bool = True
    use_ik_fast: bool = False
    max_rel_pos: float = 0.02
    max_rel_orn: float = 0.05
    magic_scaling_factor_pos: float = 1.0
    magic_scaling_factor_orn: float = 1.0
    use_target_pose: bool = True
    euler_obs: bool = True


@dataclass
class CameraCfg:
    """Camera configuration (static or gripper-attached)."""

    name: str = "static"
    fov: float = 10.0
    aspect: float = 1.0
    nearval: float = 0.01
    farval: float = 10.0
    width: int = 200
    height: int = 200
    camera_type: str = "static"  # "static" or "attached"
    # Static camera fields
    look_at: Optional[List[float]] = None
    look_from: Optional[List[float]] = None
    up_vector: Optional[List[float]] = None
    # Attached camera fields
    attached_link_name: Optional[str] = None


@dataclass
class FixedObjectCfg:
    """Configuration for a fixed (non-movable) object like the table."""

    name: str = ""
    urdf_file: str = ""
    initial_pos: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    initial_orn: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    joints: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class MovableObjectCfg:
    """Configuration for a movable object (e.g. blocks)."""

    name: str = ""
    urdf_file: str = ""
    initial_pos: Any = "any"  # List[float] or "any" for random
    initial_orn: Any = "any"  # List[float] or "any" for random


@dataclass
class DoorCfg:
    """Door/drawer/slider joint configuration."""

    name: str = ""
    initial_state: float = 0.0
    parent_object: str = ""  # name of the fixed object this belongs to


@dataclass
class ButtonCfg:
    """Button configuration."""

    name: str = ""
    initial_state: float = 0.0
    effect: str = ""  # name of the light this button controls
    parent_object: str = ""


@dataclass
class SwitchCfg:
    """Switch configuration."""

    name: str = ""
    initial_state: float = 0.0
    effect: str = ""
    parent_object: str = ""


@dataclass
class LightCfg:
    """Light configuration."""

    name: str = ""
    link: str = ""
    color_on: List[float] = field(default_factory=lambda: [1.0, 1.0, 0.0, 1.0])
    color_off: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    parent_object: str = ""


# ---------------------------------------------------------------------------
# Top-level scenario
# ---------------------------------------------------------------------------


@dataclass
class CalvinScenarioCfg:
    """Complete scenario configuration for CALVIN in MetaSim format.

    Translates CALVIN's Hydra/OmegaConf scene configs into a unified dataclass
    that MetaSim handler backends can consume.
    """

    # Physics
    timestep: float = 1.0 / 240.0
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.8)
    global_scaling: float = 0.8

    # Agent
    agent: AgentCfg = field(default_factory=AgentCfg)

    # Objects
    fixed_objects: List[FixedObjectCfg] = field(default_factory=list)
    movable_objects: List[MovableObjectCfg] = field(default_factory=list)

    # Interactive elements
    doors: List[DoorCfg] = field(default_factory=list)
    buttons: List[ButtonCfg] = field(default_factory=list)
    switches: List[SwitchCfg] = field(default_factory=list)
    lights: List[LightCfg] = field(default_factory=list)

    # Cameras
    cameras: List[CameraCfg] = field(default_factory=list)

    # Surfaces for random object placement: name -> [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    surfaces: Dict[str, List[List[float]]] = field(default_factory=dict)

    # Tasks config path (YAML with task definitions)
    tasks_cfg_path: Optional[str] = None

    # Data path (root directory for URDF assets)
    data_path: str = ""

    # Control
    control_freq: int = 30
    action_repeat: int = 8


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _resolve_data_path(cfg: DictConfig) -> Path:
    """Resolve the data_path from config, falling back to calvin_env/data/."""
    data_path = cfg.get("scene", {}).get("data_path", cfg.get("data_path", ""))
    if data_path and os.path.isabs(str(data_path)):
        return Path(str(data_path))
    # Default: relative to calvin_env package
    calvin_env_root = Path(__file__).parents[2] / "calvin_env"
    if data_path:
        return calvin_env_root / str(data_path)
    return calvin_env_root / "data"


def _parse_agent(cfg: DictConfig) -> AgentCfg:
    """Parse robot config from merged Hydra config."""
    robot = cfg.get("robot", {})
    scene = cfg.get("scene", {})
    return AgentCfg(
        urdf_path=str(robot.get("filename", "franka_panda/panda.urdf")),
        base_position=list(scene.get("robot_base_position", [-0.34, -0.46, 0.24])),
        base_orientation=list(scene.get("robot_base_orientation", [0.0, 0.0, 0.0])),
        initial_joint_positions=list(
            scene.get(
                "robot_initial_joint_positions",
                [-1.21779206, 1.03987646, 2.11978261, -2.34205014, -0.87015947, 1.64119353, 0.55344866],
            )
        ),
        arm_joint_ids=list(robot.get("arm_joint_ids", [0, 1, 2, 3, 4, 5, 6])),
        gripper_joint_ids=list(robot.get("gripper_joint_ids", [9, 10])),
        gripper_joint_limits=tuple(robot.get("gripper_joint_limits", [0.0, 0.04])),
        tcp_link_id=int(robot.get("tcp_link_id", 13)),
        end_effector_link_id=int(robot.get("end_effector_link_id", 7)),
        gripper_cam_link=int(robot.get("gripper_cam_link", 12)),
        max_joint_force=float(robot.get("max_joint_force", 200.0)),
        gripper_force=float(robot.get("gripper_force", 200.0)),
        max_velocity=float(robot.get("max_velocity", 2.0)),
        lower_joint_limits=list(
            robot.get("lower_joint_limits", [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        ),
        upper_joint_limits=list(
            robot.get("upper_joint_limits", [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        ),
        use_nullspace=bool(robot.get("use_nullspace", True)),
        use_ik_fast=bool(robot.get("use_ik_fast", False)),
        max_rel_pos=float(robot.get("max_rel_pos", 0.02)),
        max_rel_orn=float(robot.get("max_rel_orn", 0.05)),
        magic_scaling_factor_pos=float(robot.get("magic_scaling_factor_pos", 1.0)),
        magic_scaling_factor_orn=float(robot.get("magic_scaling_factor_orn", 1.0)),
        use_target_pose=bool(robot.get("use_target_pose", True)),
        euler_obs=bool(robot.get("euler_obs", True)),
    )


def _parse_cameras(cfg: DictConfig) -> List[CameraCfg]:
    """Parse camera configs from merged Hydra config."""
    cameras_cfg = cfg.get("cameras", {})
    result = []
    for cam_name, cam in cameras_cfg.items():
        target = str(cam.get("_target_", ""))
        if "StaticCamera" in target:
            result.append(
                CameraCfg(
                    name=str(cam.get("name", cam_name)),
                    fov=float(cam.get("fov", 10.0)),
                    aspect=float(cam.get("aspect", 1.0)),
                    nearval=float(cam.get("nearval", 0.01)),
                    farval=float(cam.get("farval", 10.0)),
                    width=int(cam.get("width", 200)),
                    height=int(cam.get("height", 200)),
                    camera_type="static",
                    look_at=list(cam.get("look_at", [0.0, 0.0, 0.0])),
                    look_from=list(cam.get("look_from", [0.0, 0.0, 1.0])),
                    up_vector=list(cam.get("up_vector", [0.0, 0.0, 1.0])),
                )
            )
        elif "GripperCamera" in target:
            result.append(
                CameraCfg(
                    name=str(cam.get("name", cam_name)),
                    fov=float(cam.get("fov", 75.0)),
                    aspect=float(cam.get("aspect", 1.0)),
                    nearval=float(cam.get("nearval", 0.01)),
                    farval=float(cam.get("farval", 2.0)),
                    width=int(cam.get("width", 84)),
                    height=int(cam.get("height", 84)),
                    camera_type="attached",
                    attached_link_name="gripper_cam",
                )
            )
    return result


def _parse_scene(cfg: DictConfig) -> Tuple[
    List[FixedObjectCfg],
    List[MovableObjectCfg],
    List[DoorCfg],
    List[ButtonCfg],
    List[SwitchCfg],
    List[LightCfg],
    Dict[str, List[List[float]]],
]:
    """Parse scene config into object lists."""
    scene = cfg.get("scene", {})
    objects_cfg = OmegaConf.to_container(scene.get("objects", {}), resolve=True)
    surfaces_cfg = OmegaConf.to_container(scene.get("surfaces", {}), resolve=True)

    fixed_objects = []
    movable_objects = []
    doors = []
    buttons = []
    switches = []
    lights = []

    for name, obj_cfg in objects_cfg.get("fixed_objects", {}).items():
        fixed_objects.append(
            FixedObjectCfg(
                name=name,
                urdf_file=obj_cfg["file"],
                initial_pos=obj_cfg.get("initial_pos", [0, 0, 0]),
                initial_orn=obj_cfg.get("initial_orn", [0, 0, 0]),
                joints=obj_cfg.get("joints", {}),
            )
        )
        for joint_name, joint_cfg in obj_cfg.get("joints", {}).items():
            doors.append(
                DoorCfg(
                    name=joint_name,
                    initial_state=float(joint_cfg.get("initial_state", 0)),
                    parent_object=name,
                )
            )
        for button_name, button_cfg in obj_cfg.get("buttons", {}).items():
            buttons.append(
                ButtonCfg(
                    name=button_name,
                    initial_state=float(button_cfg.get("initial_state", 0)),
                    effect=str(button_cfg.get("effect", "")),
                    parent_object=name,
                )
            )
        for switch_name, switch_cfg in obj_cfg.get("switches", {}).items():
            switches.append(
                SwitchCfg(
                    name=switch_name,
                    initial_state=float(switch_cfg.get("initial_state", 0)),
                    effect=str(switch_cfg.get("effect", "")),
                    parent_object=name,
                )
            )
        for light_name, light_cfg in obj_cfg.get("lights", {}).items():
            lights.append(
                LightCfg(
                    name=light_name,
                    link=str(light_cfg.get("link", "")),
                    color_on=list(light_cfg.get("color", [1, 1, 0, 1])),
                    parent_object=name,
                )
            )

    for name, obj_cfg in objects_cfg.get("movable_objects", {}).items():
        movable_objects.append(
            MovableObjectCfg(
                name=name,
                urdf_file=obj_cfg["file"],
                initial_pos=obj_cfg.get("initial_pos", "any"),
                initial_orn=obj_cfg.get("initial_orn", "any"),
            )
        )

    return fixed_objects, movable_objects, doors, buttons, switches, lights, surfaces_cfg


def build_scenario_from_hydra_cfg(
    cfg: DictConfig,
    obs_space: Optional[Dict] = None,
) -> CalvinScenarioCfg:
    """Build a CalvinScenarioCfg from a merged Hydra config (as loaded from .hydra/merged_config.yaml).

    Args:
        cfg: Merged OmegaConf config.
        obs_space: Optional observation space dict to filter cameras.

    Returns:
        CalvinScenarioCfg ready for MetaSim handler.
    """
    data_path = _resolve_data_path(cfg)
    agent = _parse_agent(cfg)
    cameras = _parse_cameras(cfg)
    fixed_objects, movable_objects, doors, buttons, switches, lights, surfaces = _parse_scene(cfg)

    env_cfg = cfg.get("env", {})
    bullet_time_step = float(env_cfg.get("bullet_time_step", 240.0))
    control_freq = int(env_cfg.get("control_freq", 30))
    action_repeat = int(bullet_time_step // control_freq)

    scene_cfg = cfg.get("scene", {})
    global_scaling = float(scene_cfg.get("global_scaling", 0.8))

    return CalvinScenarioCfg(
        timestep=1.0 / bullet_time_step,
        global_scaling=global_scaling,
        agent=agent,
        fixed_objects=fixed_objects,
        movable_objects=movable_objects,
        doors=doors,
        buttons=buttons,
        switches=switches,
        lights=lights,
        cameras=cameras,
        surfaces=surfaces,
        data_path=str(data_path),
        control_freq=control_freq,
        action_repeat=action_repeat,
    )
