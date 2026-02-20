"""Isaac Lab backend handler — MetaSim handler for running CALVIN on NVIDIA Isaac Lab.

Isaac Lab (formerly Orbit) is a unified GPU-accelerated robotics simulation
framework built on top of Isaac Sim. It provides a high-level API for
articulations, sensors, and environments.

Requires: omni.isaac.lab (Isaac Lab SDK)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from calvin_metasim.env.handler import SimHandler

log = logging.getLogger(__name__)


class IsaacLabHandler(SimHandler):
    """SimHandler implementation backed by NVIDIA Isaac Lab.

    Isaac Lab provides GPU-accelerated physics via PhysX and RTX rendering.
    This handler wraps the Isaac Lab articulation and sensor APIs.
    """

    def __init__(self, num_envs: int = 1, device: str = "cuda:0", headless: bool = True):
        self.num_envs = num_envs
        self.device = device
        self.headless = headless
        self._sim = None
        self._scene = None
        self._bodies: Dict[int, Any] = {}
        self._next_uid = 0
        self._timestep = 1.0 / 240.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def launch(self, show_gui: bool = False) -> None:
        try:
            from omni.isaac.lab.app import AppLauncher
        except ImportError:
            raise ImportError(
                "Isaac Lab is required. Please install: "
                "https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html"
            )

        launcher = AppLauncher(headless=self.headless and not show_gui)
        self._simulation_app = launcher.app

        import omni.isaac.lab.sim as sim_utils
        from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg

        sim_cfg = sim_utils.SimulationCfg(dt=self._timestep, device=self.device)
        self._sim = sim_utils.SimulationContext(sim_cfg)
        self._sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 0.0])
        log.info("Isaac Lab simulation context created.")

    def close(self) -> None:
        if self._simulation_app is not None:
            self._simulation_app.close()
            self._simulation_app = None
        self._sim = None

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
        import omni.isaac.lab.sim as sim_utils
        from omni.isaac.lab.assets import Articulation, ArticulationCfg

        uid = self._next_uid
        self._next_uid += 1

        prim_path = f"/World/body_{uid}"

        # Convert URDF to USD for Isaac Sim
        from omni.isaac.lab.sim.converters import UrdfConverterCfg, UrdfConverter

        converter_cfg = UrdfConverterCfg(
            asset_path=urdf_path,
            fix_base=use_fixed_base,
            force_usd_conversion=False,
        )
        converter = UrdfConverter(converter_cfg)
        usd_path = converter.usd_path

        art_cfg = ArticulationCfg(
            prim_path=prim_path,
            spawn=sim_utils.UsdFileCfg(
                usd_path=usd_path,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=use_fixed_base,
                ),
                articulation_root_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=tuple(base_position),
                rot=tuple(base_orientation_quat),
            ),
        )

        articulation = Articulation(art_cfg)
        self._bodies[uid] = {
            "articulation": articulation,
            "prim_path": prim_path,
            "num_joints": 0,  # set after scene play
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
            import omni.isaac.lab.sim as sim_utils

            prim_path = f"/World/visual_{uid}"
            sim_utils.CylinderCfg(
                radius=radius,
                height=length,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=tuple(rgba_color[:3]),
                    opacity=rgba_color[3] if len(rgba_color) > 3 else 1.0,
                ),
            ).func(prim_path, sim_utils.CylinderCfg(), translation=tuple(position))
        except Exception as e:
            log.debug(f"Visual cylinder creation skipped: {e}")

        return uid

    # ------------------------------------------------------------------
    # Physics stepping
    # ------------------------------------------------------------------

    def step_physics(self, n_steps: int = 1) -> None:
        if self._sim is None:
            return
        for _ in range(n_steps):
            self._sim.step()

    def set_gravity(self, gravity: Tuple[float, float, float]) -> None:
        if self._sim is not None:
            try:
                import omni.isaac.lab.sim as sim_utils
                physics_scene = sim_utils.PhysicsContext.instance()
                if physics_scene:
                    physics_scene.set_gravity(list(gravity))
            except Exception:
                pass

    def set_timestep(self, timestep: float) -> None:
        self._timestep = timestep

    # ------------------------------------------------------------------
    # Body / link / joint queries
    # ------------------------------------------------------------------

    def get_body_pose(self, body_id: int) -> Tuple[np.ndarray, np.ndarray]:
        body = self._bodies.get(body_id)
        if body is None or "articulation" not in body:
            return np.zeros(3), np.array([0, 0, 0, 1.0])
        art = body["articulation"]
        try:
            root_pos = art.data.root_pos_w[0].cpu().numpy()
            root_quat = art.data.root_quat_w[0].cpu().numpy()
            return root_pos, root_quat
        except Exception:
            return np.zeros(3), np.array([0, 0, 0, 1.0])

    def get_body_velocity(self, body_id: int) -> Tuple[np.ndarray, np.ndarray]:
        body = self._bodies.get(body_id)
        if body is None or "articulation" not in body:
            return np.zeros(3), np.zeros(3)
        art = body["articulation"]
        try:
            lin_vel = art.data.root_lin_vel_w[0].cpu().numpy()
            ang_vel = art.data.root_ang_vel_w[0].cpu().numpy()
            return lin_vel, ang_vel
        except Exception:
            return np.zeros(3), np.zeros(3)

    def get_link_pose(self, body_id: int, link_index: int) -> Tuple[np.ndarray, np.ndarray]:
        body = self._bodies.get(body_id)
        if body is None or "articulation" not in body:
            return np.zeros(3), np.array([0, 0, 0, 1.0])
        art = body["articulation"]
        try:
            pos = art.data.body_pos_w[0, link_index].cpu().numpy()
            quat = art.data.body_quat_w[0, link_index].cpu().numpy()
            return pos, quat
        except Exception:
            return np.zeros(3), np.array([0, 0, 0, 1.0])

    def get_num_joints(self, body_id: int) -> int:
        body = self._bodies.get(body_id)
        if body is None or "articulation" not in body:
            return 0
        try:
            return body["articulation"].num_joints
        except Exception:
            return body.get("num_joints", 0)

    def get_joint_info(self, body_id: int, joint_index: int) -> Dict[str, Any]:
        body = self._bodies.get(body_id)
        if body is None or "articulation" not in body:
            return self._default_joint_info(joint_index)
        art = body["articulation"]
        try:
            names = art.joint_names
            return {
                "index": joint_index,
                "name": names[joint_index] if joint_index < len(names) else f"joint_{joint_index}",
                "type": 0,
                "lower_limit": float(art.data.soft_joint_pos_limits[0, joint_index, 0].cpu()),
                "upper_limit": float(art.data.soft_joint_pos_limits[0, joint_index, 1].cpu()),
                "max_force": 100.0,
                "max_velocity": 2.0,
                "link_name": art.body_names[joint_index] if joint_index < len(art.body_names) else f"link_{joint_index}",
            }
        except Exception:
            return self._default_joint_info(joint_index)

    def get_joint_state(self, body_id: int, joint_index: int) -> float:
        body = self._bodies.get(body_id)
        if body is None or "articulation" not in body:
            return 0.0
        try:
            return float(body["articulation"].data.joint_pos[0, joint_index].cpu())
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
        if body is None or "articulation" not in body:
            return
        try:
            import torch

            art = body["articulation"]
            art.write_root_pose_to_sim(
                torch.tensor([position + orientation_quat], device=self.device, dtype=torch.float32)
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
        if body is None or "articulation" not in body:
            return
        try:
            import torch

            art = body["articulation"]
            joint_pos = art.data.joint_pos.clone()
            joint_vel = art.data.joint_vel.clone()
            joint_pos[0, joint_index] = target_value
            joint_vel[0, joint_index] = target_velocity
            art.write_joint_state_to_sim(joint_pos, joint_vel)
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
            import torch

            art = body["articulation"]
            targets = art.data.joint_pos.clone()
            targets[0, joint_index] = target_position
            art.set_joint_position_target(targets)
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
            import torch

            art = body["articulation"]
            targets = torch.zeros_like(art.data.joint_vel)
            art.set_joint_velocity_target(targets)
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
        log.debug("Isaac Lab: gear constraints set via USD schema. Skipping runtime creation.")
        return -1

    # ------------------------------------------------------------------
    # Contacts
    # ------------------------------------------------------------------

    def get_contacts(self, body_id: int) -> List[Tuple]:
        body = self._bodies.get(body_id)
        if body is None or "articulation" not in body:
            return []
        try:
            art = body["articulation"]
            net_forces = art.data.body_force_w[0].cpu().numpy()
            contacts = []
            for link_idx in range(net_forces.shape[0]):
                force_mag = np.linalg.norm(net_forces[link_idx])
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
        # Isaac Lab visual changes require USD API
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
        # Isaac Lab uses Camera sensor class for rendering
        try:
            from omni.isaac.lab.sensors import Camera, CameraCfg

            # Use tiled camera for GPU-accelerated rendering
            if not hasattr(self, "_camera_sensor"):
                cam_cfg = CameraCfg(
                    prim_path="/World/camera",
                    update_period=0.0,
                    height=height,
                    width=width,
                    data_types=["rgb", "distance_to_camera"],
                )
                self._camera_sensor = Camera(cam_cfg)
                self._camera_sensor.reset()

            self._camera_sensor.update(dt=self._timestep)
            rgb = self._camera_sensor.data.output["rgb"][0].cpu().numpy()
            depth = self._camera_sensor.data.output["distance_to_camera"][0].cpu().numpy()
            return rgb[:, :, :3], depth[:, :, 0]
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
        try:
            from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg

            ik_cfg = DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="dls",
            )
            ik_controller = DifferentialIKController(ik_cfg, num_envs=1, device=self.device)

            import torch

            target = torch.tensor(
                [list(target_position) + list(target_orientation_quat)],
                device=self.device,
                dtype=torch.float32,
            )
            ik_controller.set_command(target)

            body = self._bodies.get(body_id)
            if body and "articulation" in body:
                art = body["articulation"]
                jacobian = art.root_physx_view.get_jacobians()
                joint_pos = art.data.joint_pos
                actions = ik_controller.compute(
                    art.data.body_pos_w[:, end_effector_link],
                    art.data.body_quat_w[:, end_effector_link],
                    jacobian[:, end_effector_link - 1, :, :7],
                    joint_pos[:, :7],
                )
                return actions[0].cpu().numpy()
        except Exception:
            pass
        return np.zeros(7)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
