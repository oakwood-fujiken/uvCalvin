"""Main CALVIN-MetaSim environment — drop-in replacement for PlayTableSimEnv.

This environment delegates to a SimHandler backend and uses the observation/action
adapters to maintain byte-level compatibility with CALVIN's original interface.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple, Union

import gym
import gym.spaces
import numpy as np
from omegaconf import DictConfig

from calvin_metasim.cfg.scenario_cfg import CalvinScenarioCfg, build_scenario_from_hydra_cfg
from calvin_metasim.env.action_adapter import ActionAdapter
from calvin_metasim.env.handler import SimHandler
from calvin_metasim.env.observation_adapter import ObservationAdapter
from calvin_metasim.scene.calvin_scene_builder import CalvinSceneBuilder

log = logging.getLogger(__name__)


class CalvinMetaSimEnv(gym.Env):
    """Gym environment for CALVIN running on any MetaSim-supported simulator.

    Drop-in replacement for calvin_env.envs.play_table_env.PlayTableSimEnv.

    Args:
        scenario_cfg: Pre-built scenario configuration.
        handler: SimHandler instance (PyBullet, MuJoCo, etc.).
        obs_type: "state" | "rgb" | "depth" | "rgb_depth" | "all" (default "all").
        show_gui: Whether to show the simulator GUI.
        seed: Random seed.
    """

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(
        self,
        scenario_cfg: CalvinScenarioCfg,
        handler: SimHandler,
        obs_type: str = "all",
        show_gui: bool = False,
        seed: int = 0,
    ):
        super().__init__()
        self.scenario_cfg = scenario_cfg
        self.handler = handler
        self.obs_type = obs_type
        self.show_gui = show_gui

        # Random state
        self.np_random = np.random.RandomState(seed)

        # Launch simulator
        self.handler.launch(show_gui=show_gui)
        self.handler.set_timestep(scenario_cfg.timestep)

        # Build scene
        self.scene_builder = CalvinSceneBuilder(scenario_cfg, handler, self.np_random)
        robot_id, fixed_ids, movable_ids, interactive_mgr = self.scene_builder.build()

        # Observation adapter
        self.obs_adapter = ObservationAdapter(scenario_cfg, handler, interactive_mgr)
        self.obs_adapter.set_ids(robot_id, fixed_ids, movable_ids)
        self.obs_adapter.setup_cameras()

        # Action adapter
        self.action_adapter = ActionAdapter(scenario_cfg.agent, handler, robot_id)

        # Initialize action adapter target pose from current TCP
        tcp_pos, tcp_orn_quat = handler.get_link_pose(robot_id, scenario_cfg.agent.tcp_link_id)
        tcp_orn_euler = handler.quat_to_euler(tcp_orn_quat)
        self.action_adapter.reset(tcp_pos, tcp_orn_euler)

        # Step counter
        self._step_count = 0

        # Action / observation spaces
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float64)
        self.observation_space = self._build_observation_space()

        # Store IDs for external access
        self.robot_id = robot_id
        self.fixed_object_ids = fixed_ids
        self.movable_object_ids = movable_ids
        self.interactive_mgr = interactive_mgr

        log.info(f"CalvinMetaSimEnv initialized with {handler.__class__.__name__} backend")

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def step(self, action: Union[np.ndarray, tuple]) -> Tuple[Dict, float, bool, Dict]:
        """Execute one environment step.

        Args:
            action: 7D relative action or (pos, orn, gripper) absolute tuple.

        Returns:
            (obs, reward, done, info)
        """
        gripper_action = self.action_adapter.apply_action(action)
        self.obs_adapter.gripper_action = gripper_action

        # Step physics (action_repeat times)
        self.handler.step_physics(self.scenario_cfg.action_repeat)

        # Step interactive objects (buttons/switches toggle lights)
        self.interactive_mgr.step()

        self._step_count += 1

        obs = self._get_obs()
        info = self.obs_adapter.get_info()
        reward = 0.0
        done = False

        return obs, reward, done, info

    def reset(
        self,
        robot_obs: Optional[np.ndarray] = None,
        scene_obs: Optional[np.ndarray] = None,
    ) -> Dict:
        """Reset the environment.

        Args:
            robot_obs: Optional 15D robot state to reset to.
            scene_obs: Optional 24D scene state to reset to.

        Returns:
            Initial observation dict.
        """
        self.scene_builder.reset_robot(robot_obs)
        self.scene_builder.reset_scene(scene_obs)

        # Settle physics
        self.handler.step_physics(50)

        # Reset action adapter target pose
        agent_cfg = self.scenario_cfg.agent
        tcp_pos, tcp_orn_quat = self.handler.get_link_pose(self.robot_id, agent_cfg.tcp_link_id)
        tcp_orn_euler = self.handler.quat_to_euler(tcp_orn_quat)
        self.action_adapter.reset(tcp_pos, tcp_orn_euler)

        self.obs_adapter.gripper_action = 1
        self._step_count = 0

        return self._get_obs()

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        """Render a frame (returns the first camera's RGB)."""
        rgb_obs, _ = self.obs_adapter.get_camera_obs()
        if rgb_obs:
            return next(iter(rgb_obs.values()))
        return np.zeros((200, 200, 3), dtype=np.uint8)

    def close(self):
        """Shut down the simulator."""
        self.handler.close()

    def seed(self, seed: int = None):
        self.np_random = np.random.RandomState(seed)
        self.scene_builder.np_random = self.np_random
        return [seed]

    # ------------------------------------------------------------------
    # CALVIN-compatible API
    # ------------------------------------------------------------------

    def get_obs(self) -> Dict:
        """CALVIN-compatible observation getter."""
        return self._get_obs()

    def get_state_obs(self) -> Dict:
        """Return only state observations (no images)."""
        return self.obs_adapter.get_state_obs()

    def get_info(self) -> Dict:
        """Return info dict compatible with Tasks.get_task_info()."""
        return self.obs_adapter.get_info()

    def get_scene_info(self) -> Dict:
        """Return scene info dict."""
        return self.obs_adapter.get_info(use_scene_info=True).get("scene_info", {})

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_obs(self) -> Dict:
        """Build observation dict based on obs_type."""
        if self.obs_type == "state":
            return self.obs_adapter.get_state_obs()
        elif self.obs_type == "rgb":
            rgb_obs, _ = self.obs_adapter.get_camera_obs()
            state_obs = self.obs_adapter.get_state_obs()
            return {**state_obs, "rgb_obs": rgb_obs}
        elif self.obs_type == "depth":
            _, depth_obs = self.obs_adapter.get_camera_obs()
            state_obs = self.obs_adapter.get_state_obs()
            return {**state_obs, "depth_obs": depth_obs}
        else:  # "all" or "rgb_depth"
            return self.obs_adapter.get_obs()

    def _build_observation_space(self) -> gym.spaces.Dict:
        """Build a gym observation space matching CALVIN's format."""
        spaces = {
            "robot_obs": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float64),
            "scene_obs": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(24,), dtype=np.float64),
        }
        if self.obs_type in ("rgb", "all", "rgb_depth"):
            rgb_spaces = {}
            for cam_cfg in self.scenario_cfg.cameras:
                rgb_spaces[f"rgb_{cam_cfg.name}"] = gym.spaces.Box(
                    low=0, high=255, shape=(cam_cfg.height, cam_cfg.width, 3), dtype=np.uint8
                )
            spaces["rgb_obs"] = gym.spaces.Dict(rgb_spaces)
        if self.obs_type in ("depth", "all", "rgb_depth"):
            depth_spaces = {}
            for cam_cfg in self.scenario_cfg.cameras:
                depth_spaces[f"depth_{cam_cfg.name}"] = gym.spaces.Box(
                    low=0, high=np.inf, shape=(cam_cfg.height, cam_cfg.width), dtype=np.float64
                )
            spaces["depth_obs"] = gym.spaces.Dict(depth_spaces)
        return gym.spaces.Dict(spaces)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


SUPPORTED_BACKENDS = ("pybullet", "mujoco", "isaac_lab", "isaac_sim", "genesis", "sapien")


def get_metasim_env(
    hydra_cfg: Optional[DictConfig] = None,
    scenario_cfg: Optional[CalvinScenarioCfg] = None,
    backend: str = "pybullet",
    obs_type: str = "all",
    show_gui: bool = False,
    seed: int = 0,
    **handler_kwargs,
) -> CalvinMetaSimEnv:
    """Factory function to create a CalvinMetaSimEnv.

    Args:
        hydra_cfg: Merged Hydra config (alternative to scenario_cfg).
        scenario_cfg: Pre-built CalvinScenarioCfg.
        backend: One of "pybullet", "mujoco", "isaac_lab", "isaac_sim", "genesis", "sapien".
        obs_type: Observation type.
        show_gui: Show simulator GUI.
        seed: Random seed.
        **handler_kwargs: Extra kwargs passed to the handler constructor.

    Returns:
        CalvinMetaSimEnv instance.
    """
    if scenario_cfg is None:
        if hydra_cfg is None:
            raise ValueError("Either hydra_cfg or scenario_cfg must be provided.")
        scenario_cfg = build_scenario_from_hydra_cfg(hydra_cfg)

    handler = _create_handler(backend, **handler_kwargs)

    return CalvinMetaSimEnv(
        scenario_cfg=scenario_cfg,
        handler=handler,
        obs_type=obs_type,
        show_gui=show_gui,
        seed=seed,
    )


def _create_handler(backend: str, **kwargs) -> SimHandler:
    """Create a SimHandler for the given backend.

    Supported backends:
        - pybullet: PyBullet (CALVIN's original backend)
        - mujoco: MuJoCo (DeepMind's physics engine)
        - isaac_lab: NVIDIA Isaac Lab (GPU-accelerated, formerly Orbit)
        - isaac_sim: NVIDIA Isaac Sim (Omniverse-based)
        - genesis: Genesis (GPU-accelerated differentiable simulation)
        - sapien: SAPIEN (PhysX5-based with ray-traced rendering)
    """
    backend = backend.lower().replace("-", "_")
    if backend == "pybullet":
        from calvin_metasim.env.pybullet_handler import PyBulletHandler

        return PyBulletHandler(**kwargs)
    elif backend == "mujoco":
        from calvin_metasim.env.mujoco_handler import MuJoCoHandler

        return MuJoCoHandler(**kwargs)
    elif backend in ("isaac_lab", "isaaclab", "orbit"):
        from calvin_metasim.env.isaac_lab_handler import IsaacLabHandler

        return IsaacLabHandler(**kwargs)
    elif backend in ("isaac_sim", "isaacsim"):
        from calvin_metasim.env.isaac_sim_handler import IsaacSimHandler

        return IsaacSimHandler(**kwargs)
    elif backend == "genesis":
        from calvin_metasim.env.genesis_handler import GenesisHandler

        return GenesisHandler(**kwargs)
    elif backend == "sapien":
        from calvin_metasim.env.sapien_handler import SapienHandler

        return SapienHandler(**kwargs)
    else:
        raise ValueError(
            f"Unknown backend: {backend}. Supported: {SUPPORTED_BACKENDS}"
        )
