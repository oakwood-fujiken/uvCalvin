"""Compatibility wrapper — makes CalvinMetaSimEnv a drop-in for PlayTableSimEnv.

This wrapper provides all the attributes and methods that CALVIN's training
pipelines (e.g., calvin_agent, hulc) expect on the environment object,
bridging any naming or API differences.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Union

import gym
import numpy as np
from omegaconf import DictConfig

from calvin_metasim.env.calvin_metasim_env import CalvinMetaSimEnv, get_metasim_env

log = logging.getLogger(__name__)


class CalvinCompatWrapper(gym.Wrapper):
    """Gym wrapper that provides CALVIN PlayTableSimEnv compatibility.

    Makes CalvinMetaSimEnv callable from:
        - calvin_agent training scripts
        - hulc / hulc2 training pipelines
        - CALVIN evaluation scripts (rollout.py, etc.)

    Adds:
        - .robot attribute (proxy for robot API)
        - .scene attribute (proxy for scene API)
        - .cameras attribute
        - .tasks attribute
        - PlayTableSimEnv-style reset(robot_obs, scene_obs) signature
    """

    def __init__(self, env: CalvinMetaSimEnv):
        super().__init__(env)
        self._metasim_env: CalvinMetaSimEnv = env

        # Expose CALVIN-style attributes
        self.robot = _RobotProxy(env)
        self.scene = _SceneProxy(env)

    # ------------------------------------------------------------------
    # CALVIN-compatible step / reset
    # ------------------------------------------------------------------

    def step(self, action):
        return self._metasim_env.step(action)

    def reset(
        self,
        robot_obs: Optional[np.ndarray] = None,
        scene_obs: Optional[np.ndarray] = None,
        **kwargs,
    ):
        return self._metasim_env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    def get_obs(self):
        return self._metasim_env.get_obs()

    def get_state_obs(self):
        return self._metasim_env.get_state_obs()

    def get_info(self):
        return self._metasim_env.get_info()

    # ------------------------------------------------------------------
    # Attributes expected by CALVIN
    # ------------------------------------------------------------------

    @property
    def action_type(self):
        return "rel"

    @property
    def observation_space_keys(self):
        return self._metasim_env.observation_space.spaces.keys()


class _RobotProxy:
    """Proxy object mimicking CALVIN's Robot class interface."""

    def __init__(self, env: CalvinMetaSimEnv):
        self._env = env

    def get_observation(self):
        state_obs = self._env.obs_adapter.get_state_obs()
        return state_obs["robot_obs"]

    def get_tcp_pos(self):
        agent = self._env.scenario_cfg.agent
        pos, _ = self._env.handler.get_link_pose(self._env.robot_id, agent.tcp_link_id)
        return pos

    def get_tcp_orn(self):
        agent = self._env.scenario_cfg.agent
        _, orn_quat = self._env.handler.get_link_pose(self._env.robot_id, agent.tcp_link_id)
        return self._env.handler.quat_to_euler(orn_quat)

    def get_gripper_width(self):
        agent = self._env.scenario_cfg.agent
        return sum(
            self._env.handler.get_joint_state(self._env.robot_id, jid)
            for jid in agent.gripper_joint_ids
        )

    @property
    def gripper_finger_ids(self):
        return self._env.scenario_cfg.agent.gripper_joint_ids


class _SceneProxy:
    """Proxy object mimicking CALVIN's PlayTableScene interface."""

    def __init__(self, env: CalvinMetaSimEnv):
        self._env = env

    def get_obs(self):
        state_obs = self._env.obs_adapter.get_state_obs()
        return state_obs["scene_obs"]

    def get_info(self):
        return self._env.obs_adapter.get_info(use_scene_info=True).get("scene_info", {})

    @property
    def fixed_objects(self):
        return self._env.fixed_object_ids

    @property
    def movable_objects(self):
        return self._env.movable_object_ids

    @property
    def interactive_objects(self):
        return self._env.interactive_mgr


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_calvin_env(
    hydra_cfg: Optional[DictConfig] = None,
    backend: str = "pybullet",
    obs_type: str = "all",
    show_gui: bool = False,
    seed: int = 0,
    wrap_compat: bool = True,
    **handler_kwargs,
) -> Union[CalvinMetaSimEnv, CalvinCompatWrapper]:
    """Create a CALVIN environment with optional compatibility wrapper.

    Args:
        hydra_cfg: Merged Hydra config.
        backend: Simulator backend name.
        obs_type: Observation type.
        show_gui: GUI flag.
        seed: Random seed.
        wrap_compat: If True, wrap with CalvinCompatWrapper for drop-in compatibility.
        **handler_kwargs: Backend-specific kwargs.

    Returns:
        Environment instance (wrapped or unwrapped).
    """
    env = get_metasim_env(
        hydra_cfg=hydra_cfg,
        backend=backend,
        obs_type=obs_type,
        show_gui=show_gui,
        seed=seed,
        **handler_kwargs,
    )
    if wrap_compat:
        return CalvinCompatWrapper(env)
    return env
