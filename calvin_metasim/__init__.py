"""CALVIN-MetaSim Integration.

Bridge layer between CALVIN's gym-based environment and RoboVerse's MetaSim framework,
enabling CALVIN to run on any simulator backend:
    - PyBullet (original CALVIN backend)
    - MuJoCo (DeepMind)
    - Isaac Lab (NVIDIA, GPU-accelerated)
    - Isaac Sim (NVIDIA Omniverse)
    - Genesis (GPU-accelerated differentiable simulation)
    - SAPIEN (PhysX5 + ray-traced rendering)
"""

from calvin_metasim.env.calvin_metasim_env import (
    SUPPORTED_BACKENDS,
    CalvinMetaSimEnv,
    get_metasim_env,
)
from calvin_metasim.env.handler import SimHandler
from calvin_metasim.wrappers.calvin_compat_wrapper import CalvinCompatWrapper, make_calvin_env
