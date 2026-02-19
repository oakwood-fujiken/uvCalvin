"""Asset conversion utilities for CALVIN URDFs.

Provides helpers to locate CALVIN's URDF/mesh assets and adapt them
for different simulator backends (e.g. URDF-to-MJCF conversion for MuJoCo).
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


def find_calvin_data_path() -> Path:
    """Locate CALVIN's data directory (containing URDFs, meshes, etc.).

    Search order:
        1. CALVIN_DATA_PATH environment variable
        2. Relative to this package: ../../calvin_env/data
        3. Relative to this package: ../../calvin_env/calvin_env/data

    Returns:
        Path to the CALVIN data directory.

    Raises:
        FileNotFoundError: if no data directory is found.
    """
    env_path = os.environ.get("CALVIN_DATA_PATH")
    if env_path and Path(env_path).is_dir():
        return Path(env_path)

    pkg_root = Path(__file__).parents[1]
    candidates = [
        pkg_root.parent / "calvin_env" / "data",
        pkg_root.parent / "calvin_env" / "calvin_env" / "data",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        "Cannot locate CALVIN data directory. "
        "Set the CALVIN_DATA_PATH environment variable or ensure "
        "calvin_env/data exists relative to the calvin_metasim package."
    )


def urdf_to_mjcf(urdf_path: str, output_dir: Optional[str] = None) -> str:
    """Convert a URDF file to MJCF XML for MuJoCo.

    This is a thin wrapper around mujoco's compile utility. If the conversion
    has already been performed (the output XML exists), it is returned directly.

    Args:
        urdf_path: Path to the source URDF file.
        output_dir: Optional directory for the output XML. Defaults to same
                    directory as the URDF with a `.xml` extension.

    Returns:
        Path to the resulting MJCF XML file.
    """
    urdf = Path(urdf_path)
    if not urdf.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        mjcf_path = out_dir / urdf.with_suffix(".xml").name
    else:
        mjcf_path = urdf.with_suffix(".xml")

    if mjcf_path.exists():
        return str(mjcf_path)

    try:
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(urdf))
        mujoco.mj_saveLastXML(str(mjcf_path), model)
        log.info(f"Converted {urdf_path} -> {mjcf_path}")
        return str(mjcf_path)
    except ImportError:
        log.warning("mujoco package not available. Trying compile command.")
        import subprocess

        result = subprocess.run(
            ["python", "-m", "mujoco.compile", str(urdf), str(mjcf_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"URDF->MJCF conversion failed: {result.stderr}")
        return str(mjcf_path)


def ensure_mesh_paths(urdf_path: str, data_path: str) -> str:
    """Ensure that mesh paths in a URDF are resolvable.

    Some CALVIN URDFs use relative mesh paths like ``meshes/visual/foo.obj``.
    This function checks that the referenced meshes exist relative to the URDF
    directory, and if not, creates symlinks from the data_path.

    Args:
        urdf_path: Path to the URDF file.
        data_path: Root data directory where meshes may live.

    Returns:
        The urdf_path unchanged (for chaining convenience).
    """
    urdf = Path(urdf_path)
    urdf_dir = urdf.parent
    data = Path(data_path)

    with open(urdf, "r") as f:
        content = f.read()

    import re

    mesh_refs = re.findall(r'filename="([^"]+)"', content)
    for ref in mesh_refs:
        ref_path = urdf_dir / ref
        if ref_path.exists():
            continue
        # Try to find the mesh in the data_path tree
        candidates = list(data.rglob(Path(ref).name))
        if candidates:
            ref_path.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(str(candidates[0]), str(ref_path))
            log.debug(f"Linked mesh: {ref_path} -> {candidates[0]}")
        else:
            log.warning(f"Mesh not found: {ref} (from {urdf_path})")

    return urdf_path
