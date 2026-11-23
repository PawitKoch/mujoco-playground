"""Small helpers shared by the mujoco-playground examples.

Every example is a plain script; these helpers only remove boilerplate so each
`run.py` reads as "the MuJoCo 4 feature" rather than "how to write an mp4".
"""

import os

# Headless rendering by default. Set MUJOCO_GL=glfw yourself for on-screen GL.
os.environ.setdefault("MUJOCO_GL", "egl")

from mjplay.paths import OUTPUTS, REPO_ROOT, UPSTREAM_MODELS, output_path  # noqa: E402
from mjplay.cli import example_args  # noqa: E402
from mjplay.render import (  # noqa: E402
    label_frames,
    render_rollout,
    render_still,
    save_image,
    save_video,
    side_by_side,
)
from mjplay.viewer import view  # noqa: E402

__all__ = [
    "OUTPUTS",
    "REPO_ROOT",
    "UPSTREAM_MODELS",
    "example_args",
    "label_frames",
    "output_path",
    "render_rollout",
    "render_still",
    "save_image",
    "save_video",
    "side_by_side",
    "view",
]
