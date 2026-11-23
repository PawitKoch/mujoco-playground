"""Offscreen rendering helpers (classic OpenGL `mujoco.Renderer`, EGL headless)."""

from collections.abc import Callable
from pathlib import Path

import mediapy
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mjplay.paths import output_path

StepFn = Callable[[mujoco.MjModel, mujoco.MjData], None]


def _make_renderer(model: mujoco.MjModel, width: int, height: int) -> mujoco.Renderer:
    # The offscreen framebuffer must fit the requested size.
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    return mujoco.Renderer(model, height=height, width=width)


def render_rollout(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    seconds: float,
    *,
    fps: int = 30,
    width: int = 640,
    height: int = 480,
    camera: int | str = -1,
    step: StepFn | None = None,
    on_frame: StepFn | None = None,
    scene_option: mujoco.MjvOption | None = None,
) -> list[np.ndarray]:
    """Step `data` for `seconds` and return frames sampled at `fps`.

    `step` defaults to mujoco.mj_step; `on_frame` runs after each rendered frame
    (handy for logging a value at video rate).
    """
    step = step or mujoco.mj_step
    frames = []
    with _make_renderer(model, width, height) as renderer:
        t_end = data.time + seconds
        while data.time < t_end:
            frame_end = data.time + 1.0 / fps
            while data.time < frame_end:
                step(model, data)
            renderer.update_scene(data, camera=camera, scene_option=scene_option)
            frames.append(renderer.render())
            if on_frame:
                on_frame(model, data)
    return frames


def render_still(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    width: int = 640,
    height: int = 480,
    camera: int | str | mujoco.MjvCamera = -1,
    scene_option: mujoco.MjvOption | None = None,
) -> np.ndarray:
    mujoco.mj_forward(model, data)
    with _make_renderer(model, width, height) as renderer:
        renderer.update_scene(data, camera=camera, scene_option=scene_option)
        return renderer.render()


def label_frames(frames: list[np.ndarray], text: str) -> list[np.ndarray]:
    """Burn a caption into the top-left corner of every frame."""
    out = []
    font = ImageFont.load_default(size=max(14, frames[0].shape[0] // 22))
    for f in frames:
        img = Image.fromarray(f)
        draw = ImageDraw.Draw(img)
        draw.rectangle(draw.textbbox((8, 6), text, font=font), fill=(0, 0, 0))
        draw.text((8, 6), text, fill=(255, 255, 255), font=font)
        out.append(np.asarray(img))
    return out


def side_by_side(*videos: list[np.ndarray]) -> list[np.ndarray]:
    """Horizontally tile equally sized videos (shorter ones hold their last frame)."""
    n = max(len(v) for v in videos)
    padded = [v + [v[-1]] * (n - len(v)) for v in videos]
    return [np.concatenate(fs, axis=1) for fs in zip(*padded)]


def save_video(frames: list[np.ndarray], name: str, fps: int = 30) -> Path:
    path = output_path(name)
    mediapy.write_video(path, frames, fps=fps)
    print(f"wrote {path.relative_to(path.parents[1])}  ({len(frames)} frames)")
    return path


def save_image(image: np.ndarray, name: str) -> Path:
    path = output_path(name)
    mediapy.write_image(path, image)
    print(f"wrote {path.relative_to(path.parents[1])}")
    return path
