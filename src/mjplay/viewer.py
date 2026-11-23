"""Interactive viewing.

`view()` uses the classic passive viewer so examples can keep their own
control loop. For the new MuJoCo Studio viewer see examples/10_studio_viewer.
"""

import time

import mujoco

from mjplay.render import StepFn


def view(model: mujoco.MjModel, data: mujoco.MjData, step: StepFn | None = None) -> None:
    step = step or mujoco.mj_step
    import mujoco.viewer  # imported lazily: needs a display

    with mujoco.viewer.launch_passive(model, data) as v:
        while v.is_running():
            tick = time.perf_counter()
            step(model, data)
            v.sync()
            time.sleep(max(0.0, model.opt.timestep - (time.perf_counter() - tick)))
