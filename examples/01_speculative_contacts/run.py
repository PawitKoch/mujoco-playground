"""01 - Speculative contacts (new in MuJoCo 4.0).

Fast, thin objects used to "tunnel": a collision is only detected if two geoms
overlap at the END of a step. MuJoCo 4 adds `<flag speculative="enable"/>`,
which sweeps each convex geom over the step (conservative advancement) and adds
the contact *before* penetration happens.

This script runs the same scene twice (flag on / off) and writes a
side-by-side video, then does the same for the upstream Newton's cradle made
of 3 mm plates.

    uv run python examples/01_speculative_contacts/run.py
    uv run python examples/01_speculative_contacts/run.py --viewer               # flag on
    uv run python examples/01_speculative_contacts/run.py --viewer --no-speculative
"""

from pathlib import Path

import mujoco
import numpy as np

import mjplay

HERE = Path(__file__).parent
SPEC = mujoco.mjtEnableBit.mjENBL_SPECULATIVE


def set_speculative(model: mujoco.MjModel, on: bool) -> None:
    # The XML flag is just a bit in model.opt.enableflags; you can flip it at runtime.
    if on:
        model.opt.enableflags |= SPEC
    else:
        model.opt.enableflags &= ~int(SPEC)


def bullet_demo(args) -> None:
    clips, final_x = [], {}
    for on in (True, False):
        model = mujoco.MjModel.from_xml_path(str(HERE / "scene.xml"))
        set_speculative(model, on)
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key("fire").id)
        # Slow-motion: render at 600 fps simulated time, play back at 30.
        frames = mjplay.render_rollout(model, data, seconds=0.05, fps=600, camera="side",
                                       width=args.width, height=args.height)
        final_x[on] = data.body("bullet").xpos[0]
        clips.append(mjplay.label_frames(frames, f"speculative={'on' if on else 'off'}  x={final_x[on]:+.3f} m"))
    mjplay.save_video(mjplay.side_by_side(*clips), "01_bullet_speculative_on_vs_off.mp4")

    print(f"bullet final x  speculative ON : {final_x[True]:+.3f} m  (wall at x=0)")
    print(f"bullet final x  speculative OFF: {final_x[False]:+.3f} m")
    assert final_x[True] < 0 < final_x[False], "expected: stopped with speculative, tunnelled without"
    print("OK: speculative contacts stopped the bullet; without them it tunnelled through.")


def newton_demo(args) -> None:
    # Upstream example: five 3 mm steel plates; the first one hits at ~4x its thickness per step.
    path = mjplay.UPSTREAM_MODELS / "speculative" / "newton.xml"
    clips, last_swing = [], {}
    for on in (True, False):
        model = mujoco.MjModel.from_xml_path(str(path))
        set_speculative(model, on)
        data = mujoco.MjData(model)
        frames = mjplay.render_rollout(model, data, seconds=1.5, width=args.width, height=args.height)
        # Joint angles of plate 1 (released) and plate 5 (should be kicked out).
        last_swing[on] = (data.joint("j1").qpos[0], data.joint("j5").qpos[0])
        clips.append(mjplay.label_frames(frames, f"speculative={'on' if on else 'off'}"))
    mjplay.save_video(mjplay.side_by_side(*clips), "01_newton_cradle_on_vs_off.mp4")
    for on, (a1, a5) in last_swing.items():
        print(f"cradle speculative={'on ' if on else 'off'}: plate1 angle {np.degrees(a1):+6.1f} deg, "
              f"plate5 angle {np.degrees(a5):+6.1f} deg")


def main() -> None:
    args = mjplay.example_args(__doc__, no_speculative="disable speculative contacts in --viewer mode")
    if args.viewer:
        model = mujoco.MjModel.from_xml_path(str(HERE / "scene.xml"))
        set_speculative(model, not args.no_speculative)
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        model.opt.timestep = 0.002
        mjplay.view(model, data)  # press Backspace to reset, then load the "fire" keyframe
        return
    bullet_demo(args)
    newton_demo(args)


if __name__ == "__main__":
    main()
