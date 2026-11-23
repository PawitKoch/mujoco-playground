"""03 - 1D flexcomp rods: curved rest shapes, twist, box sections (new in MuJoCo 4.0).

MuJoCo 4 removes `<composite type="cable">` and the `cable` plugin. Cables, wires,
ropes and strips are now 1D `<flexcomp dim="1">` with `<elasticity>`:

* bending from `young`, torsion from `shear` (new, optional override)
* `section="box"` + `sectionsize` (new) for anisotropic strips (rulers, belts)
* curved rest shapes (new): a rod defined curved stays curved
* open elastic rods get one auto-generated *twist* hinge per edge
  (`<name>_twist_1`, ..., `<name>_twist`) - a breaking change that adds DOFs.

Part 1: our scene.xml - five clamped rods; prints the DOF bookkeeping and the
        tip sag of each rod.
Part 2: upstream `model/flex/cable.xml` - twist one end of a cable with a motor
        on its twist joint until it coils.

    uv run python examples/03_flex_rods/run.py
    uv run python examples/03_flex_rods/run.py --viewer
    uv run python examples/03_flex_rods/run.py --viewer --cable
"""

from pathlib import Path

import mujoco
import numpy as np

import mjplay

HERE = Path(__file__).parent
RODS = ["noodle", "round", "flat", "edge", "arc"]


def names(model: mujoco.MjModel, obj: mujoco.mjtObj, count: int) -> list[str]:
    return [mujoco.mj_id2name(model, obj, i) or "" for i in range(count)]


def rod_bookkeeping(model: mujoco.MjModel) -> None:
    body_names = names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody)
    joint_names = names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
    print(f"whole model: nq={model.nq}  nv={model.nv}  nbody={model.nbody}  nplastic={model.nplastic}")
    print(f"{'rod':8s} {'bodies':>6s} {'dofs':>5s} {'twist joints':>12s}  first twist joints")
    for rod in RODS:
        bodies = [i for i, n in enumerate(body_names) if n.startswith(f"{rod}_")]
        dofs = int(sum(model.body_dofnum[b] for b in bodies))
        twists = [n for n in joint_names if n.startswith(f"{rod}_twist")]
        print(f"{rod:8s} {len(bodies):6d} {dofs:5d} {len(twists):12d}  {twists[:3]}{' ...' if len(twists) > 3 else ''}")
        if rod == "noodle":
            assert not twists, "a rod without elasticity has no twist joints"
        else:
            assert twists, "an elastic open rod gets per-edge twist joints"
    # Address rod vertices by NAME (<rod>_<i>), never by index: twist bodies are
    # interleaved and shift every index after a flexcomp.


def tip(data: mujoco.MjData, rod: str) -> np.ndarray:
    last = 12 if rod == "arc" else 16
    return data.body(f"{rod}_{last}").xpos.copy()


def rods_demo(args) -> None:
    model = mujoco.MjModel.from_xml_path(str(HERE / "scene.xml"))
    data = mujoco.MjData(model)
    rod_bookkeeping(model)

    mujoco.mj_forward(model, data)
    rest = {rod: tip(data, rod) for rod in RODS}
    frames = mjplay.render_rollout(model, data, seconds=args.seconds, camera="front",
                                   width=args.width, height=args.height)
    frames = mjplay.label_frames(frames, "noodle | round | flat strip | edge strip | curved arc")
    mjplay.save_video(frames, "03_rods.mp4")
    mjplay.save_image(mjplay.render_still(model, data, camera="side", width=args.width,
                                          height=args.height), "03_rods_side.png")

    sag = {rod: rest[rod][2] - tip(data, rod)[2] for rod in RODS}
    print("tip sag after %.1f s:" % args.seconds)
    for rod in RODS:
        print(f"  {rod:8s} {sag[rod] * 100:6.1f} cm")
    assert sag["flat"] > 5 * sag["edge"], "box section should be much stiffer on edge"
    assert sag["arc"] < 0.1, "curved rest shape should be (mostly) kept"
    print("OK: strip is stiff on edge / floppy flat; the arc keeps its curved rest shape.")


def cable_demo(args) -> None:
    # Upstream cable.xml: one end clamped (pin id="0 1"), the other on a slider.
    # The motor drives `cable_twist`, the twist hinge of the LAST edge.
    path = mjplay.UPSTREAM_MODELS / "flex" / "cable.xml"
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    print(f"cable.xml: nv={model.nv}, twist joints="
          f"{sum(n.startswith('cable_twist') for n in names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt))}")

    def twist(m, d):
        d.ctrl[0] = m.actuator_ctrlrange[0, 1]  # max torque on the end's twist joint
        mujoco.mj_step(m, d)

    frames = mjplay.render_rollout(model, data, seconds=6.0, width=args.width, height=args.height, step=twist)
    mjplay.save_video(mjplay.label_frames(frames, "twisting the free end -> the cable coils"), "03_cable_twist.mp4")
    print(f"cable twist angle at the driven end after 6 s: {np.degrees(data.joint('cable_twist').qpos[0]):.0f} deg")


def main() -> None:
    args = mjplay.example_args(__doc__, seconds=4.0, cable="view upstream cable.xml in --viewer mode")
    if args.viewer:
        path = mjplay.UPSTREAM_MODELS / "flex" / "cable.xml" if args.cable else HERE / "scene.xml"
        model = mujoco.MjModel.from_xml_path(str(path))
        mjplay.view(model, mujoco.MjData(model))
        return
    rods_demo(args)
    cable_demo(args)


if __name__ == "__main__":
    main()
