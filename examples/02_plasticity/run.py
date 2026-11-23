"""02 - Elasto-plasticity (new in MuJoCo 4.0).

Springs can now *yield*. Give a joint/tendon spring a `springyield` threshold
(or a 2D/3D flex a `yield` strain) and, once it is deflected past that
threshold, its rest configuration moves. The bend stays after you let go.

The moving rest configuration is real simulation state: it lives in
`mjData.plastic` (address per joint: `mjModel.jnt_plasticadr`), is part of
`mjSTATE_PLASTIC`, and is restored by `mj_resetData`.

Part 1: our own scene.xml - push two cantilevers (elastic vs plastic) with the
        same force, release, and plot tip height + mjData.plastic over time.
Part 2: the upstream `model/plastic/wire.xml` (joint AND tendon springyield).
Part 3: the upstream `model/plastic/foil.xml` (2D flex shell with `yield`).

    uv run python examples/02_plasticity/run.py
    uv run python examples/02_plasticity/run.py --viewer     # drag a tip with Ctrl+right-click
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

import mjplay

HERE = Path(__file__).parent
PUSH_FORCE = 1.2  # N, upward, applied to both tips
PUSH_UNTIL = 1.0  # s; released afterwards


def push_then_release(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    force = PUSH_FORCE if data.time < PUSH_UNTIL else 0.0
    for tip in ("elastic_tip", "plastic_tip"):
        data.body(tip).xfrc_applied[2] = force
    mujoco.mj_step(model, data)


def cantilever_demo(args) -> None:
    model = mujoco.MjModel.from_xml_path(str(HERE / "scene.xml"))
    data = mujoco.MjData(model)
    print(f"nplastic = {model.nplastic}   (one rest angle per plastic hinge)")
    print(f"jnt_plasticadr = {model.jnt_plasticadr}   (-1 = elastic joint)")

    log = {"t": [], "elastic_z": [], "plastic_z": [], "plastic": []}

    def record(m, d):
        log["t"].append(d.time)
        log["elastic_z"].append(d.site("elastic_tip").xpos[2])
        log["plastic_z"].append(d.site("plastic_tip").xpos[2])
        log["plastic"].append(d.plastic.copy())

    frames = mjplay.render_rollout(model, data, seconds=args.seconds, camera="front",
                                   width=args.width, height=args.height,
                                   step=push_then_release, on_frame=record)
    legend = "red: elastic | blue: springyield=8deg"
    frames = [mjplay.label_frames([f], f"{'pushing...' if t < PUSH_UNTIL else 'released  '} | {legend}")[0]
              for f, t in zip(frames, log["t"])]
    mjplay.save_video(frames, "02_cantilever_elastic_vs_plastic.mp4")

    # --- plot ---------------------------------------------------------------
    t = np.array(log["t"])
    plastic = np.degrees(np.array(log["plastic"]))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(t, log["elastic_z"], "r", label="elastic tip")
    ax1.plot(t, log["plastic_z"], "b", label="plastic tip")
    ax1.axvline(PUSH_UNTIL, color="k", ls="--", lw=0.8)
    ax1.set_ylabel("tip height [m]")
    ax1.legend()
    for j in range(plastic.shape[1]):
        ax2.plot(t, plastic[:, j], label=f"p{j}")
    ax2.axvline(PUSH_UNTIL, color="k", ls="--", lw=0.8, label="release")
    ax2.set_ylabel("mjData.plastic\n(spring rest angle) [deg]")
    ax2.set_xlabel("time [s]")
    ax2.legend(ncol=5, fontsize=7)
    fig.suptitle("Joint springyield: the spring's rest angle moves once yield is exceeded")
    fig.tight_layout()
    out = mjplay.output_path("02_cantilever_plastic_state.png")
    fig.savefig(out, dpi=120)
    print(f"wrote outputs/{out.name}")

    # --- check --------------------------------------------------------------
    elastic_z, plastic_z = log["elastic_z"][-1], log["plastic_z"][-1]
    print(f"final tip height  elastic: {elastic_z:+.4f} m   plastic: {plastic_z:+.4f} m")
    print(f"final plastic rest angles [deg]: {np.round(plastic[-1], 2)}")
    assert abs(elastic_z) < 0.1 * plastic_z, "elastic beam should (nearly) spring back to z=0"
    assert plastic_z > 0.05, "plastic beam should stay bent upward"
    print("OK: elastic beam returned to straight; plastic beam kept a permanent bend.")

    # mjData.plastic is state: a reset puts the rest angles back.
    mujoco.mj_resetData(model, data)
    print(f"after mj_resetData, plastic = {data.plastic}  (undeformed again)")


def wire_demo(args) -> None:
    # Upstream: spring-steel / brass / soft-copper wires and cables hit by falling balls.
    path = mjplay.UPSTREAM_MODELS / "plastic" / "wire.xml"
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    print(f"wire.xml: nv={model.nv}, nplastic={model.nplastic} "
          f"(plastic joints: {(model.jnt_plasticadr >= 0).sum()}, "
          f"plastic tendons: {(model.tendon_plasticadr >= 0).sum()})")
    frames = mjplay.render_rollout(model, data, seconds=4.0, width=args.width, height=args.height)
    mjplay.save_video(frames, "02_upstream_wire.mp4")
    print(f"wire.xml max |plastic| after 4 s: {np.abs(data.plastic).max():.3f}")


def foil_demo(args) -> None:
    # Upstream: steel balls hail onto two 2D shells. Left sheet elastic, right sheet has
    # flex <elasticity yield="0.00016 0.0016"/> (stretch, bend) and keeps its dents.
    path = mjplay.UPSTREAM_MODELS / "plastic" / "foil.xml"
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    for f in range(model.nflex):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_FLEX, f)
        print(f"foil.xml flex '{name}': yield={model.flex_yield[f]}, "
              f"plasticmode={model.flex_plasticmode[f]} (1=bend, 2=stretch, 3=both), "
              f"plasticadr={model.flex_plasticadr[f]}")
    mujoco.mj_forward(model, data)
    rest = data.flexvert_xpos.copy()
    frames = mjplay.render_rollout(model, data, seconds=4.0, width=args.width, height=args.height)
    mjplay.save_video(mjplay.label_frames(frames, "left: elastic shell | right: yield (dents stay)"),
                      "02_upstream_foil.mp4")
    # Residual shape change of each sheet after the hail has bounced off.
    devs = {}
    for f in range(model.nflex):
        a, n = model.flex_vertadr[f], model.flex_vertnum[f]
        dev = np.linalg.norm(data.flexvert_xpos[a:a + n] - rest[a:a + n], axis=1).max()
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_FLEX, f)
        print(f"foil.xml '{name}': max vertex displacement after 4 s = {dev * 1000:.1f} mm")
        devs[name] = dev
    assert devs["foil"] > 10 * devs["steel"], "plastic foil should keep much deeper dents"
    print("OK: the elastic sheet recovered; the yielding foil kept its dents.")


def main() -> None:
    args = mjplay.example_args(__doc__, seconds=4.0, upstream="view upstream wire.xml in --viewer mode", foil="view upstream foil.xml in --viewer mode")
    if args.viewer:
        path = HERE / "scene.xml"
        if args.upstream:
            path = mjplay.UPSTREAM_MODELS / "plastic" / "wire.xml"
        if args.foil:
            path = mjplay.UPSTREAM_MODELS / "plastic" / "foil.xml"
        model = mujoco.MjModel.from_xml_path(str(path))
        mjplay.view(model, mujoco.MjData(model))
        return
    cantilever_demo(args)
    wire_demo(args)
    foil_demo(args)


if __name__ == "__main__":
    main()
