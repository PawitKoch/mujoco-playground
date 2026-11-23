"""05 - Rendering: castshadow, Filament haze/skybox, headlight="auto" (MuJoCo 4.0).

Writes three comparison images to outputs/:
  05_castshadow_classic.png  classic OpenGL renderer: authored scene vs every geom forced to cast shadows
  05_filament_haze.png       Filament renderer, low horizon camera: haze on / haze off / skybox off
  05_filament_overview.png   Filament renderer, same view as the classic still

    uv run python examples/05_rendering/run.py
    uv run python examples/05_rendering/run.py --viewer
"""

from pathlib import Path

import mujoco
import numpy as np

import mjplay

HERE = Path(__file__).parent


def load():
    model = mujoco.MjModel.from_xml_path(str(HERE / "scene.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def caption(img: np.ndarray, text: str) -> np.ndarray:
    return mjplay.label_frames([img], text)[0]


def classic_castshadow(args) -> None:
    model, data = load()
    print("castshadow per geom:", {model.geom(i).name: bool(model.geom_castshadow[i]) for i in range(model.ngeom)})
    print(f"visual.headlight.active = {model.vis.headlight.active}  (0=false, 1=true, 2=auto)")

    kw = dict(width=args.width, height=args.height, camera="overview")
    authored = mjplay.render_still(model, data, **kw)
    # castshadow is an ordinary mjModel array: flip it at runtime, no recompile.
    model.geom_castshadow[:] = True
    forced = mjplay.render_still(model, data, **kw)

    changed = np.abs(authored.astype(int) - forced.astype(int)).sum(-1) > 30
    print(f"pixels that change when the blue pillar + green marker cast shadows: {changed.mean():.1%}")
    assert changed.mean() > 0.005, "expected visible shadow difference"
    mjplay.save_image(np.concatenate([caption(authored, "authored: blue pillar castshadow=false"),
                                      caption(forced, "geom_castshadow[:] = True")], axis=1),
                      "05_castshadow_classic.png")


def filament_demo(args) -> None:
    # Filament is MuJoCo's physically based renderer (shipped in mujoco-studio).
    # It renders headless here through Vulkan; no display needed.
    from mujoco import mjrf
    from mujoco.rendering.filament import model_scene, renderer

    model, data = load()
    ctx = mjrf.Context(mjrf.ContextConfig())
    r = renderer.Renderer(ctx)
    scene = model_scene.ModelScene(ctx, model, r.scene("scn"))
    scene.update(data)
    pixels = np.empty((args.height, args.width, 3), np.uint8)
    r.target("out", pixels)

    def shot(camera: str, **view_flags) -> np.ndarray:
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = model.camera(camera).id
        view = renderer.View(scene="scn", target="out",
                             camera=mujoco.mjv_camera2GLCamera(model, data, cam), **view_flags)
        r.render([view])
        return pixels.copy()

    hazy = shot("horizon", haze=True)
    clear = shot("horizon", haze=False)
    nosky = shot("horizon", haze=False, skybox=False)
    print(f"Filament horizon: mean |haze on - haze off| = {np.abs(hazy.astype(int) - clear).mean():.1f} grey levels")
    mjplay.save_image(np.concatenate([caption(hazy, "Filament: haze on"),
                                      caption(clear, "haze off (hard horizon)"),
                                      caption(nosky, "skybox off")], axis=1), "05_filament_haze.png")
    mjplay.save_image(caption(shot("overview"), "Filament: same scene, PBR metal ball"),
                      "05_filament_overview.png")


def main() -> None:
    args = mjplay.example_args(__doc__)
    if args.viewer:
        mjplay.view(*load())
        return
    classic_castshadow(args)
    try:
        filament_demo(args)
    except Exception as e:  # Filament needs Vulkan or GL; keep the classic part useful without it.
        print(f"Filament rendering unavailable here ({type(e).__name__}: {e}); classic stills only.")


if __name__ == "__main__":
    main()
