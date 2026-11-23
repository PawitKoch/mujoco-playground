# 05 — Rendering: `castshadow`, Filament haze & skybox, `headlight="auto"`

```bash
uv run python examples/05_rendering/run.py           # writes 3 PNGs to outputs/
uv run python examples/05_rendering/run.py --viewer  # classic interactive viewer
```

## What's new in 4.x

| Feature | Where | Notes |
|---|---|---|
| `<geom castshadow="false">` | MJCF, `mjModel.geom_castshadow`, `mjsGeom.castshadow` | Per-geom shadow casting in **both** the classic OpenGL and Filament renderers. Default `true`. |
| `visual/map/haze`, `visual/rgba/haze` | MJCF | **Filament only.** Infinite planes (`size="0 0 …"`) are centred under the camera and fade into the skybox at the horizon. |
| `skybox` / `haze` per view | `mjrfRenderRequest` (`renderer.View(skybox=…, haze=…)` in Python) | Toggle per render call without touching the model. |
| `visual/headlight active="auto"` | MJCF | Now accepts `false` / `true` / `auto` (or 0/1/2). **Default changed to `auto`**: tools like Studio switch the headlight on only when the scene has no lights. Legacy renderers treat `auto` as on. |

## What to look at

- `outputs/05_castshadow_classic.png`: on the left is the scene as authored, where the blue pillar and the green "goal marker" sphere don't cast shadows. On the right, `model.geom_castshadow[:] = True` was set at runtime. This array is an ordinary `mjModel` field, so you can change it without recompiling.
- `outputs/05_filament_haze.png`: Filament from a high camera looking at the horizon. Haze on gives a soft fade into the sky. Haze off gives a hard horizon line. Skybox off gives black.
- `outputs/05_filament_overview.png`: the same scene as the classic still, rendered by Filament with PBR. Compare how the `metallic="1"` ball looks.

The haze effect scales with **camera height**: the haze fraction is a fraction of the distance to the horizon. From a camera 0.5 m above the floor it is a hairline, and from 2.5 m it is clearly visible.

## Filament, headless

The `mjrf` Python module (`from mujoco import mjrf`) plus the helpers in `mujoco.rendering.filament` (`Renderer`, `ModelScene`, `View`) render offscreen. On this machine they render through **Vulkan** with no display. The script falls back to the classic stills if Filament can't create a context.

The native C `render` sample, which replaced `record.cc` in 4.0 (`render model.xml out.rgb --duration=T --fps=F`), is **not** shipped in the Python wheel. You get it from the release archives or by building MuJoCo from source.

## Things to try

- Set `castshadow="false"` on the `sun` light instead, so that light casts no shadows at all.
- Change `<map haze>` between 0.1 and 1, and pick a `rgba haze` colour far from the sky, to see what the blend is doing.
- Delete the `<light>` and run `--viewer`: `headlight="auto"` keeps the scene lit.
- In Python: `mjrf.Renderable.set_cast_shadow(False)` toggles shadows on a Filament renderable directly.

## Docs

- [geom/castshadow](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-geom-castshadow)
- [visual/map/haze](https://mujoco.readthedocs.io/en/stable/XMLreference.html#visual-map-haze), [visual/rgba/haze](https://mujoco.readthedocs.io/en/stable/XMLreference.html#visual-rgba-haze)
- [visual/headlight/active](https://mujoco.readthedocs.io/en/stable/XMLreference.html#visual-headlight-active)
- [Python Filament rendering](https://mujoco.readthedocs.io/en/stable/python.html#filament-rendering)
- [Rendering programming guide](https://mujoco.readthedocs.io/en/stable/programming/rendering.html)
