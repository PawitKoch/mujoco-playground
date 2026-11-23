# 10 — MuJoCo Studio: the new viewer, with your own plugins

```bash
uv run python examples/10_studio_viewer/run.py            # web viewer: open the printed URL
uv run python examples/10_studio_viewer/run.py --native   # native Filament desktop window
uv run python examples/10_studio_viewer/run.py --smoke    # headless self-test (3 s), used by run_all
```

Working on a remote GPU box? Run `ssh -L 8080:localhost:8080 <host>` and open http://localhost:8080/. Use `--port 8080` so the port is predictable; the default picks the first free port at or above 8080. The web viewer has **no authentication**, so don't expose the port publicly.

Shortcut with no code at all: `uv run mujoco-studio --model=examples/10_studio_viewer/scene.xml --gfx=web`

## What's new in 4.x

- **`mujoco-studio` package.** Since 4.0 the `mujoco` pip package is a metapackage made of:
  - `mujoco-core`: the bindings and the C library, depending only on numpy
  - `mujoco-studio`: the Studio viewer and the Filament renderer bindings

  `pip install mujoco-core` gives you a GUI-free, numpy-only install for clusters.
- **Studio documentation** is new in 4.0, covering the web and native viewers, the architecture, and Python/C++ plugins. **4.0.1** enabled the web viewer on Windows through the `mujoco-studio-wasm` wheel.
- **Web viewer:**
  - Physics and the UI logic run in Python on the host.
  - The browser only rasterises, through WebAssembly.
  - A host GPU is not needed unless your own code renders camera images.
- **Plugins:** decorate methods with `@messages.handler`. The annotated argument type decides which message a method receives.

## How the three custom plugins talk

```
 sim thread                                   viewer thread
 ───────────                                  ─────────────
 StepControl  (stock: steps + paces physics)  studio_app.App (stock: menus, inspector, timeline...)
 EnergyProbe  ── EnergySnapshot (Snapshot) ─▶ EnergyPanel  (ImGui window)
              ◀── SetGravity   (Event) ────── (gravity slider)
                                              TipVelocityArrow (3D overlay via viewer.extra_geoms)
```

- A **`Snapshot`** is latest-wins and may be dropped, which suits a value that is refreshed every step. An **`Event`** is reliable and ordered, which suits a user action.
- `EnergyProbe` applies gravity changes in `PreStepEvent`. Model mutations belong in step handlers so that Studio's diff-sync sends them to the viewer.
- Viewer plugins treat `self.viewer.model/data` as **read-only mirrors**. That is why the arrow computes the tip velocity by finite difference instead of calling `mj_*` functions on the viewer's data.
- `--smoke` uses **non-blocking** mode: `launch_web.launch(...)` returns a `ViewerHandle`, and your own loop calls `model, data = handle.sync(model, data)`. Always rebind both, because the UI can swap the model (Reload, drag-and-drop).

## Verified here

- `--smoke`: the server returns its HTML page with HTTP 200 (25864 bytes), the sim steps, and the viewer-thread panel receives `EnergySnapshot` messages.
- The full web UI mode starts and serves on `--port`.
- `--native` was not exercised in this session because it needs a desktop session. On macOS, the non-blocking native launch requires `mjpython`.

## Things to try

- Add `hot_reload="all"` to `ViewerConfig`, or run `python -m mujoco.studio.viewer --model=scene.xml --plugins=run.py --hot_reload=all`. Then edit the plugin code while the viewer is running.
- Drop `studio_app.App()` from the plugin list to get a bare viewport with only your panel.
- Replace `StepControl` with a SimPlugin that steps on the GPU with MuJoCo Warp (see example 08), as the docs' *Custom Stepping Backends* pattern describes.
- Call `imgui.ShowDemoWindow()` inside `on_build_gui` to browse every widget.

## Docs

- [Studio overview](https://mujoco.readthedocs.io/en/stable/studio/index.html)
- [Web viewers](https://mujoco.readthedocs.io/en/stable/studio/web_viewers.html), [Native viewers](https://mujoco.readthedocs.io/en/stable/studio/native_viewers.html)
- [Python plugins](https://mujoco.readthedocs.io/en/stable/studio/python_plugins.html)
- [Studio samples upstream](https://github.com/google-deepmind/mujoco/tree/main/python/mujoco/sample/studio)
