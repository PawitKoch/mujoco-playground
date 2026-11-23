# mujoco-playground

A hands-on learning repo for **MuJoCo 4** (4.0.1). Each folder in `examples/` is one new
feature: a small MJCF scene, a `run.py` you can read top to bottom, and a README that covers
*what's new, what to look at, and what to try next*. Every script runs headless and writes
videos or plots to `outputs/`. Add `--viewer` to explore interactively.

Example 09 adds a **"vibe sim"** loop: you chat with Gemini 4 Hydrogen, and it edits and
verifies a MuJoCo scene with an SO-101 arm.

> Not to be confused with DeepMind's [`mujoco_playground`](https://github.com/google-deepmind/mujoco_playground)
> (RL environments on MJX). This repo is a personal feature tour.

## Setup (2 minutes)

```bash
cd mujoco-playground
uv sync --all-groups            # Python 3.12 venv: mujoco 4.0.1 + warp/mjx/jax (gpu) + google-genai (gemini)
./scripts/fetch_upstream.sh     # sparse checkout of DeepMind's 4.0.1 example models -> third_party/
uv run python examples/00_hello_mujoco4/run.py
```

- Core only (no GPU or Gemini): `uv sync`. Add groups with `uv sync --group gpu` or `--group gemini`.
- Gemini: put `GEMINI_API_KEY=...` in `.env`. It is gitignored; see `.env.example`.
- Rendering is headless EGL by default (`MUJOCO_GL=egl`). Viewer windows need a display.
- Small shared helpers live in `src/mjplay/`: render to mp4, side-by-side comparisons, captions, and the viewer.

## The tour

| # | Example | New in 4.x | Key result you'll see |
|---|---|---|---|
| 00 | [hello_mujoco4](examples/00_hello_mujoco4) | Package split (`mujoco-core` / `mujoco-studio`), bundled manual and agent skills, `geom_xquat` | Install check; lists the 7 bundled skills |
| 01 | [speculative_contacts](examples/01_speculative_contacts) | `<flag speculative>`: continuous collision detection | A 60 m/s bullet stops at a 1 cm wall; with the flag off it tunnels 1.8 m past it |
| 02 | [plasticity](examples/02_plasticity) | `springyield` (joints and tendons), flex `yield`, `mjData.plastic` | The plastic beam stays bent (+0.40 m); the elastic one springs back |
| 03 | [flex_rods](examples/03_flex_rods) | 1D flexcomp rods: twist joints, `shear`, box sections, curved rest shapes; `composite` and `cable` removed | Flat strip vs edge-on strip sag 33× differently; twist DOFs listed |
| 04 | [newton_flex_ipc](examples/04_newton_flex_ipc) | Faster sparse Newton on flex, experimental `ipc` flag | Cloth drape: penetration with ipc off vs a clean gap with ipc on; CG vs Newton timing |
| 05 | [rendering](examples/05_rendering) | `castshadow`, Filament haze and skybox, `headlight active="auto"` | Classic and Filament stills, rendered headless |
| 06 | [mjspec_editing](examples/06_mjspec_editing) | `flex.material=[...]` per element, MJB via `mj_encode`, `surfacevel`, `tendon_type` | Procedural conveyor scene; exact MJB round trip |
| 07 | [breaking_changes](examples/07_breaking_changes) | `sidesite` homotopy rule, RK4 4th-order for free and ball joints | Tendon length jump at exactly one angle; RK4 fitted order 3.96 |
| 08 | [warp_gpu_batch](examples/08_warp_gpu_batch) | MuJoCo Warp and MJX (`impl="warp"`), 4.0.1 | 8192 humanoids: about 6.3 M steps/s on the 5090, 135× one CPU thread |
| 09 | [vibe_sim_gemini](examples/09_vibe_sim_gemini) | Gemini 4 Hydrogen + MjSpec tools; Antigravity Robotics plugin | Chat → spec edit → compile → simulate → self-check |
| 10 | [studio_viewer](examples/10_studio_viewer) | MuJoCo Studio: web and native viewer, Python plugins | Web viewer with a custom ImGui panel and a 3D overlay |

Run everything as a smoke test (each example asserts that its effect really happened):

```bash
uv run python scripts/run_all.py               # ~1 min; skips 09
uv run python scripts/run_all.py 01 07         # just some
uv run python scripts/run_all.py --with-gemini # also one scripted vibe-sim turn (calls the API)
```

## Suggested learning order

1. **00 → 01 → 07**: the rigid-body changes. These may affect your existing models.
2. **06**: MjSpec, the Python model editor. Everything after this builds scenes in code.
3. **02 → 03 → 04**: deformables, the biggest area of change in 4.0.
4. **05, 10**: the new rendering and viewer stack.
5. **08**: scaling up on the GPU.
6. **09**: put it together with an LLM in the loop.

## Migrating 3.x models: cheat sheet

| 3.x | 4.x |
|---|---|
| `<composite type="cable">`, `cable` plugin | `<flexcomp dim="1" type="grid">` + `<elasticity young= shear=>` + `<pin id="0 1"/>`, with `integrator="discrete"` (ex. 03) |
| `<equality><flexvert>`, `<edge equality="vert"/>` | continuum `elasticity` or `<edge equality="true"/>` |
| Rod `qpos` indices | Open rods add `<name>_twist_*` hinge joints. Index by name, and save state with `mjSTATE_PLASTIC` |
| 1D rod bending under `implicit`/`implicitfast` | Not supported. Use `discrete` (or `Euler` with a small dt) |
| `record model.xml T F out.rgb` | `render model.xml out.rgb --duration=T --fps=F` |
| `sidesite` half-plane rule | Homotopy rule. Keep endpoints away from the ray opposite the side site (ex. 07) |
| RK4 orientations (2nd order) | 4th order. Trajectories with ball or free joints change slightly |
| `mjModel.flex_matid` per flex | Per element: `flex_matid[flex_elemadr[f] + e]` |
| Flex elastic constants | Corrected, so models get stiffer. The changelog has remapping formulas |
| `pip install -U mujoco` over a 3.x env | `pip uninstall -y mujoco && pip install mujoco` |
| Python 3.10 | 3.11+ |

## Where to read more

- The full manual ships **inside the wheel**: `.venv/lib/python3.12/site-packages/mujoco/doc/`.
  Grep it; it always matches your installed version.
- [Changelog](https://mujoco.readthedocs.io/en/stable/changelog.html) ·
  [Overview](https://mujoco.readthedocs.io/en/stable/overview.html) ·
  [XML reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html)
- Upstream example models: `third_party/mujoco/model/` (`speculative/`, `plastic/`, `flex/`, `surfacevel/` ...).
  Open them with `uv run python -m mujoco.viewer --mjcf=<file>` or
  `uv run python -m mujoco.studio.viewer --model=<file> --gfx=web`.
- [Gemini Hydrogen for robotics](https://developers.googleblog.com/gemini-hydrogen-for-robotics/) ·
  [robotics-antigravity-plugin](https://github.com/google-deepmind/robotics-antigravity-plugin)

Upstream models in `third_party/` and the scenes adapted from them are Apache-2.0 © DeepMind.
