# 04 · Newton on flex, and the experimental `ipc` contact mode

```bash
uv run python examples/04_newton_flex_ipc/run.py          # benchmark table + video (~1.5 min)
uv run python examples/04_newton_flex_ipc/run.py --viewer           # Newton, ipc off
uv run python examples/04_newton_flex_ipc/run.py --viewer --ipc     # Newton, ipc on
uv run python examples/04_newton_flex_ipc/run.py --viewer --cg      # CG
```

## What's new in 4.0
- **Faster sparse `solver="Newton"` on flex models.** It uses a fill-reducing DOF ordering, incremental active-set Hessian updates and a pre-ordered sparse Cholesky for the `discrete` integrator. The release notes quote 1.4–3x, measured against the 3.x Newton. Most upstream `model/flex` scenes now use Newton.
- **Newton + `discrete` supports flex vertices attached to articulated bodies.** Before 4.0 those models needed CG. MuJoCo Warp still needs CG for them.
- **The experimental `<flag ipc="enable"/>` accepts Newton** as well as CG. It only works with `integrator="discrete"`. Flex contact (flex–flex, self, and flex vs a static plane, sphere, capsule, box or mesh) is solved by an augmented-Lagrangian, IPC-style solve. That solve keeps a small rest gap of about 1 mm and prevents interpenetration. Its contacts are **frictionless**, and exact state replay is not supported.

## What the script does
It runs one scene (`scene.xml`: two 17×17 cloths dropped onto a ball and a box, `nv = 1734`) in three contact modes × two solvers:

| contact mode | how |
|---|---|
| passive | `<contact passive="true">` (as in `scene.xml`): a penalty force folded into the `discrete` integrator, with no constraint rows |
| constraint | `model.flex_passive[:] = 0`: flex contacts become solver constraints |
| ipc | `model.opt.enableflags \|= mjENBL_IPC` |

For each run it prints wall-clock seconds per simulated second, mean solver iterations per step, and the smallest gap between a cloth vertex and the ball.

Results on the dev box (single core, 1 s of simulation):

```
contact    solver  wall s/sim s iters/step  min vertex-ball gap
passive    CG              2.66        0.0              0.54 mm
passive    Newton          2.67        0.0              0.54 mm
constraint CG              2.76       63.9            -10.22 mm
constraint Newton          3.58        5.1            -10.22 mm
ipc        CG              3.94      188.5              0.57 mm
ipc        Newton          4.30        6.0              0.65 mm
```

How to read the table:
- **Passive contact gives the solver nothing to do** (0 iterations), so CG and Newton cost the same. This is why the upstream cloth models use passive contact.
- **With constraint contact, Newton converges in about 5 iterations against about 64 for CG.** It is still slower in wall-clock time on this model, because each Newton iteration factorises a 1734-DOF Hessian. The faster solver depends on the model, so measure your own.
- **Soft constraint contacts (`solref="0.01 1"`) let the cloth sink about 1 cm into the ball.** Passive contact and ipc keep it outside. ipc holds it about 0.6 mm off the surface, which is roughly its documented 1 mm rest gap. ipc costs about 1.5x and is frictionless.

The video `04_cloth_newton_ipc_off_vs_on.mp4` compares the two. With ipc off, the box's corners poke through the blue cloth. With ipc on, the cloth wraps cleanly.

## Things to try
- Change `<contact passive="true">` to a stiffer `solref` and rerun the constraint rows.
- Raise the cloth `count` to `25 25 1` and watch how each mode scales.
- Pin the cloth corners (`<pin id="0 16 272 288"/>`) for a hammock.
- Add friction to the scene and notice that ipc ignores it on the pairs it handles.

## Docs
- [option/solver](https://mujoco.readthedocs.io/en/stable/XMLreference.html#option-solver)
- [option/flag/ipc](https://mujoco.readthedocs.io/en/stable/XMLreference.html#option-flag-ipc)
- [flex contact passive](https://mujoco.readthedocs.io/en/stable/XMLreference.html#flex-contact-passive)
- Upstream: `model/flex/drape.xml` (this scene is adapted from it) and `model/flex/bag.xml` (ipc)
