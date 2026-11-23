# 02 · Elasto-plasticity

```bash
uv run python examples/02_plasticity/run.py            # 3 videos + 1 plot in outputs/
uv run python examples/02_plasticity/run.py --viewer   # our cantilevers (drag with Ctrl+right-click)
uv run python examples/02_plasticity/run.py --viewer --upstream   # upstream wire.xml
uv run python examples/02_plasticity/run.py --viewer --foil       # upstream foil.xml
```

## What's new in 4.0
Springs can now **yield**. Past a threshold, the spring's rest configuration moves and the deformation becomes permanent ("ideal elasto-plasticity").

| Where | Attribute | Plastic state |
|---|---|---|
| joint spring (slide, hinge, ball) | `<joint springyield="...">` (degrees or radians, following `compiler/angle`) | rest angle/position, per joint at `jnt_plasticadr` |
| tendon spring (spatial, fixed) | `<spatial springyield="...">`, `<fixed springyield="...">` | rest length, at `tendon_plasticadr` |
| 2D/3D flex | `<elasticity yield="stretch [bend]">` (dimensionless strain) | per-element metric offsets, at `flex_plasticadr` |

All of this state lives in **`mjData.plastic`** (`mjModel.nplastic` entries). It belongs to `mjSTATE_PLASTIC`, so `mj_getState`/`mj_setState` can save and restore it, and `mj_resetData` puts the model back undeformed.

## What the script does
1. **`scene.xml`** has two identical 8-link cantilevers. Every hinge has a spring with `stiffness="4"`. The blue one adds `springyield="8"` (degrees). Both tips are pushed up with 1.2 N for 1 s and then released.
   - Video `02_cantilever_elastic_vs_plastic.mp4`: red springs back, blue stays bent.
   - Plot `02_cantilever_plastic_state.png`: tip heights and each hinge's `mjData.plastic` entry over time. The rest angle only moves while a joint is deflected past 8°, and it stays there after release. Only the root joints yield, because the bending moment is largest there.
2. **Upstream `wire.xml`**: spring-steel, brass and copper wires (hinge `springyield`) and cables (tendon `springyield`) hit by falling balls.
3. **Upstream `foil.xml`**: hail on two 2D shells. Left: elastic. Right: `yield="0.00016 0.0016"` (stretch, bend). The script measures how far each sheet's vertices end up from their start. The elastic sheet comes back to about 1 mm; the foil keeps dents of about 44 mm.

## Things to try
- Change `springyield` from 8 to 2 or 20 in `scene.xml`. A smaller value means more permanent bend.
- Set `model.jnt_springyield[j] = 0` at runtime on a plastic joint. That gives zero yield, so the joint slips fully plastically.
- Save the state with `mujoco.mj_getState(m, d, buf, mujoco.mjtState.mjSTATE_PLASTIC)`, bend the beam, then restore it.
- In `foil.xml`, set `yield="0.00016 0"` so only stretching is plastic, and see how the dents change.

## Docs
- [joint springyield](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-joint-springyield)
- [tendon springyield](https://mujoco.readthedocs.io/en/stable/XMLreference.html#tendon-spatial-springyield)
- [flex elasticity yield](https://mujoco.readthedocs.io/en/stable/XMLreference.html#flex-elasticity-yield)
- [Upstream examples: model/plastic](https://github.com/google-deepmind/mujoco/tree/main/model/plastic) (Apache-2.0, loaded from `third_party/`)
