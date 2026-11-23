# 03 · 1D flexcomp rods: cables, wires and strips

```bash
uv run python examples/03_flex_rods/run.py
uv run python examples/03_flex_rods/run.py --viewer           # five rods
uv run python examples/03_flex_rods/run.py --viewer --cable   # upstream cable.xml (use the ctrl slider to twist)
```

## What's new in 4.0
`<composite type="cable">` and the `cable` elasticity plugin are **removed**. Rods are now 1D flexcomps (`dim="1"`) with `<elasticity>`:

| Feature | MJCF |
|---|---|
| bending stiffness | `<elasticity young="...">` |
| torsional stiffness (**new** optional override of E/2(1+ν)) | `<elasticity shear="...">` |
| **new** rectangular cross-section, anisotropic | `<elasticity section="box" sectionsize="half_w half_h">` |
| **new** curved rest shape | define the rod curved (`type="direct" point="..."` or `type="circle"`) and it stays curved |
| inextensible | `<edge equality="true"/>` |
| clamp the first edge (old `initial="none"`) | `<pin id="0 1"/>` |
| ball joint at the first vertex (old `initial="ball"`) | `<pin id="0"/>` |

**Breaking change:** an open elastic rod with at least 2 elements now creates one twist hinge per edge (`<name>_twist_1 ... <name>_twist`, each on its own twist body). That changes `nq`, `nv`, `nbody` and the `qpos` layout, and shifts the index of everything declared after the rod.

**Address rod bodies and joints by name, never by index.**

Rods also need `integrator="discrete"` (or `Euler`/`RK4`). `implicit` and `implicitfast` are no longer supported for 1D rod bending.

## What the script does
1. **`scene.xml`** clamps five 40 cm rods:
   - `noodle` has no elasticity
   - `round` has young + shear
   - `flat` is a box strip lying flat
   - `edge` is the same strip with `sectionsize` swapped, so it stands on edge
   - `arc` is a curved quarter circle

   The script prints the DOF bookkeeping for each rod. The noodle has 0 twist joints and the elastic rods have 15. It then measures tip sag. The flat strip sags about 33 cm while the edge strip sags about 1 cm (I ∝ wh³ vs hw³). The arc keeps its curve.
2. **Upstream `cable.xml`**: a motor on `cable_twist` (the last edge's twist joint) twists the cable until it coils into a plectoneme. This behaviour needs twist DOFs.

## Things to try
- Remove `shear="8e6"` from `round` so the shear modulus falls back to the isotropic value.
- Change `<pin id="0 1"/>` to `<pin id="0"/>` (ball joint at the root) and watch the rods swing down.
- Increase `young` on `flat` until it stops sagging.
- Make your own spring: `type="direct"` with helix points. See `third_party/mujoco/model/flex/coil.xml`.

## Docs
- [flexcomp](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-flexcomp)
- [flex elasticity (rods, section, shear)](https://mujoco.readthedocs.io/en/stable/XMLreference.html#flex-elasticity)
- [4.0.0 changelog: composite → flexcomp migration](https://mujoco.readthedocs.io/en/stable/changelog.html)
- Upstream: `model/flex/curved_rods.xml`, `cable.xml`, `coil.xml`
