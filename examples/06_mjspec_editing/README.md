# 06 — Procedural models with `MjSpec` + new 4.0 model/data fields

```bash
uv run python examples/06_mjspec_editing/run.py           # video + printed fields + MJB round trip
uv run python examples/06_mjspec_editing/run.py --viewer
```

The whole scene is built in Python except the cloth, which starts as a 10-line `<flexcomp>` fragment. A flexcomp is an MJCF macro that the parser expands, so it has to come from XML. The scene contains:

- a conveyor belt with no degrees of freedom (`surfacevel`)
- three parcels added in a loop
- a pendulum with a fixed tendon on its hinge
- an elastic spatial tendon tying the last parcel to an anchor
- a checkerboard cloth

## What's new in 4.x (and where it shows up)

| Feature | In `run.py` |
|---|---|
| **Per-element flex materials.** `mjsFlex.material` is now an `MjStringVec`. Assign `flex.material = [...]` with one name per element, or a single name to broadcast it. `mjModel.flex_matid` is `(nflexelem,)` instead of `(nflex,)`. | `colour_cloth()` gives the cloth a checkerboard |
| **MJB output from a spec.** `spec.encode(path, model, content_type="model/vnd.mujoco.mjb")` uses `mj_encode`. The content type can also be inferred from a `.mjb` / `.mjz` / `.xml` / `.txt` extension. | MJB round trip with assertions |
| **`mjData.xiquat`, `geom_xquat`, `site_xquat`.** Quaternion twins of `ximat` / `geom_xmat` / `site_xmat`, so you no longer need `mju_mat2Quat`. | Printed, and checked against `mju_mat2Quat` |
| **`mjModel.tendon_type`.** Uses the `mjtTendon` enum (`mjTENDON_FIXED` / `mjTENDON_SPATIAL`). Previously you inferred the type from the first wrap element. | Printed per tendon |
| **`mjContact.surfacevel[3]`.** The relative surface velocity in the contact frame `(tangent1, tangent2, spin)`, filled from geom `surfacevel`. | Printed for contacts with the belt |

### `surfacevel` in one line

`<geom surfacevel="vx vy vz wx wy wz">` makes the geom's *surface* move while the geom itself stays still. Friction then carries whatever touches it, which gives you conveyors, treadmills and turntables with zero DOFs. You can change it at runtime through `model.geom_surfacevel`. In the run, parcels 0 and 1 settle at the belt speed of 0.40 m/s. Parcel 2 is held back by the rubber band.

The printed contact value looks like `(0, 0.4, 0)` rather than `(0.4, 0, 0)` because it is expressed in the **contact frame**. That frame's first tangent comes from the contact normal, so the belt's x direction lands on tangent 2.

## Things to try

- Set `BELT_SPEED` negative, or add a turntable: a cylinder with `surfacevel=[0,0,0, 0,0,3]`.
- Change `flex.material = "red"` (a single string) to see broadcasting. Or colour the cloth by row to make stripes.
- Open `outputs/06_procedural_scene.xml` to see the MJCF your Python code produced (`spec.to_xml()`).
- Add a body *after* compiling and call `spec.recompile(model, data)` to keep the simulation state while the model changes.
- Compare load times: `MjModel.from_binary_path(mjb)` against `from_xml_path`. MJB is version-specific, so treat it as a cache, not an interchange format.

## Docs

- [Model editing (mjSpec)](https://mujoco.readthedocs.io/en/stable/programming/modeledit.html) — see *Model Encoding & Saving*
- [Python: model editing](https://mujoco.readthedocs.io/en/stable/python.html#model-editing)
- [geom/surfacevel](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-geom-surfacevel)
- [flexcomp](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-flexcomp)
- [4.0.0 changelog](https://mujoco.readthedocs.io/en/stable/changelog.html)

Upstream reference for `surfacevel`: `third_party/mujoco/model/surfacevel/treadmill.xml` (a humanoid on a treadmill).
