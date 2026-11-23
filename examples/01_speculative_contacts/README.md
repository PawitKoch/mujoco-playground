# 01 — Speculative contacts (continuous collision detection)

```bash
uv run python examples/01_speculative_contacts/run.py                         # writes two side-by-side videos
uv run python examples/01_speculative_contacts/run.py --viewer                # flag on
uv run python examples/01_speculative_contacts/run.py --viewer --no-speculative
```

## The problem

MuJoCo detects collisions at the **end** of each step. A 4 mm bullet moving at 60 m/s
with a 2 ms timestep moves 12 cm per step. It is in front of a 1 cm wall at one step
and behind it at the next, so the two geoms never overlap and no contact is ever made.
This is called *tunnelling*. The usual fixes were to shrink the timestep (slow) or
thicken the geometry (wrong).

## What's new in 4.0

```xml
<option><flag speculative="enable"/></option>
```

For every convex geom pair, MuJoCo sweeps the motion over the step (conservative advancement)
and adds a contact **before** the geoms touch. The constraint solver then stops the
approach exactly at the surface. At runtime this is just a bit you can flip:

```python
model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_SPECULATIVE
```

## What to look at

| Video | Left (on) | Right (off) |
|---|---|---|
| `outputs/01_bullet_speculative_on_vs_off.mp4` (slow-mo) | The bullet stops at the wall, x ≈ 0 | The bullet appears behind the wall, x ≈ +1.8 m |
| `outputs/01_newton_cradle_on_vs_off.mp4` | The momentum passes down five 3 mm plates | The red plate swings straight through its neighbours |

The cradle is DeepMind's upstream `model/speculative/newton.xml`. Read its comments on
`solref="0 -499"`, which shows how to get nearly elastic bounces with speculative contacts.

## Things to try

- Raise the bullet speed to 300 m/s, or make the wall 1 mm thick. Does speculative still hold?
- Swap the bullet for a thin, fast-spinning box. The sweep includes rotation.
- Measure the cost: time `mj_step` with the flag on and off on a dense scene. It costs nothing when nothing is fast.
- Open `third_party/mujoco/model/speculative/flipper.xml` and `chain.xml` in the viewer.

Docs: [Speculative contacts](https://mujoco.readthedocs.io/en/stable/computation/speculative.html) ·
[option/flag/speculative](https://mujoco.readthedocs.io/en/stable/XMLreference.html#option-flag-speculative)
