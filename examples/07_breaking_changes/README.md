# 07 — Breaking changes you can *see*: `sidesite` wrapping and RK4 orientation

```bash
uv run python examples/07_breaking_changes/run.py           # plots + videos + printed numbers
uv run python examples/07_breaking_changes/run.py --viewer  # the tumbling T-handle
```

Both changes leave your code compiling and running, but they change the numbers it produces. That is why they are worth a deliberate look when you upgrade.

## A) `sidesite` tendon wrapping: new homotopy rule

**4.0 rule:** the tendon goes around the geom *the same way as the polyline* `p1 → sidesite → p2`. Formally, it is the shortest path with the same winding angle around the geom. Its length is continuous everywhere except **one** place: where an endpoint crosses the **ray opposite the side site**.

**Old rule:** a half-plane test. The tendon could jump onto the geom while clear of it, giving a length discontinuity, and it could pick the wrong side for wraps longer than 180°.

`sidesite.xml` is a minimal setup:

- p1 is fixed, on the left.
- p2 rides an arm that sweeps 360° around a drum.
- The side site (green) is directly below the drum, so the tendon prefers to wrap around the bottom.
- The faint green line marks the ray opposite the side site.

Measured on this machine:

```
discontinuity between theta=270.0 and 270.5 deg: 0.424 -> 0.790 m
```

That is exactly where p2 crosses the +z ray. See `outputs/07_sidesite_length.png` and `07_sidesite_sweep.mp4`.

**Migration advice from the changelog:** put side sites near the middle of the wrapped arc, so that endpoints never cross the opposite ray over the joint range. The upstream `third_party/mujoco/model/tendon_arm/arm26.xml` moved its side sites from directly below the joints to 45° below the arm, which reproduces the old lengths.

## B) RK4 is now 4th-order for ball/free-joint orientations

Before 4.0, RK4 combined stage angular velocities without the Runge–Kutta–Munthe-Kaas correction, so free-body orientations converged at **2nd order**. `tumble.xml` is a T-handle spinning about its *intermediate* principal axis. That spin is unstable (the Dzhanibekov effect), which makes it a demanding test case.

Orientation error at T = 1 s against an RK4 reference with dt = 1e-4:

```
RK4   errors [rad]: dt=0.02 5.18e-07  dt=0.01 3.38e-08  dt=0.005 2.16e-09  dt=0.0025 1.36e-10
RK4   fitted order: 3.96
Euler fitted order: 0.94
```

`outputs/07_rk4_convergence.png` shows the slope-4 line next to a slope-2 guide, which is what the old RK4 orientation update would have given. Euler is shown as a 1st-order baseline.

Two physics details in `run.py` worth noticing:

- A free joint's angular velocity is in the **body** frame, and its linear velocity is that of the body **origin**, not the COM. `spin()` corrects for that, or the handle drifts off screen.
- The orientation error uses `mju_subQuat` rather than `2*arccos(q1·q2)`, which rounds to zero below about 1e-8 rad.

## Other 4.0 breaking changes (not demoed here)

- 1D flexcomps gain per-edge twist joints, which changes `nq` and `nv`. See example 03.
- The cable plugin and `composite` element are gone; use 1D `flexcomp`. See example 03.
- Flex elastic constants were fixed, so models get stiffer.
- Saturated actuators no longer add implicit damping.
- Duplicate `<include>` is rejected.
- `record.cc` is gone; use `render --duration --fps`.
- Python 3.10 is no longer supported.

## Things to try

- Move `side` to `pos="-.14 0 -.14"` (45° down-left) and rerun: the snap moves to the opposite ray.
- Put the side site *inside* the drum. The tendon is then forced through the geom's interior.
- In `rk4_convergence()`, swap the free joint for a `ball` joint (`<joint type="ball"/>`). You should get the same order.

## Docs

- [spatial/geom/sidesite](https://mujoco.readthedocs.io/en/stable/XMLreference.html#spatial-geom-sidesite)
- [option/integrator](https://mujoco.readthedocs.io/en/stable/XMLreference.html#option-integrator)
- [4.0.0 changelog — Breaking changes](https://mujoco.readthedocs.io/en/stable/changelog.html)
