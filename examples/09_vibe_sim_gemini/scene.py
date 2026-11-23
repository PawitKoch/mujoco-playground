"""The editable scene behind the vibe-sim loop.

Everything lives in an `mujoco.MjSpec`, MuJoCo's in-memory model editor. Every
edit Gemini asks for is a plain spec mutation followed by `spec.compile()`, so
a bad edit surfaces as a compile error we can hand straight back to the model.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

MENAGERIE = Path(os.environ.get("MENAGERIE_PATH", Path(__file__).resolve().parents[3] / "mujoco_menagerie"))
SO101 = MENAGERIE / "robotstudio_so101" / "so101.xml"

BASE_XML = """
<mujoco model="vibe_sim">
  <!-- match the SO-101 menagerie options so attaching it raises no conflicts -->
  <option timestep="0.005" integrator="implicitfast" cone="elliptic" impratio="10" iterations="10" ls_iterations="20"/>
  <visual>
    <global offwidth="1280" offheight="960" azimuth="135" elevation="-25"/>
    <headlight active="auto"/>
  </visual>
  <statistic center="0.25 0 0.75" extent="0.9"/>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1=".4 .5 .7" rgb2=".05 .05 .1" width="32" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1=".2 .25 .3" rgb2=".15 .2 .25" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="8 8" texuniform="true" reflectance=".1"/>
    <material name="wood" rgba=".55 .4 .28 1"/>
  </asset>
  <worldbody>
    <light pos="0 -1 3" dir="0 .3 -1" diffuse=".8 .8 .8"/>
    <geom name="floor" type="plane" size="3 3 .1" material="grid"/>
    <body name="table" pos="0.25 0 0">
      <geom name="table_top" type="box" pos="0 0 .7" size=".4 .3 .02" material="wood"/>
      <geom name="leg1" type="box" pos=" .36  .26 .34" size=".02 .02 .34" material="wood"/>
      <geom name="leg2" type="box" pos="-.36  .26 .34" size=".02 .02 .34" material="wood"/>
      <geom name="leg3" type="box" pos=" .36 -.26 .34" size=".02 .02 .34" material="wood"/>
      <geom name="leg4" type="box" pos="-.36 -.26 .34" size=".02 .02 .34" material="wood"/>
    </body>
    <body name="cube" pos="0.35 0.05 0.74">
      <freejoint/>
      <geom name="cube" type="box" size=".02 .02 .02" rgba=".85 .2 .2 1" mass=".05"/>
    </body>
    <body name="ball" pos="0.30 -0.12 0.745">
      <freejoint/>
      <geom name="ball" type="sphere" size=".025" rgba=".2 .5 .9 1" mass=".03"/>
    </body>
    <camera name="overview" pos="0.95 -0.85 1.35" xyaxes="0.67 0.74 0 -0.37 0.34 0.86"/>
  </worldbody>
</mujoco>
"""

SHAPES = {"box": mujoco.mjtGeom.mjGEOM_BOX, "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
          "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE, "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
          "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID}


@dataclass
class VibeScene:
    spec: mujoco.MjSpec = field(default_factory=lambda: mujoco.MjSpec.from_string(BASE_XML))
    model: mujoco.MjModel | None = None
    data: mujoco.MjData | None = None
    log: list[str] = field(default_factory=list)
    initial_velocity: dict[str, list[float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if SO101.exists():
            arm = mujoco.MjSpec.from_file(str(SO101))
            # Attaching one spec into another is a core MjSpec workflow: the
            # arm's bodies, actuators and assets are copied in under a prefix.
            frame = self.spec.worldbody.add_frame(pos=[-0.1, 0, 0.72])
            frame.attach_body(arm.worldbody.first_body(), "so101/", "")
        self.recompile()

    # -- core -----------------------------------------------------------------
    def recompile(self) -> None:
        self.model = self.spec.compile()  # raises ValueError with MuJoCo's message on bad edits
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

    def geom(self, name: str) -> mujoco.MjsGeom:
        g = self.spec.geom(name)
        if g is None:
            raise KeyError(f"no geom named {name!r}; geoms: {self.geom_names()}")
        return g

    def body(self, name: str) -> mujoco.MjsBody:
        b = self.spec.body(name)
        if b is None:
            raise KeyError(f"no body named {name!r}")
        return b

    def geom_names(self) -> list[str]:
        return [g.name for g in self.spec.geoms if g.name and not g.name.startswith("so101/")]

    # -- summaries handed to the LLM -------------------------------------------
    def describe(self) -> str:
        m, d = self.model, self.data
        r = lambda a: [round(float(x), 3) for x in a]  # noqa: E731
        lines = [f"gravity={r(m.opt.gravity)} timestep={m.opt.timestep}",
                 f"robot arm: {'SO-101 (6 actuators, prefix so101/) at x=-0.1 on the table' if SO101.exists() else 'none'}",
                 "user objects (name, type, size, world pos, rgba, mass, friction):"]
        for name in self.geom_names():
            if name.startswith(("leg", "floor")):
                continue
            gid = m.geom(name).id
            lines.append(
                f"  {name}: {mujoco.mjtGeom(m.geom_type[gid]).name[7:].lower()} size={r(m.geom_size[gid])} "
                f"pos={r(d.geom_xpos[gid])} rgba={r(m.geom_rgba[gid])} "
                f"{'static' if not m.body_dofnum[m.geom_bodyid[gid]] else f'mass={m.body_mass[m.geom_bodyid[gid]]:.3f}'} friction={m.geom_friction[gid, 0]:.2f}")
        return "\n".join(lines)

    def simulate(self, seconds: float) -> str:
        """Roll physics forward from the current spawn state and report where things end up."""
        m = self.model
        d = mujoco.MjData(m)
        for name, vel in self.initial_velocity.items():
            if name in [m.body(i).name for i in range(m.nbody)]:
                d.joint(m.body(name).jntadr[0]).qvel[:3] = vel
        mujoco.mj_forward(m, d)
        start = {n: d.geom(n).xpos.copy() for n in self.geom_names() if m.body_dofnum[m.geom_bodyid[m.geom(n).id]]}
        while d.time < seconds:
            mujoco.mj_step(m, d)
        out = [f"after {seconds:.2f}s of simulation:"]
        for n, p0 in start.items():
            p = d.geom(n).xpos
            out.append(f"  {n}: start={[round(float(x), 3) for x in p0]} end={[round(float(x), 3) for x in p]} "
                       f"moved={np.linalg.norm(p - p0):.3f}m {'(FELL OFF TABLE)' if p[2] < 0.6 else ''}")
        self.data = d
        return "\n".join(out)
