"""00 - Hello, MuJoCo 4: check the install and see what ships in the box.

Prints:
  * the engine / package versions (4.0 split `mujoco` into mujoco-core + mujoco-studio)
  * the manual and agent skills that 4.0 bundles inside the wheel (mujoco/doc/)
  * the new enable flags and mjData fields you will meet in the other examples
and renders one still to outputs/00_hello.png.

    uv run python examples/00_hello_mujoco4/run.py
"""

import importlib.metadata as md
from pathlib import Path

import mujoco
import numpy as np

import mjplay

XML = """
<mujoco model="hello">
  <visual><headlight active="auto"/></visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1=".4 .5 .7" rgb2="0 0 0" width="32" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 .2 .3" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="6 6" texuniform="true" reflectance=".2"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="2 2 .1" material="grid"/>
    <body pos="0 0 .5" euler="20 30 0">
      <freejoint/>
      <geom name="box" type="box" size=".1 .15 .05" rgba=".9 .4 .1 1"/>
      <site name="tip" pos=".1 .15 .05"/>
    </body>
    <!-- castshadow is a 4.0 geom attribute: this floating sphere casts no shadow -->
    <geom type="sphere" pos=".4 0 .4" size=".08" rgba=".2 .7 .9 1" castshadow="false" contype="0" conaffinity="0"/>
  </worldbody>
</mujoco>
"""


def main() -> None:
    print(f"MuJoCo engine   : {mujoco.mj_versionString()}")
    for pkg in ("mujoco", "mujoco-core", "mujoco-studio", "mujoco-warp", "mujoco-mjx", "jax", "google-genai"):
        try:
            print(f"  {pkg:<14}: {md.version(pkg)}")
        except md.PackageNotFoundError:
            print(f"  {pkg:<14}: (not installed)")

    doc = Path(mujoco.__file__).parent / "doc"
    skills = sorted(p.parent.name for p in (doc / "skills").glob("*/SKILL.md"))
    print(f"\nBundled manual  : {len(list(doc.rglob('*.rst')))} .rst files under {doc}")
    print(f"Agent skills    : {', '.join(skills)}")
    print("  (these are the same skills the Antigravity Robotics plugin gives Gemini - see example 09)")

    print("\nEnable flags    :", ", ".join(n for n in dir(mujoco.mjtEnableBit) if n.startswith("mjENBL_")))
    print("New mjtState bit:", "mjSTATE_PLASTIC" in dir(mujoco.mjtState))

    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    # New in 4.0: quaternion versions of the rotation matrices live next to them in mjData.
    print("\ngeom_xmat[box]  :", np.round(data.geom("box").xmat.reshape(3, 3), 3).tolist())
    print("geom_xquat[box] :", np.round(data.geom("box").xquat, 4))
    print("geom_castshadow :", model.geom_castshadow.tolist())
    print("nplastic        :", model.nplastic, "(size of mjData.plastic; 0 here - see example 02)")
    mjplay.save_image(mjplay.render_still(model, data), "00_hello.png")


if __name__ == "__main__":
    main()
