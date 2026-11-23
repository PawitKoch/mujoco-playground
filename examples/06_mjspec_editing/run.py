"""06 - Procedural models with MjSpec, plus the new 4.0 model/data fields.

Covers, in order:
  1. Build a scene in Python with mujoco.MjSpec: floor, a DOF-free conveyor belt
     (geom `surfacevel`), boxes, a pendulum, a fixed and a spatial tendon.
  2. Merge in a flexcomp cloth and give it a per-ELEMENT material list
     (`flex.material = [...]`), new in 4.0 (it used to be one material per flex).
  3. Compile, simulate, render a video.
  4. Read the new 4.0 fields: mjData.xiquat / geom_xquat / site_xquat,
     mjModel.tendon_type, mjContact.surfacevel.
  5. Serialise to MJB with MjSpec.encode (MJB output for specs is new in 4.0),
     reload with MjModel.from_binary_path and check that the model is identical.

    uv run python examples/06_mjspec_editing/run.py
    uv run python examples/06_mjspec_editing/run.py --viewer
"""

import mujoco
import numpy as np

import mjplay

BELT_SPEED = 0.4  # m/s

# A flexcomp is an MJCF *macro* (the parser expands it into a flex plus bodies),
# so the cloth starts life as an XML fragment and everything else is added in Python.
CLOTH_XML = """
<mujoco>
  <worldbody>
    <flexcomp name="cloth" type="grid" count="9 9 1" spacing=".05 .05 .05" pos="0 .55 .9"
              dim="2" radius=".006" mass=".2">
      <edge equality="true"/>
      <pin id="0 72"/>
    </flexcomp>
  </worldbody>
</mujoco>
"""


def build_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_string(CLOTH_XML)
    spec.modelname = "procedural_playground"
    spec.option.timestep = 0.002

    # --- assets -------------------------------------------------------------
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[.4, .5, .65], rgb2=[.05, .05, .1], width=32, height=512)
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[.2, .25, .3], rgb2=[.25, .3, .35], width=512, height=512)
    floor_mat = spec.add_material(name="floor", texrepeat=[4, 4], texuniform=True)
    floor_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    spec.add_material(name="red", rgba=[.85, .2, .2, 1])
    spec.add_material(name="cream", rgba=[.95, .9, .8, 1])

    world = spec.worldbody
    world.add_light(pos=[0, -1, 3], dir=[0, .3, -1], diffuse=[.8, .8, .8])
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, .05], material="floor")

    # --- conveyor: a static box whose *surface* moves -------------------------
    # surfacevel = [vx vy vz wx wy wz] in the geom frame. No joints, no actuator.
    world.add_geom(name="belt", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, -.3, .1], size=[.8, .15, .02],
                   rgba=[.15, .15, .17, 1], surfacevel=[BELT_SPEED, 0, 0, 0, 0, 0])
    world.add_site(name="anchor", pos=[.4, -.3, .6], size=[.015, 0, 0], rgba=[1, 1, 0, 1])

    # --- parcels riding the belt (loop: add bodies programmatically) ---------
    for i, colour in enumerate([[.9, .6, .2, 1], [.3, .7, .9, 1], [.6, .9, .3, 1]]):
        b = world.add_body(name=f"parcel{i}", pos=[-.6 + .2 * i, -.3, .17])
        b.add_freejoint()
        b.add_geom(name=f"parcel{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[.04, .04, .04], rgba=colour, mass=.2)
        b.add_site(name=f"parcel{i}_top", pos=[0, 0, .04], size=[.01, 0, 0])

    # --- a pendulum with a fixed tendon on its hinge -------------------------
    pend = world.add_body(name="pendulum", pos=[-.5, .2, .8])
    hinge = pend.add_joint(name="swing", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], damping=.01)
    pend.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[0, 0, 0, .25, 0, 0], size=[.015, 0, 0],
                  rgba=[.8, .8, .85, 1])
    pend.add_site(name="tip", pos=[.25, 0, 0], size=[.02, 0, 0])

    fixed = spec.add_tendon(name="swing_fixed")
    fixed.wrap_joint(hinge.name, 1.0)

    # --- a spatial tendon (elastic band) from the anchor to parcel 2 ---------
    band = spec.add_tendon(name="band", stiffness=4.0, springlength=[.5, .5], width=.004,
                           rgba=[1, .9, .2, 1])
    band.wrap_site("anchor")
    band.wrap_site("parcel2_top")

    world.add_camera(name="view", pos=[1.4, -1.6, 1.1], xyaxes=[.75, .66, 0, -.25, .3, .92])
    return spec


def colour_cloth(spec: mujoco.MjSpec, model: mujoco.MjModel) -> None:
    """Checkerboard the cloth: one material name PER ELEMENT (triangle)."""
    flex = spec.flex("cloth")
    nelem = model.flex_elemnum[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, "cloth")]
    cells_per_row = 8  # 9x9 vertices -> 8x8 quads -> 2 triangles each
    names = []
    for e in range(nelem):
        row, col = divmod(e // 2, cells_per_row)
        names.append("red" if (row + col) % 2 else "cream")
    flex.material = names  # NEW in 4.0: MjStringVec, one entry per element (or one broadcast)


def main() -> None:
    args = mjplay.example_args(__doc__, seconds=3.0)

    spec = build_spec()
    model = spec.compile()  # first compile tells us the element count
    colour_cloth(spec, model)
    model = spec.compile()
    data = mujoco.MjData(model)
    print(f"compiled: nbody={model.nbody} nq={model.nq} nv={model.nv} ngeom={model.ngeom} "
          f"nflex={model.nflex} nflexelem={model.nflexelem}")
    print(f"flex_matid is per element now: shape {model.flex_matid.shape}, "
          f"first 6 = {model.flex_matid[:6].tolist()}")

    if args.viewer:
        mjplay.view(model, data)
        return

    frames = mjplay.render_rollout(model, data, args.seconds, camera="view",
                                   width=args.width, height=args.height)
    mjplay.save_video(frames, "06_procedural_scene.mp4")

    # ---- new mjData / mjModel fields -----------------------------------------
    b = model.body("pendulum").id
    print("\nnew 4.0 fields (quaternion twins of the existing *_xmat arrays):")
    print(f"  xiquat[pendulum]     = {np.round(data.xiquat[b], 4)}")
    print(f"  geom_xquat[parcel0]  = {np.round(data.geom_xquat[model.body('parcel0').geomadr[0]], 4)}")
    print(f"  site_xquat[tip]      = {np.round(data.site_xquat[model.site('tip').id], 4)}")
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, data.geom_xmat[model.body('parcel0').geomadr[0]])
    assert np.allclose(np.abs(q @ data.geom_xquat[model.body('parcel0').geomadr[0]]), 1, atol=1e-6)

    types = {int(v): k for k, v in mujoco.mjtTendon.__members__.items()}
    for t in range(model.ntendon):
        print(f"  tendon_type[{model.tendon(t).name:12s}] = {types[int(model.tendon_type[t])]}")

    belt = model.geom("belt").id
    belt_contacts = [c for c in data.contact[:data.ncon] if belt in (c.geom1, c.geom2)]
    print(f"\ncontacts touching the belt: {len(belt_contacts)}  (mjModel.flg_surfacevel={model.flg_surfacevel})")
    for c in belt_contacts[:3]:
        print(f"  {model.geom(c.geom1).name or c.geom1}->{model.geom(c.geom2).name or c.geom2}: "
              f"surfacevel (t1, t2, spin) = {np.round(c.surfacevel, 3)}")
    for i in range(3):
        dof = model.jnt_dofadr[model.body(f"parcel{i}").jntadr[0]]
        print(f"  parcel{i} x-velocity = {data.qvel[dof]:+.3f} m/s  (belt surface runs at {BELT_SPEED} m/s)")
    assert any(np.linalg.norm(c.surfacevel[:2]) > 0.3 for c in belt_contacts), "belt surface velocity not in contacts"

    # ---- MJB round trip -------------------------------------------------------
    mjb = mjplay.output_path("06_procedural_scene.mjb")
    nbytes = spec.encode(str(mjb), model, content_type="model/vnd.mujoco.mjb")
    print(f"\nencoded {nbytes} bytes -> {mjb.name}")
    reloaded = mujoco.MjModel.from_binary_path(str(mjb))
    for field in ("nq", "nv", "nbody", "ngeom", "nflex", "nflexelem", "ntendon"):
        a, r = getattr(model, field), getattr(reloaded, field)
        assert a == r, f"{field}: {a} != {r}"
    assert np.array_equal(model.flex_matid, reloaded.flex_matid)
    print("OK: MJB reload has identical sizes and per-element flex materials")
    xml = mjplay.output_path("06_procedural_scene.xml")
    xml.write_text(spec.to_xml())
    print(f"also saved the spec as MJCF: {xml.name} (open it to see what your Python built)")


if __name__ == "__main__":
    main()
