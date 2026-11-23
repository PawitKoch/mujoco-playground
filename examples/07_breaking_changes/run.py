"""07 - Two breaking changes in MuJoCo 4.0 that silently change results.

A) `sidesite` tendon wrapping uses a new rule. The tendon follows the path
   p1 -> side site -> p2 (the shortest path with the same winding around the geom).
   Its length is continuous everywhere except ONE place: where an endpoint
   crosses the ray opposite the side site. The script sweeps an arm through
   360 deg, plots tendon length and finds that jump.

B) `RK4` is now truly 4th-order for ball/free joint orientations. Before 4.0
   the orientation update dropped the Runge-Kutta-Munthe-Kaas correction, so it
   converged at 2nd order. The script runs a timestep convergence study on a
   tumbling free body and fits the order (expect ~4 for RK4, ~1 for Euler).

    uv run python examples/07_breaking_changes/run.py
    uv run python examples/07_breaking_changes/run.py --viewer   # watch the handle flip
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import mjplay  # noqa: E402

HERE = Path(__file__).parent


# --------------------------------------------------------------------------- A
def sidesite_sweep(args) -> None:
    model = mujoco.MjModel.from_xml_path(str(HERE / "sidesite.xml"))
    data = mujoco.MjData(model)
    thetas = np.linspace(0, 360, 721)
    lengths = []
    for th in thetas:
        data.qpos[0] = np.radians(th)
        mujoco.mj_forward(model, data)  # kinematics + tendon lengths, no dynamics needed
        lengths.append(data.ten_length[0])
    lengths = np.array(lengths)

    jumps = np.abs(np.diff(lengths))
    big = np.flatnonzero(jumps > 0.05)
    print("A) sidesite sweep")
    print(f"   tendon length range {lengths.min():.3f} .. {lengths.max():.3f} m")
    for i in big:
        print(f"   discontinuity between theta={thetas[i]:.1f} and {thetas[i+1]:.1f} deg: "
              f"{lengths[i]:.3f} -> {lengths[i+1]:.3f} m")
    # p2 crosses the +z ray (opposite the side site at -z) at theta = 270 deg.
    assert len(big) == 1 and abs(thetas[big[0]] - 270) < 1, "expected exactly one snap, at 270 deg"
    print("   OK: exactly one snap, where p2 crosses the ray opposite the side site (theta=270)")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(thetas, lengths, lw=2)
    ax.axvline(270, color="g", ls="--", lw=1, label="p2 crosses ray opposite sidesite")
    ax.set(xlabel="arm angle theta [deg]", ylabel="tendon length [m]",
           title="sidesite wrapping (MuJoCo 4): one discontinuity")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(mjplay.output_path("07_sidesite_length.png"), dpi=120)
    print("   wrote outputs/07_sidesite_length.png")

    # Video: sweep the arm kinematically so you can see the tendon flip sides.
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, 0]
    cam.distance, cam.azimuth, cam.elevation = 1.3, 90, 0
    frames = []
    with mujoco.Renderer(model, args.height, args.width) as r:
        for th in np.linspace(0, 360, 180):
            data.qpos[0] = np.radians(th)
            mujoco.mj_forward(model, data)
            r.update_scene(data, camera=cam)
            frames.append(r.render())
    labels = [mjplay.label_frames([f], f"theta={th:5.1f} deg  L={lengths[int(th * 2)]:.3f} m")[0]
              for f, th in zip(frames, np.linspace(0, 360, 180))]
    mjplay.save_video(labels, "07_sidesite_sweep.mp4")


# --------------------------------------------------------------------------- B
def spin(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Spin about the intermediate principal axis plus a tiny perturbation.

    A free joint's angular velocity is in the body frame, and its linear velocity
    is that of the body ORIGIN, so the origin must move as -w x (com - origin)
    for the centre of mass to stay put.
    """
    # principal axes of the body inertia, expressed in the body frame
    axes = np.zeros(9)
    mujoco.mju_quat2Mat(axes, model.body_iquat[1])
    axes = axes.reshape(3, 3)
    mid = int(np.argsort(model.body_inertia[1])[1])  # intermediate moment
    w_local = 10.0 * axes[:, mid] + 0.02 * axes[:, (mid + 1) % 3]
    mujoco.mj_forward(model, data)
    w_world = data.xmat[1].reshape(3, 3) @ w_local
    data.qvel[:3] = -np.cross(w_world, data.xipos[1] - data.xpos[1])
    data.qvel[3:] = w_local


def simulate(integrator: int, dt: float, T: float) -> np.ndarray:
    model = mujoco.MjModel.from_xml_path(str(HERE / "tumble.xml"))
    model.opt.integrator = integrator
    model.opt.timestep = dt
    data = mujoco.MjData(model)
    spin(model, data)
    for _ in range(int(round(T / dt))):
        mujoco.mj_step(model, data)
    return data.qpos[3:7].copy()


def quat_angle(q1: np.ndarray, q2: np.ndarray) -> float:
    # mju_subQuat returns the rotation vector taking q2 to q1; it stays accurate
    # for tiny angles, where 2*arccos(q1.q2) would round to zero.
    v = np.zeros(3)
    mujoco.mju_subQuat(v, q1, q2)
    return float(np.linalg.norm(v))


def rk4_convergence() -> None:
    T = 1.0
    print("\nB) orientation convergence on a tumbling free body (T = 1 s)")
    ref = simulate(mujoco.mjtIntegrator.mjINT_RK4, 1e-4, T)
    dts = np.array([0.02, 0.01, 0.005, 0.0025])
    results = {}
    for name, integ in [("RK4", mujoco.mjtIntegrator.mjINT_RK4), ("Euler", mujoco.mjtIntegrator.mjINT_EULER)]:
        errs = np.array([quat_angle(simulate(integ, dt, T), ref) for dt in dts])
        order = np.polyfit(np.log(dts), np.log(errs), 1)[0]
        results[name] = (errs, order)
        print(f"   {name:5s} errors [rad]: " + "  ".join(f"dt={dt:<7g}{e:.2e}" for dt, e in zip(dts, errs)))
        print(f"   {name:5s} fitted order: {order:.2f}")
    assert results["RK4"][1] > 3.5, "RK4 should be ~4th order for free-joint orientation in MuJoCo 4"
    print("   OK: RK4 is 4th order on orientation (pre-4.0 it was ~2 for ball/free joints)")

    fig, ax = plt.subplots(figsize=(5, 4))
    for name, (errs, order) in results.items():
        ax.loglog(dts, errs, "o-", label=f"{name} (order {order:.2f})")
    ax.loglog(dts, results["RK4"][0][0] * (dts / dts[0]) ** 2, "k:", lw=1,
              label="slope 2 (old RK4 orientation)")
    ax.set(xlabel="timestep [s]", ylabel="orientation error [rad]", title="free-joint orientation convergence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(mjplay.output_path("07_rk4_convergence.png"), dpi=120)
    print("   wrote outputs/07_rk4_convergence.png")


def main() -> None:
    args = mjplay.example_args(__doc__, seconds=8.0)
    if args.viewer:
        model = mujoco.MjModel.from_xml_path(str(HERE / "tumble.xml"))
        data = mujoco.MjData(model)
        spin(model, data)
        mjplay.view(model, data)
        return
    sidesite_sweep(args)
    rk4_convergence()

    model = mujoco.MjModel.from_xml_path(str(HERE / "tumble.xml"))
    data = mujoco.MjData(model)
    spin(model, data)
    frames = mjplay.render_rollout(model, data, args.seconds, camera="cam", width=args.width, height=args.height)
    mjplay.save_video(mjplay.label_frames(frames, "Dzhanibekov flip, RK4 dt=10 ms"), "07_tumbling_handle.mp4")


if __name__ == "__main__":
    main()
