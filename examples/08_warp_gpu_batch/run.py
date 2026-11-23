"""08 - Thousands of parallel worlds on the GPU with MuJoCo Warp (4.0.1).

MuJoCo Warp (`mujoco_warp`, imported as `mjw`) is MuJoCo re-implemented in
NVIDIA Warp. You build an ordinary `mujoco.MjModel` on the CPU, copy it to the
GPU once, then step `nworld` independent copies of the simulation in one call.

This script:
  1. benchmarks the upstream humanoid at N = 1, 64, 1024, 8192 GPU worlds,
  2. compares against the CPU: a single-threaded `mj_step` loop and the
     multi-threaded `mujoco.rollout`,
  3. writes a throughput plot and a 2x2 video of four GPU worlds (state copied
     back to the CPU and rendered with the normal `mujoco.Renderer`).

    uv run --group gpu python examples/08_warp_gpu_batch/run.py
    uv run --group gpu python examples/08_warp_gpu_batch/run.py --quick   # smoke test, < 1 min

Note: GPU numbers depend heavily on what else is running on the card. Check
`nvidia-smi` before trusting a benchmark.
"""

import os
import shutil
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import mujoco.rollout
import numpy as np
import warp as wp

import mjplay

wp.config.log_level = wp.LOG_WARNING  # silence the per-kernel "Module ... load" lines
import mujoco_warp as mjw  # noqa: E402

MODEL = mjplay.UPSTREAM_MODELS / "humanoid" / "humanoid.xml"
SEED = 0
# Per-world buffer sizes. MJWarp preallocates these (they are NOT read from the XML).
# Too small -> "overflow" warnings and dropped contacts; too big -> wasted memory/time.
# Sized from a CPU run of fallen humanoids (max ~17 contacts, ~54 constraint rows),
# with headroom because some MJWarp colliders emit more contacts than MuJoCo.
NCONMAX, NJMAX = 32, 192


def perturbed_states(mjm: mujoco.MjModel, n: int) -> tuple[np.ndarray, np.ndarray]:
    """n slightly different starting poses: noise on joint angles and velocities."""
    rng = np.random.default_rng(SEED)
    qpos = np.tile(mjm.qpos0, (n, 1))
    qpos[:, 7:] += rng.normal(0, 0.1, (n, mjm.nq - 7))  # leave the free joint's pose alone
    qvel = rng.normal(0, 0.5, (n, mjm.nv))
    return qpos, qvel


# --------------------------------------------------------------------------- GPU
def gpu_world_steps_per_s(mjm: mujoco.MjModel, nworld: int, nstep: int) -> float:
    m = mjw.put_model(mjm)                    # MjModel  -> mjw.Model (GPU arrays)
    d = mjw.make_data(mjm, nworld=nworld, nconmax=NCONMAX, njmax=NJMAX)     # nworld copies of MjData on the GPU
    qpos, qvel = perturbed_states(mjm, nworld)
    wp.copy(d.qpos, wp.array(qpos, dtype=float))
    wp.copy(d.qvel, wp.array(qvel, dtype=float))

    mjw.step(m, d)  # first call JIT-compiles the kernels (slow, once per process)

    # CUDA graph capture: record the ~100 kernel launches of one step once,
    # then replay the whole graph with a single launch. Big win for small N.
    with wp.ScopedCapture() as capture:
        mjw.step(m, d)

    for _ in range(10):  # warm-up
        wp.capture_launch(capture.graph)
    wp.synchronize()
    t0 = time.perf_counter()
    for _ in range(nstep):
        wp.capture_launch(capture.graph)
    wp.synchronize()  # kernel launches are async: wait before stopping the clock
    return nworld * nstep / (time.perf_counter() - t0)


# --------------------------------------------------------------------------- CPU
def cpu_single_thread(mjm: mujoco.MjModel, nstep: int) -> float:
    d = mujoco.MjData(mjm)
    qpos, qvel = perturbed_states(mjm, 1)
    d.qpos[:], d.qvel[:] = qpos[0], qvel[0]
    t0 = time.perf_counter()
    for _ in range(nstep):
        mujoco.mj_step(mjm, d)
    return nstep / (time.perf_counter() - t0)


def cpu_rollout(mjm: mujoco.MjModel, nbatch: int, nstep: int) -> float:
    """mujoco.rollout: nbatch trajectories spread over a C++ thread pool."""
    nthread = os.cpu_count() or 1
    datas = [mujoco.MjData(mjm) for _ in range(nthread)]
    qpos, qvel = perturbed_states(mjm, nbatch)
    spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    init = np.zeros((nbatch, mujoco.mj_stateSize(mjm, spec)))
    scratch = mujoco.MjData(mjm)
    for i in range(nbatch):
        scratch.qpos[:], scratch.qvel[:] = qpos[i], qvel[i]
        mujoco.mj_getState(mjm, scratch, init[i], spec)
    mujoco.rollout.rollout(mjm, datas, init, nstep=2, persistent_pool=True)  # warm the pool
    t0 = time.perf_counter()
    mujoco.rollout.rollout(mjm, datas, init, nstep=nstep, persistent_pool=True)
    elapsed = time.perf_counter() - t0
    mujoco.rollout.shutdown_persistent_pool()
    return nbatch * nstep / elapsed


# ------------------------------------------------------------------------- video
def render_gpu_worlds(mjm: mujoco.MjModel, nworld: int, seconds: float, args) -> None:
    """Simulate on the GPU, pull qpos of 4 worlds back each frame, render on the CPU."""
    m = mjw.put_model(mjm)
    d = mjw.make_data(mjm, nworld=nworld, nconmax=NCONMAX, njmax=NJMAX)
    qpos, qvel = perturbed_states(mjm, nworld)
    qvel[:, :3] += np.random.default_rng(1).normal(0, 1.0, (nworld, 3))  # shove them around a bit
    wp.copy(d.qpos, wp.array(qpos, dtype=float))
    wp.copy(d.qvel, wp.array(qvel, dtype=float))
    with wp.ScopedCapture() as capture:
        mjw.step(m, d)

    show = np.linspace(0, nworld - 1, 4).astype(int)  # which worlds to film
    cpu = mujoco.MjData(mjm)
    fps, w, h = 30, args.width // 2, args.height // 2
    steps_per_frame = round(1 / (fps * mjm.opt.timestep))
    cam = mujoco.MjvCamera()
    tiles = [[] for _ in show]
    with mujoco.Renderer(mjm, height=h, width=w) as r:
        for _ in range(int(seconds * fps)):
            for _ in range(steps_per_frame):
                wp.capture_launch(capture.graph)
            all_qpos = d.qpos.numpy()  # (nworld, nq): one device->host copy per frame
            for k, world in enumerate(show):
                cpu.qpos[:] = all_qpos[world]
                mujoco.mj_forward(mjm, cpu)  # kinematics only, to place the geoms
                cam.lookat[:] = cpu.qpos[:3]
                cam.distance, cam.azimuth, cam.elevation = 3.5, 90 + 30 * k, -15
                r.update_scene(cpu, camera=cam)
                tiles[k].append(r.render())
    labelled = [mjplay.label_frames(t, f"world {wid} / {nworld}") for t, wid in zip(tiles, show)]
    top = mjplay.side_by_side(labelled[0], labelled[1])
    bottom = mjplay.side_by_side(labelled[2], labelled[3])
    mjplay.save_video([np.concatenate(fb, axis=0) for fb in zip(top, bottom)], "08_gpu_worlds.mp4", fps=fps)


def main() -> None:
    args = mjplay.example_args(__doc__, seconds=3.0, quick="tiny config for smoke tests")
    if args.viewer:
        # MJWarp ships its own interactive viewer that steps on the GPU.
        # Resolve next to this interpreter so it works with or without an activated venv.
        exe = os.path.join(os.path.dirname(sys.executable), "mjwarp-viewer")
        exe = exe if os.path.exists(exe) else shutil.which("mjwarp-viewer")
        if exe is None:
            sys.exit("mjwarp-viewer not found: run `uv sync --group gpu` (it ships with mujoco-warp)")
        os.execv(exe, [exe, str(MODEL)])
    mjm =mujoco.MjModel.from_xml_path(str(MODEL))
    print(f"model: humanoid  nq={mjm.nq} nv={mjm.nv} ngeom={mjm.ngeom}  dt={mjm.opt.timestep}")
    print(f"warp device: {wp.get_device()}  ({wp.get_device().name})")

    worlds = [1, 64] if args.quick else [1, 64, 1024, 8192]
    nstep = 50 if args.quick else 300
    cpu_nstep = 500 if args.quick else 3000

    results = {}
    for n in worlds:
        try:
            results[f"warp N={n}"] = gpu_world_steps_per_s(mjm, n, nstep)
        except Exception as e:  # e.g. out of GPU memory: report and keep going
            print(f"warp N={n}: failed ({type(e).__name__}: {e})")
    results["cpu mj_step (1 thread)"] = cpu_single_thread(mjm, cpu_nstep)
    nbatch = 64 if args.quick else 1024
    results[f"cpu rollout ({os.cpu_count()} threads, {nbatch} traj)"] = cpu_rollout(
        mjm, nbatch, 50 if args.quick else 500)

    base = results["cpu mj_step (1 thread)"]
    print(f"\n{'backend':<38}{'world-steps/s':>16}{'x 1 CPU thread':>16}")
    for name, sps in results.items():
        print(f"{name:<38}{sps:>16,.0f}{sps / base:>15.1f}x")
    rt = max(results.values()) * mjm.opt.timestep
    print(f"\nbest: {rt:,.0f} simulated seconds per wall-clock second")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = list(results)
    colors = ["tab:green" if n.startswith("warp") else "tab:gray" for n in names]
    ax.barh(names, [results[n] for n in names], color=colors)
    ax.set_xscale("log")
    ax.set_xlabel("world-steps per second (log scale)")
    ax.set_title("Humanoid: MuJoCo Warp (GPU) vs CPU")
    ax.invert_yaxis()
    fig.tight_layout()
    path = mjplay.output_path("08_throughput.png")
    fig.savefig(path, dpi=120)
    print(f"wrote outputs/{path.name}")

    render_gpu_worlds(mjm, 64 if args.quick else 1024, 1.0 if args.quick else args.seconds, args)

    assert results[f"warp N={worlds[-1]}"] > results["warp N=1"], "batching should raise throughput"
    print("OK")


if __name__ == "__main__":
    main()
