"""04 - Sparse Newton on flex models + the experimental `ipc` contact mode (MuJoCo 4.0).

What changed in 4.0 for deformables:
* `solver="Newton"` got a fill-reducing DOF ordering, incremental Hessian updates and a
  pre-ordered sparse Cholesky for the `discrete` integrator: 1.4-3x faster than the 3.x
  Newton on flex models. Most upstream flex models now default to Newton.
* Newton + `discrete` now handles flex vertices attached to articulated bodies.
* The experimental `<flag ipc="enable"/>` (discrete integrator only) resolves flex
  contact with an augmented-Lagrangian, IPC-style solve; it now accepts Newton as
  well as CG.

This script runs ONE cloth scene with three flex-contact modes (passive penalty,
constraint, ipc) x two solvers (CG, Newton) and prints wall-clock cost per simulated
second, mean solver iterations per step, and how close the cloth gets to the ball.
Then it renders Newton vs Newton+ipc side by side.

Note: the 1.4-3x figure is Newton(4.0) vs Newton(3.x). Against CG the answer is
model-dependent - measure on YOUR model, which is the point of the table.

    uv run python examples/04_newton_flex_ipc/run.py
    uv run python examples/04_newton_flex_ipc/run.py --viewer          # Newton, ipc off
    uv run python examples/04_newton_flex_ipc/run.py --viewer --ipc
"""

import time
from pathlib import Path

import mujoco
import numpy as np

import mjplay

HERE = Path(__file__).parent
SOLVERS = {"CG": mujoco.mjtSolver.mjSOL_CG, "Newton": mujoco.mjtSolver.mjSOL_NEWTON}
IPC = mujoco.mjtEnableBit.mjENBL_IPC
BENCH_SECONDS = 1.0


def load(solver: str, ipc: bool, passive: bool = True) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(str(HERE / "scene.xml"))
    model.opt.solver = SOLVERS[solver]
    if ipc:
        model.opt.enableflags |= IPC
    # passive=True (as in scene.xml): flex contact is a penalty force folded into the
    # discrete integrator - no constraint rows, so the solver has nothing to do.
    # passive=False: flex contacts become constraints and the solver does the work.
    model.flex_passive[:] = passive
    return model, mujoco.MjData(model)


def min_ball_gap(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Smallest distance from a cloth vertex (mid-surface) to the ball surface, metres."""
    ball = model.geom("ball").id
    centre, radius = data.geom_xpos[ball], model.geom_size[ball, 0]
    return float((np.linalg.norm(data.flexvert_xpos - centre, axis=1) - radius).min())


CASES = [  # (contact mode, solver, ipc)
    ("passive", "CG", False),
    ("passive", "Newton", False),
    ("constraint", "CG", False),
    ("constraint", "Newton", False),
    ("ipc", "CG", True),
    ("ipc", "Newton", True),
]


def benchmark() -> dict[tuple[str, str], float]:
    """Print the comparison table; return the min ball gap per (mode, solver)."""
    print(f"\n{'contact':10s} {'solver':7s} {'wall s/sim s':>12s} {'iters/step':>10s} {'min vertex-ball gap':>20s}")
    results = {}
    for mode, solver, ipc in CASES:
        model, data = load(solver, ipc, passive=(mode != "constraint"))
        iters, gap = [], np.inf
        tic = time.perf_counter()
        while data.time < BENCH_SECONDS:
            mujoco.mj_step(model, data)
            iters.append(int(data.solver_niter[: max(1, data.nisland)].max()))
            gap = min(gap, min_ball_gap(model, data))
        cost = (time.perf_counter() - tic) / data.time
        results[mode, solver] = gap
        print(f"{mode:10s} {solver:7s} {cost:12.2f} {np.mean(iters):10.1f} {gap * 1000:17.2f} mm")
    print("(gap < 0: the cloth mid-surface is inside the ball; the cloth radius is 4 mm)")
    return results


def render_comparison(args) -> None:
    clips = []
    for ipc in (False, True):
        model, data = load("Newton", ipc)
        frames = mjplay.render_rollout(model, data, seconds=args.seconds, camera="front",
                                       width=args.width, height=args.height)
        clips.append(mjplay.label_frames(frames, f"Newton, ipc={'on' if ipc else 'off'}"))
    mjplay.save_video(mjplay.side_by_side(*clips), "04_cloth_newton_ipc_off_vs_on.mp4")


def main() -> None:
    args = mjplay.example_args(__doc__, seconds=2.5, ipc="enable the ipc flag in --viewer mode",
                               cg="use solver=CG in --viewer mode")
    if args.viewer:
        model, data = load("CG" if args.cg else "Newton", args.ipc)
        mjplay.view(model, data)
        return
    print(f"nv = {load('Newton', False)[0].nv} (two 17x17 cloths, 3 DOFs per vertex)")
    gaps = benchmark()
    assert gaps["ipc", "Newton"] > 0, "ipc should keep the cloth mid-surface outside the ball"
    print("OK: with ipc the cloth never enters the ball (it keeps the ~1 mm rest gap).")
    render_comparison(args)


if __name__ == "__main__":
    main()
