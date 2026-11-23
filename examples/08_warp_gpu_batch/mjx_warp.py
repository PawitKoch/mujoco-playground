"""08b - The same batched humanoid through MJX, with impl="jax" vs impl="warp".

MJX is MuJoCo's JAX API. Since 4.x it fronts two implementations:
  * impl="jax"  - pure JAX/XLA. Runs on CPU/GPU/TPU and is differentiable (jax.grad).
  * impl="warp" - calls MuJoCo Warp kernels through the JAX FFI. NVIDIA-only, not
                  differentiable, but much faster for contact-rich scenes.
Your code is identical either way: put_model / make_data / jax.vmap(mjx.step).

With impl="warp" the CUDA graph capture mode matters (see doc/mjx.rst
"MJX-Warp graph modes"): JAX may move buffers between calls, which forces Warp
to re-capture its graph. WARP_STAGED copies through fixed staging buffers instead.

    uv run --group gpu python examples/08_warp_gpu_batch/mjx_warp.py
    uv run --group gpu python examples/08_warp_gpu_batch/mjx_warp.py --quick
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # don't grab 75% of VRAM up front

import time

import jax
import jax.numpy as jp
import mujoco
import numpy as np
import warp as wp
from mujoco import mjx

import mjplay

wp.config.log_level = wp.LOG_WARNING

MODEL = mjplay.UPSTREAM_MODELS / "humanoid" / "humanoid.xml"
NCONMAX, NJMAX = 32, 192  # per-world sizes; see run.py for how these were chosen


def batched_data(mjm: mujoco.MjModel, impl: str, n: int) -> mjx.Data:
    kw = {"naconmax": NCONMAX * n, "njmax": NJMAX} if impl == "warp" else {}  # naconmax is for ALL worlds
    d0 = mjx.make_data(mjm, impl=impl, **kw)
    rng = np.random.default_rng(0)
    qpos = np.tile(mjm.qpos0, (n, 1))
    qpos[:, 7:] += rng.normal(0, 0.1, (n, mjm.nq - 7))
    # vmap over a function that replaces qpos -> a Data with a leading batch axis.
    return jax.vmap(lambda q: d0.replace(qpos=q))(jp.asarray(qpos))


def bench(mjm: mujoco.MjModel, impl: str, n: int, nstep: int, graph_mode=None) -> float:
    m = mjx.put_model(mjm, impl=impl, **({"graph_mode": graph_mode} if graph_mode is not None else {}))
    d = batched_data(mjm, impl, n)
    step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))  # model shared, data batched

    def run(d):
        for _ in range(nstep):
            d = step(m, d)
        return d

    t0 = time.perf_counter()
    d = jax.block_until_ready(run(d))  # includes JIT compile
    compile_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    d = jax.block_until_ready(run(d))
    sps = n * nstep / (time.perf_counter() - t0)
    print(f"  {impl:5s} {getattr(graph_mode, 'name', ''):12s} first call {compile_s:5.1f}s   "
          f"{sps:>12,.0f} world-steps/s   mean torso z {float(d.qpos[:, 2].mean()):.2f} m")
    return sps


def main() -> None:
    args = mjplay.example_args(__doc__, quick="tiny config for smoke tests")
    devices = jax.devices()
    print(f"JAX devices: {devices}")
    if devices[0].platform != "gpu":
        print("JAX sees no GPU. Install the gpu group (`uv sync --group gpu`, which pulls jax[cuda12]) "
              "and check `nvidia-smi`. impl='warp' needs an NVIDIA GPU; skipping.")
        return

    mjm = mujoco.MjModel.from_xml_path(str(MODEL))
    n, nstep = (64, 20) if args.quick else (1024, 100)
    print(f"humanoid x {n} worlds, {nstep} steps")
    res = {
        "jax": bench(mjm, "jax", n, nstep),
        "warp/WARP": bench(mjm, "warp", n, nstep, wp.JaxCallableGraphMode.WARP),
        "warp/WARP_STAGED": bench(mjm, "warp", n, nstep, wp.JaxCallableGraphMode.WARP_STAGED),
    }
    best = max(res["warp/WARP"], res["warp/WARP_STAGED"])
    print(f"MJX-Warp is {best / res['jax']:.0f}x faster than MJX-JAX here. "
          "Pure mujoco_warp with a captured graph (run.py) is faster still: no JAX FFI hop.")


if __name__ == "__main__":
    main()
