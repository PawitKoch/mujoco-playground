# 08 · Batched GPU simulation: MuJoCo Warp and MJX

```bash
uv sync --group gpu                                                # mujoco-warp[cuda], mujoco-mjx[warp], jax[cuda12]
uv run python examples/08_warp_gpu_batch/run.py                    # MJWarp vs CPU benchmark + video
uv run python examples/08_warp_gpu_batch/mjx_warp.py               # MJX impl="jax" vs impl="warp"
uv run python examples/08_warp_gpu_batch/run.py --viewer           # launches mjwarp-viewer (GPU-stepped viewer)
```

Outputs: `outputs/08_throughput.png`, `outputs/08_gpu_worlds.mp4` (four of 1024 GPU worlds, copied back to the CPU each frame and rendered).

## The three ways to run many simulations

| | runs on | API | differentiable | best for |
|---|---|---|---|---|
| `mujoco.rollout` | CPU threads | numpy in, numpy out | no | MPC, sysid, moderate batches, anything latency-sensitive |
| **MuJoCo Warp** (`import mujoco_warp as mjw`) | NVIDIA GPU | Warp arrays (zero-copy to PyTorch) | no | RL at 1k–10k+ worlds, contact-rich scenes |
| **MJX** (`from mujoco import mjx`) | GPU/TPU/CPU via JAX | JAX pytrees, `jax.vmap` | yes with `impl="jax"` | JAX training stacks; `impl="warp"` gives Warp's speed behind the same JAX API |

MJWarp is built for **throughput**, not latency. One world on the GPU is about 40× *slower* than one CPU thread, because every step launches about 100 tiny kernels. It overtakes the CPU at somewhere around 64–1024 worlds.

## What the code teaches

`run.py`:
- `mjw.put_model(mjm)` and `mjw.make_data(mjm, nworld=N, nconmax=…, njmax=…)` copy the model and data to the GPU. **Contact and constraint buffers are fixed-size, and their sizes are not read from the XML.** If they're too small you get `nefc overflow` warnings and dropped contacts. The script picks its sizes by measuring a CPU run first; try `njmax=64` to see the warning.
- `wp.ScopedCapture()` followed by `wp.capture_launch(graph)` records one step as a CUDA graph and replays it with a single launch.
- `wp.synchronize()` must run before you stop the timer. Launches are asynchronous, so without it you only measure how long the launch calls took.
- `d.qpos.numpy()` copies everything from GPU to host. Do it once per rendered frame, not once per step.

`mjx_warp.py`:
- `mjx.put_model(mjm, impl="warp", graph_mode=wp.JaxCallableGraphMode.WARP_STAGED)`. The bundled `doc/mjx.rst` spells this `mjxw.GraphMode`, but in 4.0.1 the enum is `warp.JaxCallableGraphMode`.
- `naconmax` here counts contacts **across all worlds**, so scale it by your `vmap` batch size.
- Contacts live in the private `d._impl`. To read contacts in MJX-Warp, use contact sensors.

## MuJoCo 4.0 notes
- Flex models whose vertices are attached to articulated bodies need `solver="CG"` under MuJoCo Warp. CPU MuJoCo 4.0 now also handles these with `Newton` plus the `discrete` integrator (changelog 4.0.0 #19).
- `mujoco-warp` and `mujoco-mjx` now share MuJoCo's version number (`4.0.1`). `mujoco-warp` depends only on `mujoco-core`, the numpy-only half of the new package split.
- Install MJX with the Warp backend using `pip install mujoco-mjx[warp]`.

## Things to try
1. Run `uv run mjwarp-testspeed third_party/mujoco/model/humanoid/humanoid.xml --nworld=8192 --measure_alloc`. It shows the real contact and constraint counts, so you can tune `nconmax`/`njmax`.
2. Swap `MODEL` for `third_party/mujoco/model/humanoid/22_humanoids.xml` (high `nv`) and see where MJWarp's scaling flattens. The docs say single mechanisms above ~60 DoF are still being worked on.
3. Change the `njmax` per-world size and watch overflow behaviour and throughput.
4. Use `wp.to_torch(d.qpos)` to hand the state to PyTorch without a copy.
5. Look at the bundled agent skill: `.venv/lib/python3.12/site-packages/mujoco/doc/skills/accelerated/SKILL.md`.

## Measured here (RTX 5090, 32-core CPU, humanoid)
The GPU was shared with other jobs at ~99% utilisation during this run, so treat these as rough numbers:

| backend | world-steps/s |
|---|---|
| warp N=1 | 1.6k |
| warp N=64 | 103k |
| warp N=1024 | 1.7M |
| warp N=8192 | 6.3M |
| CPU `mj_step`, 1 thread | 47k–70k |
| CPU `mujoco.rollout`, 32 threads, 1024 traj | 0.8M–1.0M |
| MJX impl="jax", 1024 | 40k |
| MJX impl="warp", 1024 | 0.3M–0.7M (graph mode ranking flipped between runs) |

## Docs
- MuJoCo Warp: https://mujoco.readthedocs.io/en/stable/mjwarp/index.html (local copy: `mujoco/doc/mjwarp/index.rst`)
- MJX: https://mujoco.readthedocs.io/en/stable/mjx.html, in particular the "MJX-Warp" and "graph modes" sections
- rollout: https://mujoco.readthedocs.io/en/stable/python.html#rollout
- MJWarp tutorial notebook: https://github.com/google-deepmind/mujoco_warp/blob/main/notebooks/tutorial.ipynb
