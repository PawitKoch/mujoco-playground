# 00 — Hello, MuJoCo 4

```bash
uv run python examples/00_hello_mujoco4/run.py
```

This is an install check that also gives a quick tour of what changed in the box.

## What's new that you see here

- **The package is split.** `pip install mujoco` is now a metapackage made of:
  - `mujoco-core`: the C library and bindings, needing only numpy. Use it for servers and RL workers.
  - `mujoco-studio`: the new Studio viewer and the Filament renderer (see example 10).

  To upgrade an older env, run `pip uninstall -y mujoco && pip install mujoco`.
- **The manual and agent skills are bundled** under `mujoco/doc/`. There are 45 `.rst` pages of the manual,
  plus `doc/skills/*/SKILL.md`. The skills are the same ones the Gemini/Antigravity Robotics plugin uses
  (see example 09). Grep the bundled docs instead of the website: they always match your installed version.
  ```bash
  grep -n "springyield" .venv/lib/python3.12/site-packages/mujoco/doc/XMLreference.rst
  ```
- **New mjData fields.** `xiquat`, `geom_xquat` and `site_xquat` sit next to the `*xmat` rotation matrices,
  so you no longer need `mju_mat2Quat`.
- **New geom attribute `castshadow`.** The floating blue sphere in `outputs/00_hello.png` casts no shadow.
- **New enable flags.** `mjENBL_SPECULATIVE` is covered in example 01, `mjENBL_IPC` in example 04, and
  `mjSTATE_PLASTIC` in example 02.
- Python 3.10 is no longer supported. This repo pins 3.12.

Docs: [Python bindings](https://mujoco.readthedocs.io/en/stable/python.html) ·
[Changelog 4.0](https://mujoco.readthedocs.io/en/stable/changelog.html)
