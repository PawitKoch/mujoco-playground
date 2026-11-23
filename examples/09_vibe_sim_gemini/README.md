# 09 — Vibe sim with Gemini 4 Hydrogen

> "You can then 'vibe sim' by adjusting shapes, positions, and physical properties
> through chat to verify reaching and grasping before deploying to physical hardware."
> — [Gemini Hydrogen for robotics](https://developers.googleblog.com/gemini-hydrogen-for-robotics/)

"Vibe sim" is not a MuJoCo API. It is a workflow: an LLM agent edits a MuJoCo
scene for you in response to chat, then **runs the physics to check its own edit**.
This folder builds a small, readable version of that loop, so you can see every
moving part.

```
you ──"make the ball roll off the table"──▶ Gemini (sees: scene text + rendered PNG)
                                              │  function calls
                                              ▼
              set_object / add_object / push_object / set_physics  ──▶  MjSpec edit → spec.compile()
                                              │                          (compile error → reverted, sent back)
                                              ▼
                                         simulate(2.0)  ──▶ "ball: end z=0.02 (FELL OFF TABLE)"
                                              │
                                     not what you asked? edit again
```

## Run it

```bash
uv sync --group gemini        # google-genai, python-dotenv (already done if you ran uv sync --all-groups)
# .env at the repo root must contain GEMINI_API_KEY=...

uv run python examples/09_vibe_sim_gemini/vibe_sim.py
you> make the cube twice as big and paint it green
you> stack three small boxes into a tower next to the arm, and check it doesn't topple
you> make the table icy and push the ball towards the robot
you> /view       # opens the MuJoCo viewer on the current scene
you> /xml        # prints the MJCF Gemini has built so far
```

Non-interactive: `--script "prompt 1" "prompt 2"`. Each turn writes
`outputs/09_turn_NN_before.png` / `_after.png`. The current scene is always saved to
`outputs/09_vibe_scene.xml`, so you can open it later in any MuJoCo viewer.

`gemini-4-hydrogen` is new and is often at capacity (HTTP 503 or 504). The script
retries, then works down the `--fallback` chain (default `gemini-3.8-flash,gemini-3.5-flash`).
The banner tells you which model actually answered.
Pick the model and reasoning level with `--model` and `--thinking low|medium|high|xhigh`.

## Files

| File | What to read it for |
|---|---|
| `scene.py` | The base scene as an `MjSpec`: a table and objects, with the [Menagerie](https://github.com/google-deepmind/mujoco_menagerie) SO-101 arm attached through `frame.attach_body`. It also has `describe()`, the text Gemini sees, and `simulate()`, the verification step. |
| `vibe_sim.py` | The tools are **plain Python functions**. `google-genai` turns their signatures and docstrings into tool schemas and runs the call loop for you (automatic function calling). `_edit()` shows the safe-edit pattern: copy the spec, mutate it, compile it, and restore the copy on error. |

The SO-101 is loaded from `../mujoco_menagerie` (your workspace checkout). Override
the location with `MENAGERIE_PATH=...`. If the arm isn't found, the scene works without it.

## Things to try

- Add a tool: `set_arm_pose(joint angles)`, which writes `data.ctrl` before `simulate`. Then ask Gemini
  to "reach the cube". This is the "verify reaching" step from the blog post.
- Give Gemini the speculative-contacts flag from example 01, or the plasticity from example 02, as tools.
  "Drop a soft clay block on the table and dent it."
- Send a short video instead of a single still: render `simulate` with `mjplay.render_rollout`.
- Swap the hand-written tools for the official agent skill (below), and compare how much
  context each approach needs.

## The official route: Antigravity + the Robotics plugin

Google's own integration is not a Python SDK. It is a set of **agent skills** loaded into the
[Antigravity](https://antigravity.google/) agent IDE:

```bash
# Robotics plugin: writes a controller, tests it in MuJoCo, runs the same code on a real arm
agy plugin install https://github.com/google-deepmind/robotics-antigravity-plugin
# just the MuJoCo skills (mujoco-python, mujoco-spec-editing, mujoco-rendering, ...)
npx skills add google-deepmind/mujoco
```

MuJoCo 4.0 also **ships those same skills inside the wheel**. Run example 00 to list them, or read
them directly:

```bash
ls .venv/lib/python3.12/site-packages/mujoco/doc/skills/
cat .venv/lib/python3.12/site-packages/mujoco/doc/skills/spec_editing/SKILL.md
```

Those SKILL.md files are good learning material even without an agent. They are DeepMind's
own condensed "how to use MjSpec / rendering / MJX correctly" notes. Example prompts from the plugin README:
"Generate a pelican riding a bicycle simulation in MuJoCo", and
"Create a MuJoCo grocery sorting task using a LeRobot SO-101 arm".

## Privacy

Running `vibe_sim.py` sends the scene description and rendered images to the Gemini API.
Nothing is sent unless you run it.
