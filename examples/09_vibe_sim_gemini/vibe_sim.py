"""09 - "Vibe sim": edit a MuJoCo scene by chatting with Gemini 4 Hydrogen.

You type what you want ("make the ball heavier and roll it off the table").
Gemini sees the scene description plus a rendered image, then calls Python
tools below. Each tool is a plain MjSpec edit + recompile. It can then call
`simulate` to check that the result does what you asked, and fix things if not.
That check-and-fix loop is what Google calls "vibe sim".

    uv run python examples/09_vibe_sim_gemini/vibe_sim.py                       # interactive chat
    uv run python examples/09_vibe_sim_gemini/vibe_sim.py --script "make the cube red and twice as big"
    uv run python examples/09_vibe_sim_gemini/vibe_sim.py --model gemini-3.8-flash

Chat commands: /view (open the viewer), /xml (print the current MJCF), /reset, /quit
Needs GEMINI_API_KEY in .env (repo root). Sends scene text + images to the Gemini API.
"""

import argparse
import sys
import time
from pathlib import Path

import httpx
import mujoco
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

import mjplay
from scene import SHAPES, VibeScene

REPO = Path(__file__).resolve().parents[2]

SYSTEM = """You are a MuJoCo simulation engineer editing a live scene for the user.
Units: metres, kilograms, seconds. +z is up; the table top surface is at z=0.72.
Geom sizes follow MuJoCo conventions: box = half-extents [x,y,z], sphere = [radius],
capsule/cylinder = [radius, half-length], ellipsoid = [rx, ry, rz].
Object `pos` is the centre of the object's body, so an object resting on the table
needs z = 0.72 + its half-height.
Always use the tools to make changes; never just describe XML. After making
physical changes, call `simulate` to check the outcome matches the user's intent,
and keep fixing until it does. Reply briefly: what you changed and what the
simulation showed."""

scene = VibeScene()


# --- tools Gemini can call (docstrings + type hints become the tool schema) ---

def _edit(fn) -> str:
    """Apply an edit to a copy of the spec; keep it only if it compiles."""
    backup = scene.spec.copy()
    try:
        msg = fn()
        scene.recompile()
    except Exception as e:  # noqa: BLE001 - errors go back to the model as text
        scene.spec = backup
        scene.recompile()
        return f"ERROR (edit reverted): {e}"
    scene.log.append(msg)
    print(f"   [tool] {msg}")
    return msg + "\n" + scene.describe()


def describe_scene() -> str:
    """Return the current physics options and every object's type, size, position, colour, mass and friction."""
    return scene.describe()


def set_object(
    name: str,
    size: list[float] | None = None,
    pos: list[float] | None = None,
    rgba: list[float] | None = None,
    mass: float | None = None,
    friction: float | None = None,
) -> str:
    """Change an existing object (geom) by name. Only the given fields change.

    Args:
      name: geom name, e.g. "cube", "ball", "table_top".
      size: MuJoCo geom size (see system prompt for conventions).
      pos: world position [x, y, z] of the object's body (movable objects) or the geom (static ones).
      rgba: colour, 4 floats in [0, 1].
      mass: kilograms.
      friction: sliding friction coefficient (typically 0.1 - 2).
    """
    def apply():
        g = scene.geom(name)
        movable = scene.model.body_dofnum[scene.model.geom_bodyid[scene.model.geom(name).id]] > 0
        if size is not None:
            g.size[: len(size)] = size
        if pos is not None:
            if movable:
                g.parent.pos = pos
            else:
                g.pos = [p - b for p, b in zip(pos, scene.model.body(g.parent.name).pos)]
        if rgba is not None:
            g.rgba = rgba
            g.material = ""  # an explicit colour should win over a material
        if mass is not None:
            g.mass = mass
        if friction is not None:
            g.friction[0] = friction
        return f"set_object({name}, size={size}, pos={pos}, rgba={rgba}, mass={mass}, friction={friction})"
    return _edit(apply)


def add_object(name: str, shape: str, size: list[float], pos: list[float], rgba: list[float], mass: float) -> str:
    """Add a new free-moving object.

    Args:
      name: unique name for the new object.
      shape: one of box, sphere, capsule, cylinder, ellipsoid.
      size: MuJoCo geom size.
      pos: world position [x, y, z] of its centre.
      rgba: colour, 4 floats in [0, 1].
      mass: kilograms.
    """
    def apply():
        if shape not in SHAPES:
            raise ValueError(f"shape must be one of {list(SHAPES)}")
        body = scene.spec.worldbody.add_body(name=name, pos=pos)
        body.add_freejoint()
        body.add_geom(name=name, type=SHAPES[shape], size=size + [0] * (3 - len(size)), rgba=rgba, mass=mass)
        return f"add_object({name}, {shape}, size={size}, pos={pos})"
    return _edit(apply)


def remove_object(name: str) -> str:
    """Delete a free-moving object (and its body) by name."""
    def apply():
        scene.spec.delete(scene.body(name))
        return f"remove_object({name})"
    return _edit(apply)


def set_physics(gravity: list[float] | None = None, timestep: float | None = None) -> str:
    """Change global physics: gravity vector [gx, gy, gz] in m/s^2, and/or the timestep in seconds."""
    def apply():
        if gravity is not None:
            scene.spec.option.gravity = gravity
        if timestep is not None:
            scene.spec.option.timestep = timestep
        return f"set_physics(gravity={gravity}, timestep={timestep})"
    return _edit(apply)


def push_object(name: str, velocity: list[float]) -> str:
    """Give a movable object an initial linear velocity [vx, vy, vz] (m/s) for the next simulate call."""
    if scene.model.body_dofnum[scene.model.body(name).id] != 6:
        return f"ERROR: {name} is not a free-moving object"
    scene.initial_velocity[name] = velocity
    msg = f"push_object({name}, {velocity})"
    scene.log.append(msg)
    print(f"   [tool] {msg}")
    return msg


def simulate(seconds: float) -> str:
    """Run the physics from the spawn state for `seconds` (max 5) and report where each object ends up."""
    seconds = min(float(seconds), 5.0)
    print(f"   [tool] simulate({seconds})")
    return scene.simulate(seconds)


TOOLS = [describe_scene, set_object, add_object, remove_object, set_physics, push_object, simulate]


# --- chat loop ------------------------------------------------------------------

def snapshot(turn: int, tag: str) -> types.Part:
    """Render the scene, save the frame + MJCF, and return the frame as an image Part for Gemini."""
    data = mujoco.MjData(scene.model)
    image = mjplay.render_still(scene.model, data, camera="overview", width=640, height=480)
    path = mjplay.save_image(image, f"09_turn_{turn:02d}_{tag}.png")
    mjplay.output_path("09_vibe_scene.xml").write_text(scene.spec.to_xml())
    return types.Part.from_bytes(data=path.read_bytes(), mime_type="image/png")


def send(chat_factory, chat, parts, models: list[str]):
    """send_message with retry/backoff and model fallback (preview models are often at capacity)."""
    for model in models:
        for attempt in range(2):
            try:
                return chat, chat.send_message(parts)
            except (errors.ServerError, httpx.TimeoutException) as e:
                wait = 2 ** attempt * 5
                print(f"   [{model}] {getattr(e, 'status', type(e).__name__)}; retrying in {wait}s")
                time.sleep(wait)
            except errors.ClientError as e:
                if e.code != 429:
                    raise
                print(f"   [{model}] rate limited; retrying in 20s")
                time.sleep(20)
        if model != models[-1]:
            nxt = models[models.index(model) + 1]
            print(f"   falling back to {nxt}")
            chat = chat_factory(nxt, chat.get_history())
    raise RuntimeError("all models unavailable; try again later")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="gemini-4-hydrogen")
    p.add_argument("--fallback", default="gemini-3.8-flash,gemini-3.5-flash",
                   help="comma-separated models tried in order if --model is unavailable ('' to disable)")
    p.add_argument("--thinking", default="medium", choices=["low", "medium", "high", "xhigh"])
    p.add_argument("--script", nargs="*", help="run these prompts non-interactively, then exit")
    args = p.parse_args()

    load_dotenv(REPO / ".env")
    # Per-request timeout (ms): without one, a stalled request blocks forever.
    client = genai.Client(http_options=types.HttpOptions(timeout=180_000))

    def chat_factory(model: str, history=None):
        return client.chats.create(
            model=model,
            history=history,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM,
                tools=TOOLS,  # plain Python functions: the SDK builds schemas and runs the tool loop
                automatic_function_calling=types.AutomaticFunctionCallingConfig(maximum_remote_calls=20),
                thinking_config=types.ThinkingConfig(thinking_level=args.thinking),
            ),
        )

    models = [args.model] + [m for m in args.fallback.split(",") if m and m != args.model]
    chat = chat_factory(models[0])
    print(f"vibe sim with {args.model}. Scene:\n{scene.describe()}\n")

    prompts = iter(args.script) if args.script is not None else None
    turn = 0
    while True:
        try:
            text = next(prompts) if prompts else input("you> ").strip()
        except (StopIteration, EOFError):
            break
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/xml":
            print(scene.spec.to_xml())
            continue
        if text == "/view":
            mjplay.view(scene.model, mujoco.MjData(scene.model))
            continue
        if text == "/reset":
            globals()["scene"] = VibeScene()
            chat = chat_factory(models[0])
            print("scene and chat reset")
            continue

        turn += 1
        print(f"you> {text}" if prompts else "", end="\n" if prompts else "")
        parts = [snapshot(turn, "before"), f"Current scene:\n{scene.describe()}\n\nRequest: {text}"]
        chat, reply = send(chat_factory, chat, parts, models)
        print(f"gemini> {reply.text}\n")
        snapshot(turn, "after")

    print(f"final scene saved to {mjplay.output_path('09_vibe_scene.xml')}")


if __name__ == "__main__":
    sys.exit(main())
