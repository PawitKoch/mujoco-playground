import argparse


def example_args(description: str, seconds: float = 3.0, **extra) -> argparse.Namespace:
    """Common CLI: headless video by default, `--viewer` for the interactive viewer.

    `extra` adds boolean flags, e.g. example_args(..., no_speculative="run with it off").
    """
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--viewer", action="store_true", help="open the interactive viewer instead of rendering")
    p.add_argument("--seconds", type=float, default=seconds, help="simulated seconds to render")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    for flag, help_text in extra.items():
        p.add_argument(f"--{flag.replace('_', '-')}", action="store_true", help=help_text)
    return p.parse_args()
