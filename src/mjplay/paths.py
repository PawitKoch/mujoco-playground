from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = REPO_ROOT / "outputs"
# Upstream MuJoCo 4.0.1 example models (sparse checkout, see scripts/fetch_upstream.sh).
UPSTREAM_MODELS = REPO_ROOT / "third_party" / "mujoco" / "model"


def output_path(name: str) -> Path:
    OUTPUTS.mkdir(exist_ok=True)
    return OUTPUTS / name
