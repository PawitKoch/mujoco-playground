#!/usr/bin/env bash
# Sparse-checkout the upstream MuJoCo example models (model/ only) at the pinned tag.
# Several examples load these directly so you can compare against the originals.
set -euo pipefail
TAG="${MUJOCO_TAG:-4.0.1}"
DEST="$(cd "$(dirname "$0")/.." && pwd)/third_party/mujoco"
if [[ -d "$DEST/.git" ]]; then
  echo "already present: $DEST"; exit 0
fi
git clone --quiet --depth 1 --branch "$TAG" --filter=blob:none --sparse \
  https://github.com/google-deepmind/mujoco.git "$DEST"
git -C "$DEST" sparse-checkout set model
echo "fetched mujoco $TAG models into $DEST/model"
