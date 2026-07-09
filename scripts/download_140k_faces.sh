#!/usr/bin/env bash
# Download the "140k Real and Fake Faces" dataset from Kaggle.
# Dataset: xhlulu/140k-real-and-fake-faces  (~4 GB)
#
# Requires the Kaggle CLI and credentials:
#   pip install kaggle
#   put kaggle.json in ~/.kaggle/  (chmod 600), OR export KAGGLE_USERNAME / KAGGLE_KEY
set -euo pipefail

DEST="${1:-data/raw/140k_faces}"
SLUG="xhlulu/140k-real-and-fake-faces"

mkdir -p "$DEST"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "ERROR: kaggle CLI not found. Install with: pip install kaggle" >&2
  exit 1
fi

echo ">> Downloading $SLUG -> $DEST"
kaggle datasets download -d "$SLUG" -p "$DEST" --unzip

echo ">> Done. Layout:"
find "$DEST" -maxdepth 2 -type d | head -20
echo ">> Expected: real_vs_fake/real-vs-fake/{train,valid,test}/{real,fake}/"
