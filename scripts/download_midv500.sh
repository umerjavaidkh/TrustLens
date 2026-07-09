#!/usr/bin/env bash
# Download the MIDV-500 identity-document dataset (50 document types, video frames).
# Project page: ftp://smartengines.com/midv-500  (mirrored over HTTP below)
# Each of the 50 archives is downloaded and extracted into $DEST.
set -euo pipefail

DEST="${1:-data/raw/midv500}"
BASE="ftp://smartengines.com/midv-500/dataset"

mkdir -p "$DEST"

# The 50 documents are numbered 01..50 with descriptive slugs. To keep this
# script robust we pull the official file list and fetch each archive.
LIST_URL="ftp://smartengines.com/midv-500/list_of_files.txt"

echo ">> Fetching file list from $LIST_URL"
if command -v wget >/dev/null 2>&1; then
  wget -q -O "$DEST/list_of_files.txt" "$LIST_URL" || {
    echo "Could not fetch list; see https://github.com/fcakyon/midv500 for a Python downloader." >&2
    echo "Alternative:  pip install midv500 && python -c \"import midv500; midv500.download_dataset('$DEST')\"" >&2
    exit 1
  }
else
  echo "wget not found." >&2
  exit 1
fi

echo ">> Downloading archives listed in $DEST/list_of_files.txt"
while IFS= read -r rel; do
  [ -z "$rel" ] && continue
  echo "   - $rel"
  wget -q -P "$DEST" "$BASE/$rel" || echo "     (skip: $rel)"
done < "$DEST/list_of_files.txt"

echo ">> Extracting .zip archives"
find "$DEST" -name '*.zip' -exec unzip -n -q {} -d "$DEST" \;

echo ">> Done. Tip: the fcakyon/midv500 pip package is the most reliable downloader."
