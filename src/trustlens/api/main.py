"""FastAPI scoring service.

POST /score with an image file -> Tier-1 heuristic fake probability + features.
GET  /health -> liveness.

The model is the rule-based default at startup; if a calibrated model exists at
$TRUSTLENS_MODEL it is loaded instead.
"""
from __future__ import annotations

import io
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from ..heuristics.classifier import HeuristicClassifier, extract_features, features_to_vector

app = FastAPI(title="TrustLens", version="0.1.0",
              description="Tier-1 heuristic image-fraud scoring.")


def _load_model() -> HeuristicClassifier:
    path = os.environ.get("TRUSTLENS_MODEL")
    if path and os.path.exists(path):
        return HeuristicClassifier.load(path)
    return HeuristicClassifier.default()


model = _load_model()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "fitted": model.fitted, "threshold": model.threshold}


@app.post("/score")
async def score(file: UploadFile = File(...)) -> dict:
    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Not a readable image.")

    feats = extract_features(img)
    prob = float(model.predict_proba_features(features_to_vector(feats))[0])
    return {
        "filename": file.filename,
        "fake_probability": prob,
        "predicted_label": int(prob >= model.threshold),
        "threshold": model.threshold,
        "features": feats,
    }
