# TrustLens

Image & identity-document fraud detection pipeline. Detects manipulated,
synthetic (GAN/diffusion), and forged imagery using a **tiered** approach:

- **Tier 1 — Heuristics (this milestone):** fast, explainable signal from
  Error Level Analysis (ELA), EXIF-consistency checks, and FFT spectral
  analysis. Ships first as the "simple thing that works" baseline.
- **Tier 2 — Learned models (later):** CNN / transformer classifiers.
- **Tier 3 — Ensemble + human review (later).**

## Why heuristics first

Fraud detection is **asymmetric-cost**: a missed fake (false negative) is far
more expensive than a false alarm. So we do not optimize plain accuracy. We fix
an acceptable **false-positive rate (FPR)** and maximize **recall (TPR)** there.
The Tier-1 heuristic classifier is the baseline every later model must beat at
the same operating point.

## Architecture

| Service    | Role                                   | Image (official)        |
|------------|----------------------------------------|-------------------------|
| FastAPI    | Scoring API (`/score`) + health        | built from `docker/api` |
| Kafka      | Ingest stream of images to score       | `bitnami/kafka`         |
| ClickHouse | Store scores/features for analytics    | `clickhouse/clickhouse-server` |
| Superset   | Dashboards over ClickHouse             | `apache/superset`       |
| Airflow    | Batch scoring / dataset ETL orchestration | `apache/airflow`     |

See `docker-compose.yml`. Everything comes up with `make up`.

## Quickstart

```bash
# 1. Python env for local dev + tests
make setup            # venv + pip install -e .[dev]

# 2. Run the test suite (uses synthetic fixtures, no dataset needed)
make test

# 3. Bring up the full stack
make up               # docker compose up -d
#    FastAPI  -> http://localhost:8000/docs
#    Superset -> http://localhost:8088   (admin/admin)
#    Airflow  -> http://localhost:8080   (admin/admin)

# 4. Score an image
curl -F "file=@some.jpg" http://localhost:8000/score
```

## Datasets

Real downloads are auth-gated / multi-GB, so they are **not** committed. Scripts
fetch them into `data/raw/`:

```bash
scripts/download_140k_faces.sh   # Kaggle: xhlulu/140k-real-and-fake-faces
scripts/download_midv500.sh      # MIDV-500 identity documents
```

Tests and CI run on small **synthetic fixtures** generated on the fly
(`scripts/make_fixtures.py`) so nothing is blocked on the big downloads.

## Layout

```
src/trustlens/
  heuristics/   ela.py  exif.py  fft.py  classifier.py
  metrics/      precision/recall @ fixed FPR, ROC/PR, threshold selection
  data/         fixture generation + loaders
  api/          FastAPI app
scripts/        dataset downloads, fixture generation
notebooks/      01_eda.ipynb
infra/          clickhouse / superset / airflow configs
tests/          pytest suite
```

## Metrics contract

`trustlens.metrics.evaluate_at_fpr(y_true, scores, target_fpr=0.01)` returns the
threshold, recall, precision, and confusion counts at the chosen FPR. This is
the single number Day-2+ models are graded against.
