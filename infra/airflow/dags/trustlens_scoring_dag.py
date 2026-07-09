"""Batch-scoring DAG skeleton for TrustLens.

Walks a dataset directory, runs the Tier-1 heuristic classifier over each image,
and (eventually) writes rows to ClickHouse. On Day 1 the write step is a stub so
the DAG parses and runs end-to-end without external state.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "trustlens",
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
}


def score_batch(**context) -> int:
    """Score every image under DATA_DIR. Returns count scored."""
    import os
    from pathlib import Path

    # Imported lazily so the scheduler can parse the DAG without the package.
    from trustlens.heuristics.classifier import HeuristicClassifier

    data_dir = Path(os.environ.get("TRUSTLENS_DATA_DIR", "/opt/airflow/data/raw"))
    clf = HeuristicClassifier.default()

    n = 0
    for path in data_dir.rglob("*.jp*g"):
        try:
            clf.score_path(str(path))
            n += 1
        except Exception:  # noqa: BLE001 - never fail the whole batch on one image
            continue
    return n


with DAG(
    dag_id="trustlens_scoring",
    description="Batch Tier-1 heuristic scoring over a dataset directory.",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    default_args=default_args,
    tags=["trustlens", "tier1"],
) as dag:
    PythonOperator(task_id="score_batch", python_callable=score_batch)
