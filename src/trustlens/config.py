"""Central configuration, env-overridable."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # Operating point: the FPR we are willing to tolerate. Recall is maximized here.
    target_fpr: float = float(os.environ.get("TARGET_FPR", "0.01"))
    # ELA recompression quality.
    ela_quality: int = int(os.environ.get("ELA_QUALITY", "90"))
    # Longest side images are resized to before feature extraction (speed + scale invariance).
    work_size: int = int(os.environ.get("WORK_SIZE", "512"))

    kafka_broker: str = os.environ.get("KAFKA_BROKER", "kafka:9092")
    clickhouse_host: str = os.environ.get("CLICKHOUSE_HOST", "clickhouse")
    clickhouse_port: int = int(os.environ.get("CLICKHOUSE_PORT", "8123"))


settings = Settings()
