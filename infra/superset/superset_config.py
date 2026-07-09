"""Minimal Superset config for TrustLens.

The ClickHouse connection is added at bootstrap; add it manually via the UI
(Data -> Databases) with SQLAlchemy URI:

    clickhousedb://trustlens:trustlens@clickhouse:8123/trustlens
"""
import os

SECRET_KEY = os.environ.get("SUPERSET_SECRET_KEY", "change-me-in-prod")
SQLALCHEMY_DATABASE_URI = "sqlite:////app/superset_home/superset.db"

FEATURE_FLAGS = {"EMBEDDED_SUPERSET": True}
SQLLAB_CTAS_NO_LIMIT = True
