-- TrustLens ClickHouse schema. Runs once on first container start.
CREATE DATABASE IF NOT EXISTS trustlens;

-- One row per scored image. Features are the Tier-1 heuristic outputs so we can
-- slice fraud rate by any signal in Superset.
CREATE TABLE IF NOT EXISTS trustlens.scores
(
    scored_at        DateTime DEFAULT now(),
    image_id         String,
    source           LowCardinality(String),        -- '140k_faces' | 'midv500' | 'live'
    fake_probability Float32,
    predicted_label  UInt8,                          -- 1 = fake/fraud
    threshold        Float32,
    -- ELA
    ela_mean         Float32,
    ela_std          Float32,
    ela_p99          Float32,
    -- EXIF
    exif_present     UInt8,
    exif_suspicion   Float32,
    -- FFT
    fft_highfreq_ratio Float32,
    fft_spectral_slope Float32,
    fft_peakiness      Float32
)
ENGINE = MergeTree
ORDER BY (scored_at, image_id);

-- Rolling operating-point view: fraud rate per source per hour.
CREATE TABLE IF NOT EXISTS trustlens.hourly_fraud_rate
(
    hour        DateTime,
    source      LowCardinality(String),
    n           UInt64,
    n_flagged   UInt64,
    fraud_rate  Float32
)
ENGINE = SummingMergeTree
ORDER BY (hour, source);
