"""Unit tests for data_loader module."""

import io
from datetime import datetime, timezone
import pandas as pd
import pytest

from data_loader import (
    fetch_nasa_firms_data,
    ingest_csv_data,
    normalize_column_names,
    normalize_confidence,
    parse_observation_timestamp,
)


def test_normalize_column_names_aliases():
    raw_df = pd.DataFrame(
        {
            "lat": [37.5],
            "lon": [-120.2],
            "date": ["2026-10-16"],
            "time": ["1430"],
            "conf": [85],
            "power": [55.4],
        }
    )
    norm_df = normalize_column_names(raw_df)
    assert "latitude" in norm_df.columns
    assert "longitude" in norm_df.columns
    assert "acq_date" in norm_df.columns
    assert "acq_time" in norm_df.columns
    assert "confidence" in norm_df.columns
    assert "frp" in norm_df.columns


def test_parse_observation_timestamp_formats():
    # ISO date with standard military time
    dt1 = parse_observation_timestamp("2026-10-16", "1430")
    assert dt1 == datetime(2026, 10, 16, 14, 30, tzinfo=timezone.utc)

    # Colon formatted time
    dt2 = parse_observation_timestamp("2026-10-16", "08:15")
    assert dt2 == datetime(2026, 10, 16, 8, 15, tzinfo=timezone.utc)

    # 3-digit morning time
    dt3 = parse_observation_timestamp("2026-10-16", "530")
    assert dt3 == datetime(2026, 10, 16, 5, 30, tzinfo=timezone.utc)

    # Invalid dates and times return None
    assert parse_observation_timestamp("invalid-date", "1430") is None
    assert parse_observation_timestamp("2026-10-16", "2599") is None
    assert parse_observation_timestamp(None, None) is None


def test_normalize_confidence_viirs_and_modis():
    # VIIRS categorical
    assert normalize_confidence("h") == 100.0
    assert normalize_confidence("high") == 100.0
    assert normalize_confidence("n") == 65.0
    assert normalize_confidence("nominal") == 65.0
    assert normalize_confidence("l") == 30.0
    assert normalize_confidence("low") == 30.0

    # MODIS numeric percentage
    assert normalize_confidence("85") == 85.0
    assert normalize_confidence(92) == 92.0
    assert normalize_confidence(0.95) == 95.0

    # Invalid / out of range
    assert normalize_confidence("unknown") is None
    assert normalize_confidence(150) is None
    assert normalize_confidence(None) is None


def test_ingest_csv_data_basic():
    csv_text = """latitude,longitude,acq_date,acq_time,satellite,instrument,confidence,frp
38.5,-121.2,2026-10-16,1200,NOAA-20,VIIRS,h,55.0
39.0,-120.8,2026-10-16,1205,Terra,MODIS,80,35.0
"""
    df, meta = ingest_csv_data(io.StringIO(csv_text), source_label="Test")
    assert len(df) == 2
    assert meta["raw_record_count"] == 2
    assert "datetime_utc" in df.columns
    assert "confidence_norm" in df.columns
    assert df["confidence_norm"].iloc[0] == 100.0
    assert df["confidence_norm"].iloc[1] == 80.0


def test_fetch_nasa_firms_missing_key():
    df, meta = fetch_nasa_firms_data(api_key="", country_code="USA")
    assert df.empty
    assert "missing" in meta["error"].lower()
