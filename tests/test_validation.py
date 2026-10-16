"""Unit tests for validation module."""

from datetime import datetime, timezone
import pandas as pd
import pytest

from validation import (
    STATUS_INVALID,
    STATUS_VALID,
    STATUS_WARNING,
    validate_fire_records,
)


def test_validation_clean_records():
    df = pd.DataFrame(
        {
            "record_id": ["REC_01"],
            "latitude": [39.75],
            "longitude": [-121.62],
            "datetime_utc": [datetime(2026, 10, 16, 14, 30, tzinfo=timezone.utc)],
            "confidence": ["h"],
            "confidence_norm": [100.0],
            "frp": [145.8],
            "satellite": ["NOAA-20"],
        }
    )
    df_val, summary = validate_fire_records(df)
    assert len(df_val) == 1
    assert df_val["validation_status"].iloc[0] == STATUS_VALID
    assert bool(df_val["is_valid"].iloc[0]) is True
    assert summary["valid_records"] == 1
    assert summary["invalid_records"] == 0


def test_validation_out_of_bounds_coordinates():
    df = pd.DataFrame(
        {
            "record_id": ["REC_BAD_LAT", "REC_BAD_LON"],
            "latitude": [105.0, 35.0],
            "longitude": [-120.0, -250.0],
            "datetime_utc": [
                datetime(2026, 10, 16, 12, 0, tzinfo=timezone.utc),
                datetime(2026, 10, 16, 12, 0, tzinfo=timezone.utc),
            ],
            "confidence_norm": [80.0, 80.0],
            "frp": [30.0, 30.0],
            "satellite": ["Terra", "Terra"],
        }
    )
    df_val, summary = validate_fire_records(df)
    assert summary["invalid_records"] == 2
    assert df_val["validation_status"].iloc[0] == STATUS_INVALID
    assert df_val["validation_status"].iloc[1] == STATUS_INVALID
    # Verify records were NOT silently dropped
    assert len(df_val) == 2


def test_validation_negative_frp():
    df = pd.DataFrame(
        {
            "record_id": ["REC_NEG_FRP"],
            "latitude": [38.4],
            "longitude": [-122.1],
            "datetime_utc": [datetime(2026, 10, 16, 13, 0, tzinfo=timezone.utc)],
            "confidence_norm": [90.0],
            "frp": [-25.5],
            "satellite": ["NOAA-21"],
        }
    )
    df_val, summary = validate_fire_records(df)
    assert df_val["validation_status"].iloc[0] == STATUS_INVALID
    assert summary["invalid_records"] == 1
    assert any("negative" in issue.lower() for issue in df_val["validation_issues"].iloc[0])


def test_validation_extreme_frp_warning():
    df = pd.DataFrame(
        {
            "record_id": ["REC_EXTREME"],
            "latitude": [35.1],
            "longitude": [-115.8],
            "datetime_utc": [datetime(2026, 10, 16, 13, 15, tzinfo=timezone.utc)],
            "confidence_norm": [99.0],
            "frp": [7500.0],
            "satellite": ["Aqua"],
        }
    )
    df_val, summary = validate_fire_records(df)
    assert df_val["validation_status"].iloc[0] == STATUS_WARNING
    assert summary["warning_records"] == 1
    assert bool(df_val["is_valid"].iloc[0]) is True


def test_validation_duplicate_detection():
    dt = datetime(2026, 10, 16, 14, 30, tzinfo=timezone.utc)
    df = pd.DataFrame(
        {
            "record_id": ["REC_ORIGINAL", "REC_DUP"],
            "latitude": [39.7592, 39.7593],  # rounds to 39.759
            "longitude": [-121.6219, -121.6218],  # rounds to -121.622
            "datetime_utc": [dt, dt],
            "confidence_norm": [100.0, 100.0],
            "frp": [145.8, 145.8],
            "satellite": ["NOAA-20", "NOAA-20"],
        }
    )
    df_val, summary = validate_fire_records(df)
    assert bool(df_val["is_duplicate"].iloc[0]) is False
    assert bool(df_val["is_duplicate"].iloc[1]) is True
    assert summary["duplicate_records"] == 1
