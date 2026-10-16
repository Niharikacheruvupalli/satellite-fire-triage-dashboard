"""Unit tests for ranking module."""

from datetime import datetime, timezone
import pandas as pd
import pytest

from ranking import (
    calculate_priority_scores,
    compute_confidence_score,
    compute_frp_score,
    compute_recency_score,
    compute_repeat_detections,
    compute_repeat_score,
    haversine_distance_km,
)


def test_haversine_distance():
    # Roughly 1 degree of latitude is ~111 km
    dist = haversine_distance_km(37.0, -120.0, 38.0, -120.0)
    assert 110.0 <= dist <= 112.0


def test_compute_recency_score():
    assert compute_recency_score(0.0) == 100.0
    # Half-life is 12h
    assert 49.0 <= compute_recency_score(12.0) <= 51.0
    # 24h should be approx 25
    assert 24.0 <= compute_recency_score(24.0) <= 26.0
    # Future observation (negative hours) clamped to 100
    assert compute_recency_score(-2.0) == 100.0


def test_compute_confidence_score():
    assert compute_confidence_score(100.0) == 100.0
    assert compute_confidence_score(65.0) == 65.0
    assert compute_confidence_score(0.0) == 0.0
    assert compute_confidence_score(None) is None


def test_compute_frp_score():
    assert compute_frp_score(0.0) == 0.0
    # 250 MW is reference high FRP mapped to 100
    assert compute_frp_score(250.0) == 100.0
    # Higher FRP saturated at 100
    assert compute_frp_score(500.0) == 100.0
    assert compute_frp_score(None) is None


def test_compute_repeat_score():
    assert compute_repeat_score(0) == 20.0
    assert compute_repeat_score(1) == 36.0
    assert compute_repeat_score(5) == 100.0
    assert compute_repeat_score(10) == 100.0


def test_calculate_priority_scores_complete():
    t_ref = datetime(2026, 10, 16, 15, 0, tzinfo=timezone.utc)
    df = pd.DataFrame(
        {
            "record_id": ["REC_01"],
            "latitude": [39.75],
            "longitude": [-121.62],
            "datetime_utc": [datetime(2026, 10, 16, 14, 0, tzinfo=timezone.utc)],  # 1h ago
            "confidence_norm": [100.0],
            "frp": [150.0],
            "satellite": ["NOAA-20"],
        }
    )
    df_scored, meta = calculate_priority_scores(df, reference_time=t_ref)
    assert len(df_scored) == 1
    score = df_scored["priority_score"].iloc[0]
    assert 0.0 <= score <= 100.0
    assert score >= 70.0  # High recency, confidence, and FRP
    assert bool(df_scored["score_incomplete"].iloc[0]) is False
    assert "explanation" in df_scored.columns
    assert "Observation-Priority Score" in df_scored["explanation"].iloc[0]


def test_calculate_priority_scores_missing_data_renormalization():
    t_ref = datetime(2026, 10, 16, 15, 0, tzinfo=timezone.utc)
    # FRP is None, so weights should renormalize over recency, confidence, and repeats
    df = pd.DataFrame(
        {
            "record_id": ["REC_MISSING_FRP"],
            "latitude": [39.75],
            "longitude": [-121.62],
            "datetime_utc": [datetime(2026, 10, 16, 14, 0, tzinfo=timezone.utc)],
            "confidence_norm": [100.0],
            "frp": [None],
            "satellite": ["NOAA-20"],
        }
    )
    df_scored, _ = calculate_priority_scores(df, reference_time=t_ref)
    assert bool(df_scored["score_incomplete"].iloc[0]) is True
    assert df_scored["frp_score"].iloc[0] is None
    # Score is still positive and non-zero
    assert df_scored["priority_score"].iloc[0] > 50.0
    assert "renormalized" in df_scored["explanation"].iloc[0].lower()
