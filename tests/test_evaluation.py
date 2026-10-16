"""Unit tests for evaluation module."""

import os
import pytest

from evaluation import (
    OPERATIONAL_LIMITATIONS,
    evaluate_sample_benchmark,
)


def test_operational_limitations_structure():
    assert "revisit_gaps" in OPERATIONAL_LIMITATIONS
    assert "cloud_interference" in OPERATIONAL_LIMITATIONS
    assert "false_positives" in OPERATIONAL_LIMITATIONS
    assert "geolocation_uncertainty" in OPERATIONAL_LIMITATIONS
    assert "sensor_discrepancy" in OPERATIONAL_LIMITATIONS
    assert "lack_of_ground_truth" in OPERATIONAL_LIMITATIONS

    for key, data in OPERATIONAL_LIMITATIONS.items():
        assert "title" in data
        assert "description" in data
        assert len(data["description"]) > 20


def test_evaluate_sample_benchmark():
    df_sorted, metrics = evaluate_sample_benchmark()
    assert len(df_sorted) > 0
    assert "priority_score" in df_sorted.columns
    assert "ground_truth_tier" in df_sorted.columns

    # Verify metrics dictionary contents
    assert "precision_at_k" in metrics
    assert "recall_at_k" in metrics
    assert "f1_score" in metrics
    assert "tier_score_distribution" in metrics
    assert "P@5" in metrics["precision_at_k"]
    # High precision expected on the top 5
    assert metrics["precision_at_k"]["P@5"] >= 0.8
    assert metrics["f1_score"] >= 0.7

    # Verify heuristic ranking caveat is explicitly documented
    assert "heuristic" in metrics["methodology"].lower()
    assert "not a machine learning model" in metrics["methodology"].lower()
