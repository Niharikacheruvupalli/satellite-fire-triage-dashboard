"""Benchmark evaluation module for satellite active fire priority scoring.

Evaluates triage priority ranking against explicitly defined synthetic sample labels.
Includes transparent limitations documentation distinguishing heuristic scoring
from validated machine learning and ground-truth fire suppression systems.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data_loader import ingest_csv_data
from ranking import calculate_priority_scores
from validation import validate_fire_records


# Structured operational limitations documentation
OPERATIONAL_LIMITATIONS: Dict[str, Dict[str, str]] = {
    "revisit_gaps": {
        "title": "Satellite Revisit Gaps & Temporal Latency",
        "description": (
            "Polar-orbiting satellites (Terra, Aqua, Suomi-NPP, NOAA-20, NOAA-21) provide only "
            "1 to 2 overpasses per day per instrument over mid-latitudes. Fires igniting, "
            "accelerating, or dying down between satellite passes will not be captured in real time. "
            "An absence of thermal detections does not confirm the absence of an active fire."
        ),
    },
    "cloud_interference": {
        "title": "Cloud, Heavy Smoke & Canopy Obscuration",
        "description": (
            "Thermal infrared sensors cannot penetrate thick cloud decks, dense smoke plumes, or heavy tree "
            "canopies. Under overcast weather conditions, intense wildland fires can be completely obscured "
            "from satellite detection (false negatives)."
        ),
    },
    "false_positives": {
        "title": "False Positives & Non-Wildfire Thermal Sources",
        "description": (
            "Thermal anomaly algorithms detect radiant heat signatures. Non-wildfire sources such as "
            "industrial gas flares, oil refineries, smelting plants, volcanic vents, and solar glint "
            "reflected off water bodies or solar panels can trigger active fire detections."
        ),
    },
    "geolocation_uncertainty": {
        "title": "Geolocation Uncertainty & Swath Expansion",
        "description": (
            "Nominal spatial resolution at nadir is 375m for VIIRS I-bands and 1km for MODIS. Towards the "
            "edge of the satellite scan swath, pixel footprint expands (bow-tie distortion) up to 2-3x, "
            "leading to positional displacement and duplicate detections across overlapping scans."
        ),
    },
    "sensor_discrepancy": {
        "title": "Sensor Resolution & Saturation Differences",
        "description": (
            "VIIRS 375m pixels are ~10x smaller in area than MODIS 1km pixels, allowing VIIRS to detect "
            "substantially smaller, cooler fires and measure separate fire fronts that MODIS aggregates "
            "into a single pixel. Direct numerical comparisons between raw MODIS and VIIRS FRP must "
            "account for sensor design differences."
        ),
    },
    "lack_of_ground_truth": {
        "title": "Lack of Ground-Truth Fire Confirmation",
        "description": (
            "Satellite observations detect instantaneous radiance at the instant of overpass. They do not "
            "measure fire containment status, fuel type, local wind velocity, rate of spread, or suppression "
            "actions. This tool prioritizes observations for analyst review and must never replace official "
            "wildfire dispatch or emergency-management decision systems."
        ),
    },
}


def evaluate_sample_benchmark(
    sample_csv_path: Optional[str] = None,
    priority_threshold: float = 60.0,
    top_k_list: Optional[List[int]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Evaluate observation-priority scoring alignment against explicitly labelled sample data.

    Calculates Precision@K, Recall@K, and threshold classification metrics (Sensitivity, Specificity)
    solely against the provided benchmark sample.

    DOES NOT claim or imply trained ML model generalization.
    """
    if top_k_list is None:
        top_k_list = [5, 10, 15]

    if sample_csv_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        sample_csv_path = os.path.join(base_dir, "data", "evaluation_labeled_sample.csv")

    if not os.path.exists(sample_csv_path):
        raise FileNotFoundError(f"Evaluation benchmark file not found: {sample_csv_path}")

    # Load and process benchmark
    df_raw, _ = ingest_csv_data(sample_csv_path, source_label="Evaluation Benchmark")
    df_valid, _ = validate_fire_records(df_raw)
    df_scored, _ = calculate_priority_scores(df_valid)

    # Merge ground truth columns if present in raw file
    raw_csv = pd.read_csv(sample_csv_path, comment="#")
    gt_map = {}
    for _, r in raw_csv.iterrows():
        rec_id = str(r.get("record_id", "")).strip()
        gt_map[rec_id] = {
            "ground_truth_tier": r.get("ground_truth_tier", "Unknown"),
            "ground_truth_binary": int(r.get("ground_truth_binary", 0)),
            "analyst_notes": r.get("analyst_notes", ""),
        }

    df_scored["ground_truth_tier"] = [gt_map.get(str(r["record_id"]), {}).get("ground_truth_tier", "Unknown") for _, r in df_scored.iterrows()]
    df_scored["ground_truth_binary"] = [gt_map.get(str(r["record_id"]), {}).get("ground_truth_binary", 0) for _, r in df_scored.iterrows()]
    df_scored["analyst_notes"] = [gt_map.get(str(r["record_id"]), {}).get("analyst_notes", "") for _, r in df_scored.iterrows()]

    # Sort descending by priority score
    df_sorted = df_scored.sort_values(by="priority_score", ascending=False).reset_index(drop=True)

    total_samples = len(df_sorted)
    total_positives = int(df_sorted["ground_truth_binary"].sum())
    total_negatives = total_samples - total_positives

    # Precision and Recall at K
    precision_at_k: Dict[str, float] = {}
    recall_at_k: Dict[str, float] = {}

    for k in top_k_list:
        if k <= total_samples:
            top_k_slice = df_sorted.head(k)
            k_positives = int(top_k_slice["ground_truth_binary"].sum())
            prec = k_positives / k
            rec = (k_positives / total_positives) if total_positives > 0 else 0.0
            precision_at_k[f"P@{k}"] = round(prec, 3)
            recall_at_k[f"R@{k}"] = round(rec, 3)

    # Threshold metrics
    preds_binary = (df_sorted["priority_score"] >= priority_threshold).astype(int)
    actual_binary = df_sorted["ground_truth_binary"].astype(int)

    tp = int(((preds_binary == 1) & (actual_binary == 1)).sum())
    fp = int(((preds_binary == 1) & (actual_binary == 0)).sum())
    tn = int(((preds_binary == 0) & (actual_binary == 0)).sum())
    fn = int(((preds_binary == 0) & (actual_binary == 1)).sum())

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

    # Average score per tier
    tier_scores = (
        df_sorted.groupby("ground_truth_tier")["priority_score"]
        .agg(["count", "mean", "min", "max"])
        .round(1)
        .to_dict(orient="index")
    )

    metrics = {
        "dataset_name": os.path.basename(sample_csv_path),
        "total_benchmark_records": total_samples,
        "priority_threshold": priority_threshold,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "sensitivity": round(sensitivity, 3),
        "specificity": round(specificity, 3),
        "f1_score": round(f1, 3),
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "tier_score_distribution": tier_scores,
        "methodology": (
            "Heuristic rule-based priority ranking. Not a machine learning model. "
            "Evaluated strictly against pre-annotated synthetic triage tiers."
        ),
    }

    return df_sorted, metrics
