"""Explainable observation-priority ranking module for satellite fire triage.

Calculates a 0-100 heuristic score based on Recency, Confidence, FRP,
and Repeat Spatial-Temporal Cluster corroboration with weight renormalization
and human-interpretable explanations.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Default configuration weights
DEFAULT_WEIGHTS: Dict[str, float] = {
    "recency": 0.30,
    "confidence": 0.30,
    "frp": 0.25,
    "repeat": 0.15,
}


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two points in kilometers."""
    r = 6371.0  # Earth radius in kilometers
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c


def compute_recency_score(delta_hours: Optional[float], half_life_hours: float = 12.0) -> Optional[float]:
    """Compute recency score [0-100] using exponential decay relative to reference time.

    Observations <= 0h get 100. At half_life_hours (default 12h), score is 50.
    """
    if delta_hours is None or pd.isna(delta_hours):
        return None

    dh = max(0.0, float(delta_hours))
    decay_constant = math.log(2) / half_life_hours
    score = 100.0 * math.exp(-decay_constant * dh)
    return round(float(np.clip(score, 0.0, 100.0)), 1)


def compute_confidence_score(conf_norm: Optional[float]) -> Optional[float]:
    """Compute normalized confidence score [0-100]."""
    if conf_norm is None or pd.isna(conf_norm):
        return None
    try:
        val = float(conf_norm)
        return round(float(np.clip(val, 0.0, 100.0)), 1)
    except (ValueError, TypeError):
        return None


def compute_frp_score(frp_val: Optional[float], reference_high_frp: float = 250.0) -> Optional[float]:
    """Compute FRP thermal intensity score [0-100] using logarithmic scaling.

    Maps 0 -> 0, ~25 MW -> ~59, 250+ MW -> 100.
    """
    if frp_val is None or pd.isna(frp_val):
        return None
    try:
        f = float(frp_val)
        if f <= 0.0:
            return 0.0
        # Logarithmic saturation
        score = (math.log10(1.0 + f) / math.log10(1.0 + reference_high_frp)) * 100.0
        return round(float(np.clip(score, 0.0, 100.0)), 1)
    except (ValueError, TypeError):
        return None


def compute_repeat_detections(
    df: pd.DataFrame,
    radius_km: float = 5.0,
    time_window_hours: float = 48.0,
) -> List[int]:
    """Count corroborated repeat detections within radius_km and time_window_hours."""
    n = len(df)
    repeat_counts = [0] * n

    # Extract coordinates and timestamps for valid records
    lats = df["latitude"].values
    lons = df["longitude"].values
    datetimes = df["datetime_utc"].values

    valid_mask = [
        not pd.isna(lats[i]) and not pd.isna(lons[i]) and datetimes[i] is not None
        for i in range(n)
    ]

    valid_indices = [i for i, v in enumerate(valid_mask) if v]
    m = len(valid_indices)

    # Pre-convert datetimes to timestamps in seconds for fast comparison
    timestamps_sec = np.zeros(n)
    for idx in valid_indices:
        dt = datetimes[idx]
        if hasattr(dt, "timestamp"):
            timestamps_sec[idx] = dt.timestamp()
        else:
            timestamps_sec[idx] = 0.0

    window_sec = time_window_hours * 3600.0

    # Efficient pairwise spatial-temporal search
    for i_pos in range(m):
        idx_i = valid_indices[i_pos]
        lat_i, lon_i, t_i = lats[idx_i], lons[idx_i], timestamps_sec[idx_i]

        count = 0
        for j_pos in range(m):
            if i_pos == j_pos:
                continue
            idx_j = valid_indices[j_pos]
            t_j = timestamps_sec[idx_j]

            # Fast temporal filter first
            if abs(t_i - t_j) > window_sec:
                continue

            # Spatial distance filter (rough bounding box check before haversine)
            lat_j, lon_j = lats[idx_j], lons[idx_j]
            if abs(lat_i - lat_j) > (radius_km / 110.0):
                continue

            dist = haversine_distance_km(lat_i, lon_i, lat_j, lon_j)
            if dist <= radius_km:
                count += 1

        repeat_counts[idx_i] = count

    return repeat_counts


def compute_repeat_score(repeat_count: int) -> float:
    """Map repeat count to a corroboration score [0-100].

    Single isolated detection gets baseline 20. 5+ repeats get 100.
    """
    if repeat_count <= 0:
        return 20.0
    score = 20.0 + (repeat_count * 16.0)
    return round(float(np.clip(score, 20.0, 100.0)), 1)


def generate_explanation(
    total_score: float,
    scores: Dict[str, Optional[float]],
    effective_weights: Dict[str, float],
    missing_components: List[str],
    repeat_count: int,
    delta_hours: Optional[float],
    frp_val: Optional[float],
) -> str:
    """Produce clear natural language explanation of the priority score."""
    tier = (
        "HIGH PRIORITY"
        if total_score >= 70.0
        else "ELEVATED PRIORITY"
        if total_score >= 45.0
        else "ROUTINE PRIORITY"
    )

    reasons: List[str] = []

    # Recency reason
    rec_score = scores.get("recency")
    if rec_score is not None:
        if delta_hours is not None and delta_hours < 6.0:
            reasons.append(f"fresh observation ({delta_hours:.1f}h ago)")
        elif delta_hours is not None and delta_hours < 24.0:
            reasons.append(f"acquired within {delta_hours:.1f}h")
        else:
            reasons.append(f"older observation ({delta_hours:.1f}h ago)" if delta_hours else "moderate recency")

    # Confidence reason
    conf_score = scores.get("confidence")
    if conf_score is not None:
        if conf_score >= 80.0:
            reasons.append(f"high sensor confidence ({conf_score:.0f}%)")
        elif conf_score >= 50.0:
            reasons.append(f"nominal confidence ({conf_score:.0f}%)")
        else:
            reasons.append(f"low sensor confidence ({conf_score:.0f}%)")

    # FRP reason
    frp_score = scores.get("frp")
    if frp_score is not None:
        if frp_val is not None and frp_val >= 50.0:
            reasons.append(f"elevated thermal intensity ({frp_val:.1f} MW)")
        elif frp_val is not None:
            reasons.append(f"moderate thermal radiative power ({frp_val:.1f} MW)")

    # Repeat reason
    if repeat_count > 0:
        reasons.append(f"corroborated by {repeat_count} nearby detections within 5km/48h")
    else:
        reasons.append("isolated thermal detection with no immediate cluster repeats")

    reason_str = ", ".join(reasons)
    explanation = f"Observation-Priority Score: {total_score:.1f}/100 ({tier}). "
    if total_score >= 70.0:
        explanation += f"Ranked with high triage priority because of {reason_str}."
    elif total_score >= 45.0:
        explanation += f"Ranked for monitoring because of {reason_str}."
    else:
        explanation += f"Ranked with routine priority due to {reason_str}."

    if missing_components:
        missing_str = ", ".join(missing_components)
        explanation += f" [Note: Missing data ({missing_str}); weights were automatically renormalized over available components]."

    return explanation


def calculate_priority_scores(
    df: pd.DataFrame,
    custom_weights: Optional[Dict[str, float]] = None,
    reference_time: Optional[datetime] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Calculate transparent 0-100 observation-priority scores for all observations.

    Normalizes each component, handles missing data via dynamic weight renormalization,
    and appends component scores, weights, completeness flags, and explanations.
    """
    if df.empty:
        df_empty = df.copy()
        for col in [
            "priority_score",
            "score_incomplete",
            "recency_score",
            "confidence_score",
            "frp_score",
            "repeat_score",
            "repeat_count",
            "explanation",
        ]:
            df_empty[col] = pd.Series(dtype=object)
        return df_empty, {"weights_used": DEFAULT_WEIGHTS}

    df_scored = df.copy()
    weights = dict(DEFAULT_WEIGHTS)
    if custom_weights:
        for k, v in custom_weights.items():
            if k in weights and v >= 0:
                weights[k] = float(v)

    # Normalize weights so they sum to 1.0
    w_sum = sum(weights.values())
    if w_sum > 0:
        weights = {k: v / w_sum for k, v in weights.items()}
    else:
        weights = dict(DEFAULT_WEIGHTS)

    # Determine reference time for recency calculation (latest observation if not provided)
    valid_dts = [dt for dt in df_scored["datetime_utc"] if dt is not None and not pd.isna(dt)]
    if reference_time is None:
        if valid_dts:
            ref_dt = max(valid_dts)
        else:
            ref_dt = datetime.now(timezone.utc)
    else:
        ref_dt = reference_time

    # Compute repeat detections
    repeat_counts = compute_repeat_detections(df_scored, radius_km=5.0, time_window_hours=48.0)
    df_scored["repeat_count"] = repeat_counts

    priority_scores: List[float] = []
    incomplete_flags: List[bool] = []
    rec_scores: List[Optional[float]] = []
    conf_scores: List[Optional[float]] = []
    frp_scores: List[Optional[float]] = []
    rep_scores: List[float] = []
    explanations: List[str] = []

    for idx, row in df_scored.iterrows():
        # 1. Recency
        dt = row.get("datetime_utc")
        if dt is not None and not pd.isna(dt):
            delta_sec = (ref_dt - dt).total_seconds()
            delta_h = max(0.0, delta_sec / 3600.0)
            s_rec = compute_recency_score(delta_h)
        else:
            delta_h = None
            s_rec = None

        # 2. Confidence
        s_conf = compute_confidence_score(row.get("confidence_norm"))

        # 3. FRP
        raw_frp = row.get("frp")
        s_frp = compute_frp_score(raw_frp)

        # 4. Repeat detections
        rep_count = repeat_counts[idx] if isinstance(idx, int) and idx < len(repeat_counts) else 0
        s_rep = compute_repeat_score(rep_count)

        # Missing data handling and weight renormalization
        scores_map = {
            "recency": s_rec,
            "confidence": s_conf,
            "frp": s_frp,
            "repeat": s_rep,
        }

        available_weights = {k: weights[k] for k, v in scores_map.items() if v is not None}
        missing_components = [k for k, v in scores_map.items() if v is None]
        is_incomplete = len(missing_components) > 0

        avail_sum = sum(available_weights.values())
        if avail_sum > 0:
            renormalized_weights = {k: v / avail_sum for k, v in available_weights.items()}
            total = sum(renormalized_weights[k] * scores_map[k] for k in available_weights)
        else:
            renormalized_weights = {}
            total = 0.0

        final_score = round(float(np.clip(total, 0.0, 100.0)), 1)
        expl = generate_explanation(
            total_score=final_score,
            scores=scores_map,
            effective_weights=renormalized_weights,
            missing_components=missing_components,
            repeat_count=rep_count,
            delta_hours=delta_h,
            frp_val=float(raw_frp) if raw_frp is not None and not pd.isna(raw_frp) else None,
        )

        priority_scores.append(final_score)
        incomplete_flags.append(is_incomplete)
        rec_scores.append(s_rec)
        conf_scores.append(s_conf)
        frp_scores.append(s_frp)
        rep_scores.append(s_rep)
        explanations.append(expl)

    df_scored["priority_score"] = priority_scores
    df_scored["score_incomplete"] = incomplete_flags
    df_scored["recency_score"] = rec_scores
    df_scored["confidence_score"] = conf_scores
    df_scored["frp_score"] = frp_scores
    df_scored["repeat_score"] = rep_scores
    df_scored["explanation"] = explanations

    metadata = {
        "weights_configured": weights,
        "reference_time_utc": ref_dt.isoformat(),
        "total_scored": len(df_scored),
        "mean_priority_score": round(float(np.mean(priority_scores)), 1) if priority_scores else 0.0,
        "high_priority_count": sum(1 for s in priority_scores if s >= 70.0),
    }

    return df_scored, metadata
