"""Data validation engine for satellite-derived active fire observations.

Enforces coordinate bounds, schema integrity, sensor value ranges,
and duplicate identification without silently discarding invalid records.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd


# Validation status constants
STATUS_VALID = "VALID"
STATUS_WARNING = "WARNING"
STATUS_INVALID = "INVALID"


def validate_fire_records(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Validate incoming fire observations and attach validation audit metadata.

    Preserves every original record, annotating each row with:
    - validation_status: 'VALID', 'WARNING', or 'INVALID'
    - is_valid: bool (True for VALID / WARNING, False for INVALID)
    - validation_issues: List of explanatory strings
    - is_duplicate: bool (True if duplicate detection matches earlier record)

    Returns the augmented DataFrame and an aggregate validation summary report.
    """
    if df.empty:
        summary = {
            "total_records": 0,
            "valid_records": 0,
            "warning_records": 0,
            "invalid_records": 0,
            "duplicate_records": 0,
            "issue_breakdown": {},
        }
        df_empty = df.copy()
        df_empty["validation_status"] = pd.Series(dtype=str)
        df_empty["is_valid"] = pd.Series(dtype=bool)
        df_empty["validation_issues"] = pd.Series(dtype=object)
        df_empty["is_duplicate"] = pd.Series(dtype=bool)
        return df_empty, summary

    df_validated = df.copy()
    statuses: List[str] = []
    is_valid_flags: List[bool] = []
    all_issues: List[List[str]] = []
    issue_counts: Dict[str, int] = {}

    def log_issue(issue_list: List[str], issue_code: str, issue_desc: str) -> None:
        issue_list.append(issue_desc)
        issue_counts[issue_code] = issue_counts.get(issue_code, 0) + 1

    # Track duplicates based on spatial (rounded to 3 decimals ~110m) and temporal match
    seen_keys: Set[Tuple[float, float, str, str]] = set()
    duplicate_flags: List[bool] = []

    for _, row in df_validated.iterrows():
        row_issues: List[str] = []
        is_invalid = False
        is_warning = False

        lat = row.get("latitude")
        lon = row.get("longitude")
        dt_val = row.get("datetime_utc")
        conf = row.get("confidence")
        conf_norm = row.get("confidence_norm")
        frp = row.get("frp")
        sat = str(row.get("satellite", "")).strip()

        # 1. Coordinate Bounds Validation
        if pd.isna(lat) or lat is None:
            is_invalid = True
            log_issue(row_issues, "MISSING_LATITUDE", "Latitude coordinate is missing.")
        elif not (-90.0 <= float(lat) <= 90.0):
            is_invalid = True
            log_issue(row_issues, "INVALID_LATITUDE_RANGE", f"Latitude {lat}° is out of valid range [-90, 90].")

        if pd.isna(lon) or lon is None:
            is_invalid = True
            log_issue(row_issues, "MISSING_LONGITUDE", "Longitude coordinate is missing.")
        elif not (-180.0 <= float(lon) <= 180.0):
            is_invalid = True
            log_issue(row_issues, "INVALID_LONGITUDE_RANGE", f"Longitude {lon}° is out of valid range [-180, 180].")

        # 2. Timestamp Validation
        if pd.isna(dt_val) or dt_val is None:
            is_invalid = True
            log_issue(row_issues, "UNPARSEABLE_TIMESTAMP", f"Observation acquisition time '{row.get('acq_date')} {row.get('acq_time')}' could not be parsed.")

        # 3. Confidence Range & Schema Validation
        if pd.isna(conf_norm) or conf_norm is None:
            is_warning = True
            log_issue(row_issues, "MISSING_OR_UNRECOGNIZED_CONFIDENCE", f"Confidence value '{conf}' is missing or outside sensor specifications.")
        elif not (0.0 <= float(conf_norm) <= 100.0):
            is_warning = True
            log_issue(row_issues, "CONFIDENCE_OUT_OF_BOUNDS", f"Normalized confidence {conf_norm} is outside [0, 100].")

        # 4. Fire Radiative Power (FRP) Validation
        if pd.isna(frp) or frp is None:
            is_warning = True
            log_issue(row_issues, "MISSING_FRP", "Fire Radiative Power (FRP) is unrecorded or null.")
        else:
            try:
                frp_float = float(frp)
                if frp_float < 0.0:
                    is_invalid = True
                    log_issue(row_issues, "NEGATIVE_FRP", f"Negative Fire Radiative Power ({frp_float} MW) violates physical constraints.")
                elif frp_float > 5000.0:
                    is_warning = True
                    log_issue(row_issues, "EXTREME_FRP_ANOMALY", f"Extreme FRP value ({frp_float} MW) warrants manual analyst verification.")
            except (ValueError, TypeError):
                is_invalid = True
                log_issue(row_issues, "MALFORMED_FRP", f"FRP value '{frp}' is not a valid numeric value.")

        # 5. Duplicate Detection
        is_dup = False
        if not is_invalid and not pd.isna(lat) and not pd.isna(lon) and dt_val is not None:
            # Round coordinates to 3 decimals (~110 meters)
            dup_key = (
                round(float(lat), 3),
                round(float(lon), 3),
                dt_val.isoformat() if hasattr(dt_val, "isoformat") else str(dt_val),
                sat.lower(),
            )
            if dup_key in seen_keys:
                is_dup = True
                is_warning = True
                log_issue(row_issues, "DUPLICATE_DETECTION", f"Duplicate detection: matching satellite {sat} within 100m at {dt_val}.")
            else:
                seen_keys.add(dup_key)
        duplicate_flags.append(is_dup)

        # Assign composite status
        if is_invalid:
            status = STATUS_INVALID
            valid_flag = False
        elif is_warning:
            status = STATUS_WARNING
            valid_flag = True
        else:
            status = STATUS_VALID
            valid_flag = True

        statuses.append(status)
        is_valid_flags.append(valid_flag)
        all_issues.append(row_issues)

    df_validated["validation_status"] = statuses
    df_validated["is_valid"] = is_valid_flags
    df_validated["validation_issues"] = all_issues
    df_validated["is_duplicate"] = duplicate_flags

    # Summary report
    summary = {
        "total_records": len(df_validated),
        "valid_records": sum(1 for s in statuses if s == STATUS_VALID),
        "warning_records": sum(1 for s in statuses if s == STATUS_WARNING),
        "invalid_records": sum(1 for s in statuses if s == STATUS_INVALID),
        "duplicate_records": sum(duplicate_flags),
        "issue_breakdown": issue_counts,
    }

    return df_validated, summary
