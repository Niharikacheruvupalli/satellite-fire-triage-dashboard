"""Data ingestion and schema normalization module for satellite fire observations.

Supports MODIS and VIIRS CSV datasets and optional NASA FIRMS REST API ingestion.
"""

from __future__ import annotations

import io
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests


# Canonical schema columns
CANONICAL_COLUMNS: List[str] = [
    "record_id",
    "latitude",
    "longitude",
    "acq_date",
    "acq_time",
    "datetime_utc",
    "satellite",
    "instrument",
    "confidence",
    "confidence_norm",
    "frp",
    "daynight",
    "brightness",
    "source",
    "processing_version",
]

# Common column alias mappings
COLUMN_ALIASES: Dict[str, List[str]] = {
    "latitude": ["latitude", "lat", "y"],
    "longitude": ["longitude", "lon", "long", "x"],
    "acq_date": ["acq_date", "date", "acquisition_date"],
    "acq_time": ["acq_time", "time", "acquisition_time"],
    "satellite": ["satellite", "sat"],
    "instrument": ["instrument", "sensor"],
    "confidence": ["confidence", "conf", "confidence_level"],
    "frp": ["frp", "fire_radiative_power", "power"],
    "daynight": ["daynight", "day_night", "dn"],
    "brightness": ["bright_ti4", "brightness", "bright_t31", "bright_ti5"],
    "processing_version": ["version", "processing_version", "proc_ver"],
}


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Map varying input column headers to canonical names while preserving unmatched columns."""
    renamed: Dict[str, str] = {}
    lower_to_original = {str(c).strip().lower(): c for c in df.columns}

    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in df.columns:
            continue
        for alias in aliases:
            alias_lower = alias.lower()
            if alias_lower in lower_to_original:
                renamed[lower_to_original[alias_lower]] = canonical
                break

    return df.rename(columns=renamed)


def parse_observation_timestamp(date_val: Any, time_val: Any) -> Optional[datetime]:
    """Parse satellite acquisition date and time into a UTC datetime object.

    Handles ISO dates (YYYY-MM-DD), slash dates (YYYY/MM/DD), and time formats:
    '1345', '13:45', '1345.0', '930', '0930'.
    """
    if pd.isna(date_val) or date_val is None:
        return None

    date_str = str(date_val).strip().split(" ")[0].replace("/", "-")
    # Extract YYYY-MM-DD
    date_match = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", date_str)
    if not date_match:
        return None

    year, month, day = map(int, date_match.groups())

    # Format time
    if pd.isna(time_val) or time_val is None:
        hour, minute = 0, 0
    else:
        time_clean = str(time_val).split(".")[0].strip().replace(":", "")
        time_clean = time_clean.zfill(4)
        if len(time_clean) >= 4 and time_clean[:4].isdigit():
            hour = int(time_clean[:2])
            minute = int(time_clean[2:4])
            if hour > 23 or minute > 59:
                return None
        else:
            return None

    try:
        return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
    except (ValueError, OverflowError):
        return None


def normalize_confidence(conf_val: Any, instrument: str = "") -> Optional[float]:
    """Normalize confidence values across MODIS (0-100) and VIIRS (l, n, h) to a 0-100 float."""
    if pd.isna(conf_val) or conf_val is None:
        return None

    conf_str = str(conf_val).strip().lower()

    # VIIRS categorical mapping
    viirs_map = {
        "l": 30.0,
        "low": 30.0,
        "n": 65.0,
        "nominal": 65.0,
        "h": 100.0,
        "high": 100.0,
    }
    if conf_str in viirs_map:
        return viirs_map[conf_str]

    # Numeric handling
    try:
        num = float(conf_str)
        if 0.0 < num < 1.0:  # Normalized float ratio (e.g. 0.85 -> 85%)
            return round(num * 100.0, 1)
        elif 0.0 <= num <= 100.0:
            return round(num, 1)
        return None
    except ValueError:
        return None


def ingest_csv_data(
    source_input: Union[str, bytes, io.BytesIO, io.StringIO],
    source_label: str = "CSV Upload",
    processing_version: str = "N/A",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Ingest CSV data from a file path, raw bytes, or file buffer.

    Returns the normalized dataframe and ingestion metadata.
    """
    retrieval_time = datetime.now(timezone.utc)
    metadata: Dict[str, Any] = {
        "source": source_label,
        "retrieval_time_utc": retrieval_time.isoformat(),
        "processing_version": processing_version,
        "raw_record_count": 0,
        "error": None,
    }

    try:
        if isinstance(source_input, (bytes, bytearray)):
            df = pd.read_csv(io.BytesIO(source_input), comment="#")
        elif isinstance(source_input, (io.BytesIO, io.StringIO)):
            df = pd.read_csv(source_input, comment="#")
        else:
            df = pd.read_csv(source_input, comment="#")
    except Exception as exc:
        metadata["error"] = f"Failed to parse CSV: {exc}"
        return pd.DataFrame(columns=CANONICAL_COLUMNS), metadata

    metadata["raw_record_count"] = len(df)
    if df.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS), metadata

    # Normalize column names
    df = normalize_column_names(df)

    # Assign stable record IDs if missing
    if "record_id" not in df.columns:
        df["record_id"] = [f"REC_{i+1:05d}" for i in range(len(df))]

    # Ensure required canonical columns exist
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    # Parse timestamps
    timestamps = [
        parse_observation_timestamp(r.get("acq_date"), r.get("acq_time"))
        for _, r in df.iterrows()
    ]
    df["datetime_utc"] = timestamps

    # Clean numeric fields
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["frp"] = pd.to_numeric(df["frp"], errors="coerce")
    df["brightness"] = pd.to_numeric(df["brightness"], errors="coerce")

    # Normalize confidence
    conf_norm = [
        normalize_confidence(r.get("confidence"), str(r.get("instrument", "")))
        for _, r in df.iterrows()
    ]
    df["confidence_norm"] = conf_norm

    # Populate provenance fields
    df["source"] = df["source"].fillna(source_label)
    df["processing_version"] = df["processing_version"].fillna(processing_version)
    df["satellite"] = df["satellite"].fillna("Unknown")
    df["instrument"] = df["instrument"].fillna("Unknown")

    return df, metadata


def fetch_nasa_firms_data(
    api_key: Optional[str] = None,
    country_code: str = "USA",
    source_product: str = "VIIRS_SNPP_NRT",
    day_range: int = 1,
    timeout_sec: int = 15,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Fetch active fire detections from NASA FIRMS API.

    API key is read from FIRMS_MAP_KEY environment variable if not passed directly.
    Credentials are never hardcoded.
    """
    key = api_key or os.getenv("FIRMS_MAP_KEY", "").strip()
    retrieval_time = datetime.now(timezone.utc)
    metadata: Dict[str, Any] = {
        "source": f"NASA FIRMS ({source_product})",
        "retrieval_time_utc": retrieval_time.isoformat(),
        "processing_version": "FIRMS API v1",
        "country_code": country_code,
        "day_range": day_range,
        "raw_record_count": 0,
        "error": None,
    }

    if not key:
        metadata["error"] = (
            "NASA FIRMS API key is missing. Set the 'FIRMS_MAP_KEY' environment variable "
            "or enter your MAP KEY in the dashboard sidebar."
        )
        return pd.DataFrame(columns=CANONICAL_COLUMNS), metadata

    # NASA FIRMS country CSV endpoint
    # Format: https://firms.modaps.eosdis.nasa.gov/api/country/csv/[MAP_KEY]/[SOURCE]/[COUNTRY]/[DAY_RANGE]
    url = f"https://firms.modaps.eosdis.nasa.gov/api/country/csv/{key}/{source_product}/{country_code}/{day_range}"

    try:
        resp = requests.get(url, timeout=timeout_sec)
    except requests.exceptions.Timeout:
        metadata["error"] = f"Request to NASA FIRMS timed out after {timeout_sec}s."
        return pd.DataFrame(columns=CANONICAL_COLUMNS), metadata
    except requests.exceptions.RequestException as req_err:
        metadata["error"] = f"Network connection to NASA FIRMS failed: {req_err}"
        return pd.DataFrame(columns=CANONICAL_COLUMNS), metadata

    if resp.status_code == 401 or resp.status_code == 403:
        metadata["error"] = "Invalid or unauthorized NASA FIRMS MAP_KEY."
        return pd.DataFrame(columns=CANONICAL_COLUMNS), metadata
    elif resp.status_code == 429:
        metadata["error"] = "NASA FIRMS API rate limit reached. Please wait before retrying."
        return pd.DataFrame(columns=CANONICAL_COLUMNS), metadata
    elif resp.status_code != 200:
        metadata["error"] = f"NASA FIRMS API returned HTTP status {resp.status_code}: {resp.text[:200]}"
        return pd.DataFrame(columns=CANONICAL_COLUMNS), metadata

    content = resp.text.strip()
    if not content or "latitude" not in content.lower():
        metadata["error"] = "NASA FIRMS returned an empty response or no detections for the selected query."
        return pd.DataFrame(columns=CANONICAL_COLUMNS), metadata

    df, sub_meta = ingest_csv_data(
        io.StringIO(content),
        source_label=f"NASA FIRMS ({source_product} - {country_code})",
        processing_version=source_product,
    )
    metadata["raw_record_count"] = sub_meta["raw_record_count"]
    return df, metadata
