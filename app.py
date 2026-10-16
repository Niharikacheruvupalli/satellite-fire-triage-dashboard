"""Main Streamlit Application: Satellite-Derived Active Fire Triage Dashboard.

Interactive analyst decision-support system for ingesting, validating,
ranking, and visualizing satellite active fire observations.
"""

from __future__ import annotations

import io
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from data_loader import (
    CANONICAL_COLUMNS,
    fetch_nasa_firms_data,
    ingest_csv_data,
)
from evaluation import OPERATIONAL_LIMITATIONS, evaluate_sample_benchmark
from ranking import DEFAULT_WEIGHTS, calculate_priority_scores
from validation import STATUS_INVALID, STATUS_VALID, STATUS_WARNING, validate_fire_records
from visualization import (
    COLOR_ELEVATED,
    COLOR_HIGH,
    COLOR_INVALID,
    COLOR_ROUTINE,
    create_confidence_distribution,
    create_fire_map,
    create_frp_distribution,
    create_observation_breakdown_chart,
    create_timeline_chart,
)


# Application configuration
st.set_page_config(
    page_title="Satellite Active Fire Triage",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished interface with blue accents and compact cards
st.markdown(
    """
    <style>
    .main-header {
        font-family: 'Segoe UI', Helvetica, Arial, sans-serif;
        color: #0d47a1;
        font-size: 26px;
        font-weight: 700;
        margin-bottom: 2px;
    }
    .sub-header {
        color: #455a64;
        font-size: 14px;
        margin-bottom: 16px;
    }
    .disclaimer-banner {
        background-color: #e3f2fd;
        border-left: 4px solid #1976d2;
        padding: 10px 14px;
        border-radius: 4px;
        font-size: 13px;
        color: #0d47a1;
        margin-bottom: 18px;
    }
    .kpi-container {
        display: flex;
        gap: 12px;
        margin-bottom: 16px;
    }
    .kpi-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 12px 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        flex: 1;
    }
    .kpi-label {
        font-size: 12px;
        color: #5f6368;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 24px;
        font-weight: 700;
        color: #1a73e8;
    }
    .kpi-sub {
        font-size: 11px;
        color: #80868b;
    }
    .tier-badge-high {
        background-color: #d32f2f;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .tier-badge-elevated {
        background-color: #f57c00;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .tier-badge-routine {
        background-color: #2e7d32;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_bundled_data() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load the bundled demonstration dataset."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    bundled_path = os.path.join(base_dir, "data", "synthetic_fires_modis_viirs.csv")
    return ingest_csv_data(
        bundled_path,
        source_label="Bundled Synthetic Dataset (MODIS & VIIRS)",
        processing_version="Synthetic v2.1",
    )


# --- SIDEBAR: DATA INGESTION & FILTER CONTROLS ---
with st.sidebar:
    st.markdown("### 🔥 Fire Triage Settings")
    st.markdown("---")

    st.markdown("#### 1. Ingestion Source")
    data_source_mode = st.radio(
        "Select Data Source:",
        ["Bundled Synthetic Data", "Upload CSV File", "NASA FIRMS API (Optional)"],
        index=0,
    )

    df_raw: pd.DataFrame = pd.DataFrame()
    ingestion_meta: Dict[str, Any] = {}

    if data_source_mode == "Bundled Synthetic Data":
        df_raw, ingestion_meta = load_bundled_data()
        st.success(f"Loaded {len(df_raw)} bundled observations.")

    elif data_source_mode == "Upload CSV File":
        uploaded_file = st.file_uploader(
            "Upload MODIS or VIIRS CSV:",
            type=["csv", "txt"],
            help="Accepts standard NASA FIRMS CSV or normalized CSV files.",
        )
        if uploaded_file is not None:
            df_raw, ingestion_meta = ingest_csv_data(
                uploaded_file.getvalue(),
                source_label=f"Uploaded: {uploaded_file.name}",
                processing_version="User Upload",
            )
            st.success(f"Parsed {len(df_raw)} records from {uploaded_file.name}")
        else:
            st.info("Upload a CSV file or switch to Bundled Synthetic Data.")

    elif data_source_mode == "NASA FIRMS API (Optional)":
        st.caption("Live query from NASA EOSDIS FIRMS API")
        api_key_input = st.text_input(
            "FIRMS MAP_KEY:",
            value=os.getenv("FIRMS_MAP_KEY", ""),
            type="password",
            help="Get your free key from https://firms.modaps.eosdis.nasa.gov/api/map_key/",
        )
        country = st.text_input("Country Code (ISO 3-letter):", value="USA")
        product = st.selectbox(
            "Sensor / Product:",
            ["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "MODIS_NRT"],
            index=0,
        )
        day_range = st.slider("Past Days:", min_value=1, max_value=5, value=1)

        if st.button("Fetch FIRMS Data"):
            with st.spinner("Querying NASA FIRMS API..."):
                df_raw, ingestion_meta = fetch_nasa_firms_data(
                    api_key=api_key_input,
                    country_code=country.strip().upper(),
                    source_product=product,
                    day_range=day_range,
                )
                if ingestion_meta.get("error"):
                    st.error(ingestion_meta["error"])
                else:
                    st.success(f"Retrieved {len(df_raw)} records from NASA FIRMS.")

    st.markdown("---")
    st.markdown("#### 2. Scoring Configuration")
    with st.expander("Adjust Heuristic Weights", expanded=False):
        st.caption("Weights will be automatically normalized to sum to 100%.")
        w_rec = st.slider("Recency Weight", 0.0, 1.0, DEFAULT_WEIGHTS["recency"], 0.05)
        w_conf = st.slider("Confidence Weight", 0.0, 1.0, DEFAULT_WEIGHTS["confidence"], 0.05)
        w_frp = st.slider("FRP (Intensity) Weight", 0.0, 1.0, DEFAULT_WEIGHTS["frp"], 0.05)
        w_rep = st.slider("Repeat Cluster Weight", 0.0, 1.0, DEFAULT_WEIGHTS["repeat"], 0.05)
        custom_weights = {
            "recency": w_rec,
            "confidence": w_conf,
            "frp": w_frp,
            "repeat": w_rep,
        }
        total_w = sum(custom_weights.values())
        st.caption(f"Raw weight sum: {total_w:.2f} (normalized to 1.0 internally)")

    st.markdown("---")
    st.markdown("#### 3. Spatial & Filter Controls")

    # Region preset
    region_preset = st.selectbox(
        "Geographic Preset:",
        [
            "All Regions",
            "California Sierra Foothills",
            "Pacific Northwest",
            "Northern Boreal (Alberta/NWT)",
            "Kansas Flint Hills (Ag)",
        ],
        index=0,
    )

    filter_satellite = st.multiselect(
        "Satellite Instrument:",
        options=["MODIS", "VIIRS", "Aqua", "Terra", "NOAA-20", "NOAA-21", "Suomi-NPP"],
        default=[],
        help="Leave empty to include all satellites.",
    )

    min_frp_filter = st.slider("Minimum FRP (MW):", min_value=0.0, max_value=200.0, value=0.0, step=5.0)
    min_priority_filter = st.slider("Minimum Priority Score:", min_value=0.0, max_value=90.0, value=0.0, step=5.0)
    daynight_filter = st.selectbox("Day / Night:", ["All Observations", "Day Only (D)", "Night Only (N)"])
    show_invalid = st.checkbox("Include Flagged / Invalid Records in Table", value=True)


# --- DATA PIPELINE: VALIDATION & SCORING ---
df_validated, val_summary = validate_fire_records(df_raw)
df_scored, score_meta = calculate_priority_scores(df_validated, custom_weights=custom_weights)

# --- FILTERING LOGIC ---
df_filtered = df_scored.copy()

if not df_filtered.empty:
    # 1. Geographic preset filter
    if region_preset == "California Sierra Foothills":
        df_filtered = df_filtered[
            (df_filtered["latitude"] >= 38.0)
            & (df_filtered["latitude"] <= 41.0)
            & (df_filtered["longitude"] >= -123.0)
            & (df_filtered["longitude"] <= -120.0)
        ]
    elif region_preset == "Pacific Northwest":
        df_filtered = df_filtered[
            (df_filtered["latitude"] >= 42.0)
            & (df_filtered["latitude"] <= 49.0)
            & (df_filtered["longitude"] >= -125.0)
            & (df_filtered["longitude"] <= -119.0)
        ]
    elif region_preset == "Northern Boreal (Alberta/NWT)":
        df_filtered = df_filtered[
            (df_filtered["latitude"] >= 53.0)
            & (df_filtered["latitude"] <= 65.0)
            & (df_filtered["longitude"] >= -125.0)
            & (df_filtered["longitude"] <= -110.0)
        ]
    elif region_preset == "Kansas Flint Hills (Ag)":
        df_filtered = df_filtered[
            (df_filtered["latitude"] >= 36.5)
            & (df_filtered["latitude"] <= 39.5)
            & (df_filtered["longitude"] >= -98.0)
            & (df_filtered["longitude"] <= -95.0)
        ]

    # 2. Satellite filter
    if filter_satellite:
        sat_mask = (
            df_filtered["satellite"].astype(str).isin(filter_satellite)
            | df_filtered["instrument"].astype(str).isin(filter_satellite)
        )
        df_filtered = df_filtered[sat_mask]

    # 3. FRP filter (preserve records with NaN if requested, else check value)
    if min_frp_filter > 0.0:
        df_filtered = df_filtered[df_filtered["frp"].fillna(0.0) >= min_frp_filter]

    # 4. Priority Score filter
    if min_priority_filter > 0.0:
        df_filtered = df_filtered[df_filtered["priority_score"].fillna(0.0) >= min_priority_filter]

    # 5. Day/Night filter
    if daynight_filter == "Day Only (D)":
        df_filtered = df_filtered[df_filtered["daynight"].astype(str).str.upper() == "D"]
    elif daynight_filter == "Night Only (N)":
        df_filtered = df_filtered[df_filtered["daynight"].astype(str).str.upper() == "N"]

    # 6. Flagged records visibility
    if not show_invalid:
        df_filtered = df_filtered[df_filtered["validation_status"] != STATUS_INVALID]


# --- TOP MAIN HEADER ---
st.markdown('<div class="main-header">🛰️ Satellite-Derived Active Fire Triage Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Analyst decision-support system for prioritizing satellite thermal anomaly observations.</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="disclaimer-banner">
        <b>OPERATIONAL NOTICE & SCOPE DISCLAIMER:</b> This dashboard calculates a heuristic
        <b>observation-priority score</b> to assist human analysts in reviewing high-interest satellite detections.
        It does <b>not</b> confirm ground-truth fire events, predict fire propagation velocity, or replace official
        fire management emergency dispatch systems.
    </div>
    """,
    unsafe_allow_html=True,
)

# --- KPI METRIC CARDS ---
total_loaded = len(df_raw)
valid_count = val_summary.get("valid_records", 0)
warning_count = val_summary.get("warning_records", 0)
invalid_count = val_summary.get("invalid_records", 0)
high_priority_count = (
    sum(1 for s in df_filtered["priority_score"] if s >= 70.0) if not df_filtered.empty else 0
)
filtered_count = len(df_filtered)

kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
with kpi_col1:
    st.metric(label="TOTAL OBSERVATIONS", value=total_loaded, delta=f"{filtered_count} after filters")
with kpi_col2:
    st.metric(label="VALID OBSERVATIONS", value=valid_count, delta=f"{warning_count} with warnings")
with kpi_col3:
    st.metric(label="HIGH-PRIORITY (>= 70)", value=high_priority_count, delta="Immediate Analyst Review")
with kpi_col4:
    st.metric(label="FLAGGED / INVALID", value=invalid_count, delta="Preserved in audit log", delta_color="inverse")

st.markdown("<br>", unsafe_allow_html=True)


# --- MAIN TABS ---
tab_map, tab_analytics, tab_validation, tab_evaluation, tab_limitations = st.tabs(
    [
        "🗺️ Map & Triage Table",
        "📊 Analytical Trends",
        "🔍 Data Quality Audit",
        "🧪 Benchmark Evaluation",
        "📖 Operational Limitations",
    ]
)

# --- TAB 1: MAP & TRIAGE TABLE ---
with tab_map:
    st.markdown("#### Interactive Geospatial Triage Map")
    st.caption(
        "Markers are color-coded by observation-priority score (🔴 High >=70, 🟠 Elevated 45-69, 🟢 Routine <45, ⚪ Flagged). "
        "Circle radius scales with Fire Radiative Power (MW)."
    )

    map_col, inspect_col = st.columns([1.6, 1.0])

    with map_col:
        fire_map = create_fire_map(df_filtered)
        st_folium(fire_map, width="100%", height=480, returned_objects=[])

    with inspect_col:
        st.markdown("#### 🔬 Observation Inspector")
        if not df_filtered.empty:
            sorted_records = df_filtered.sort_values(by="priority_score", ascending=False)
            record_options = sorted_records["record_id"].tolist()
            selected_id = st.selectbox("Select Observation to Inspect:", record_options, index=0)

            selected_row = df_filtered[df_filtered["record_id"] == selected_id].iloc[0]

            score_val = selected_row.get("priority_score", 0.0)
            score_color = COLOR_HIGH if score_val >= 70.0 else COLOR_ELEVATED if score_val >= 45.0 else COLOR_ROUTINE

            st.markdown(
                f"""
                <div style="background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 6px; padding: 12px; margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 700; font-size: 15px;">{selected_id}</span>
                        <span style="background-color: {score_color}; color: white; padding: 4px 10px; border-radius: 4px; font-weight: 700;">
                            Priority Score: {score_val:.1f}/100
                        </span>
                    </div>
                    <div style="font-size: 12px; color: #495057; margin-top: 6px;">
                        <b>Location:</b> {selected_row.get('latitude', 0.0):.4f}°, {selected_row.get('longitude', 0.0):.4f}° | 
                        <b>Acquired:</b> {selected_row.get('datetime_utc', 'N/A')}
                    </div>
                    <div style="font-size: 12px; color: #495057; margin-top: 3px;">
                        <b>Sensor:</b> {selected_row.get('instrument', 'N/A')} ({selected_row.get('satellite', 'N/A')}) | 
                        <b>FRP:</b> {selected_row.get('frp', 'N/A')} MW | 
                        <b>Confidence:</b> {selected_row.get('confidence', 'N/A')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Rationale explanation box
            st.info(f"**Triage Rationale:** {selected_row.get('explanation', '')}")

            # Component breakdown chart
            breakdown_fig = create_observation_breakdown_chart(selected_row)
            st.plotly_chart(breakdown_fig, use_container_width=True)

            if selected_row.get("validation_issues"):
                st.warning(f"Validation Notes: {'; '.join(selected_row['validation_issues'])}")
        else:
            st.info("No records match the current filter criteria.")

    st.markdown("---")
    st.markdown(f"#### 📋 Ranked Observations Table ({len(df_filtered)} records)")

    if not df_filtered.empty:
        display_cols = [
            "record_id",
            "priority_score",
            "latitude",
            "longitude",
            "datetime_utc",
            "satellite",
            "instrument",
            "confidence",
            "frp",
            "repeat_count",
            "validation_status",
            "score_incomplete",
        ]
        available_display_cols = [c for c in display_cols if c in df_filtered.columns]

        table_df = df_filtered[available_display_cols].sort_values(by="priority_score", ascending=False)
        st.dataframe(
            table_df,
            use_container_width=True,
            column_config={
                "priority_score": st.column_config.ProgressColumn(
                    "Priority Score",
                    help="Heuristic priority score (0-100)",
                    format="%.1f",
                    min_value=0,
                    max_value=100,
                ),
                "latitude": st.column_config.NumberColumn("Latitude", format="%.4f"),
                "longitude": st.column_config.NumberColumn("Longitude", format="%.4f"),
                "frp": st.column_config.NumberColumn("FRP (MW)", format="%.1f"),
            },
            hide_index=True,
        )

        # Export CSV Button
        csv_buffer = io.StringIO()
        df_filtered.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Export Filtered Triage Records (CSV)",
            data=csv_buffer.getvalue(),
            file_name=f"satellite_fire_triage_export_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
    else:
        st.info("No records to display.")


# --- TAB 2: ANALYTICAL TRENDS ---
with tab_analytics:
    st.markdown("#### Observation Analytics & Sensor Distributions")

    timeline_fig = create_timeline_chart(df_filtered)
    st.plotly_chart(timeline_fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        conf_fig = create_confidence_distribution(df_filtered)
        st.plotly_chart(conf_fig, use_container_width=True)
    with c2:
        frp_fig = create_frp_distribution(df_filtered)
        st.plotly_chart(frp_fig, use_container_width=True)

    # Statistical summary table
    if not df_filtered.empty:
        st.markdown("##### Summary Statistics by Instrument")
        stats_df = (
            df_filtered.groupby("instrument")
            .agg(
                Total_Count=("record_id", "count"),
                Mean_Priority=("priority_score", "mean"),
                Median_FRP=("frp", "median"),
                Max_FRP=("frp", "max"),
                Mean_Confidence=("confidence_norm", "mean"),
            )
            .round(1)
            .reset_index()
        )
        st.table(stats_df)


# --- TAB 3: DATA QUALITY AUDIT ---
with tab_validation:
    st.markdown("#### Data Validation & Quality Audit Report")
    st.caption(
        "Incoming observations are rigorously verified for coordinate bounds, time integrity, "
        "and physical sensor constraints. Non-conforming records are flagged with explanatory codes "
        "and preserved for analyst auditing."
    )

    q1, q2, q3, q4 = st.columns(4)
    with q1:
        st.metric("Total Records Ingested", val_summary.get("total_records", 0))
    with q2:
        st.metric("Clean Records (Valid)", val_summary.get("valid_records", 0))
    with q3:
        st.metric("Warnings / Anomalies", val_summary.get("warning_records", 0))
    with q4:
        st.metric("Invalid Records", val_summary.get("invalid_records", 0))

    st.markdown("##### Issue Breakdown")
    breakdown = val_summary.get("issue_breakdown", {})
    if breakdown:
        breakdown_df = pd.DataFrame(
            [{"Issue Code": k, "Occurrences": v} for k, v in breakdown.items()]
        )
        st.table(breakdown_df)
    else:
        st.success("Zero data validation issues detected in this dataset.")

    st.markdown("##### Flagged / Incomplete Records Log")
    flagged_records = df_scored[df_scored["validation_status"].isin([STATUS_WARNING, STATUS_INVALID])]
    if not flagged_records.empty:
        st.dataframe(
            flagged_records[
                [
                    "record_id",
                    "validation_status",
                    "latitude",
                    "longitude",
                    "acq_date",
                    "acq_time",
                    "satellite",
                    "instrument",
                    "frp",
                    "validation_issues",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No flagged or invalid records present.")


# --- TAB 4: BENCHMARK EVALUATION ---
with tab_evaluation:
    st.markdown("#### Synthetic Benchmark Evaluation")
    st.markdown(
        """
        <div class="disclaimer-banner">
            <b>EVALUATION METHODOLOGY CAVEAT:</b> This evaluation assesses how well the heuristic
            priority formula ranks observations against an <i>explicitly defined, expert-annotated synthetic sample</i>
            (<code>evaluation_labeled_sample.csv</code>). It does <b>NOT</b> represent a trained machine learning
            model or general validation against unobserved global wildfire complexes.
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        eval_df, eval_metrics = evaluate_sample_benchmark()

        em1, em2, em3, em4 = st.columns(4)
        with em1:
            st.metric("Benchmark Sample Size", eval_metrics["total_benchmark_records"])
        with em2:
            st.metric("Precision @ 5 (P@5)", f"{eval_metrics['precision_at_k'].get('P@5', 0.0) * 100:.1f}%")
        with em3:
            st.metric("Precision @ 10 (P@10)", f"{eval_metrics['precision_at_k'].get('P@10', 0.0) * 100:.1f}%")
        with em4:
            st.metric("F1-Score (Threshold >= 60)", f"{eval_metrics['f1_score']:.3f}")

        st.markdown("##### Ground-Truth Tier Priority Score Separation")
        tier_stats = pd.DataFrame.from_dict(eval_metrics["tier_score_distribution"], orient="index")
        st.table(tier_stats)

        st.markdown("##### Labelled Benchmark Records & Triage Ranking")
        st.dataframe(
            eval_df[
                [
                    "record_id",
                    "priority_score",
                    "ground_truth_tier",
                    "satellite",
                    "instrument",
                    "frp",
                    "confidence",
                    "analyst_notes",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    except Exception as eval_err:
        st.error(f"Failed to run benchmark evaluation: {eval_err}")


# --- TAB 5: OPERATIONAL LIMITATIONS ---
with tab_limitations:
    st.markdown("#### Satellite Remote Sensing Constraints & Limitations")
    st.markdown(
        "Remote-sensing active fire triage must be conducted with full awareness of orbital, "
        "atmospheric, and sensor-specific physical limitations."
    )

    for key, item in OPERATIONAL_LIMITATIONS.items():
        with st.expander(f"⚠️ {item['title']}", expanded=True):
            st.write(item["description"])

    st.markdown("---")
    st.caption("Satellite-Derived Active Fire Triage System | Developed for Hackathon MVP")
