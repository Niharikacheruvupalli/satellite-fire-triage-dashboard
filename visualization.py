"""Geospatial and analytical visualization module for active fire triage.

Builds interactive Folium maps with priority-graded markers and Plotly charts
for temporal, confidence, and thermal radiative power distributions.
"""

from __future__ import annotations

import html
from typing import Any, Dict, List, Optional

import folium
from folium.plugins import FastMarkerCluster, Fullscreen
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# Color palette for priority tiers
COLOR_HIGH = "#d32f2f"       # High priority / Critical
COLOR_ELEVATED = "#f57c00"   # Elevated / Watchlist
COLOR_ROUTINE = "#2e7d32"    # Routine / Low
COLOR_INVALID = "#757575"    # Invalid / Flagged


def get_marker_color(priority_score: Optional[float], is_valid: bool = True) -> str:
    """Return hex color corresponding to priority score."""
    if not is_valid or priority_score is None or pd.isna(priority_score):
        return COLOR_INVALID
    score = float(priority_score)
    if score >= 70.0:
        return COLOR_HIGH
    elif score >= 45.0:
        return COLOR_ELEVATED
    else:
        return COLOR_ROUTINE


def calculate_marker_radius(frp: Optional[float]) -> float:
    """Scale marker radius based on Fire Radiative Power (FRP in MW)."""
    if frp is None or pd.isna(frp):
        return 5.0
    try:
        val = float(frp)
        if val <= 0:
            return 4.0
        # Smooth logarithmic scaling from 5px to 20px
        radius = 5.0 + 3.0 * np.log1p(val)
        return float(np.clip(radius, 4.0, 22.0))
    except (ValueError, TypeError):
        return 5.0


def create_fire_map(
    df: pd.DataFrame,
    selected_record_id: Optional[str] = None,
    max_markers: int = 1200,
) -> folium.Map:
    """Generate an interactive Folium map populated with fire observation markers."""
    # Determine center coordinates
    valid_coords = df.dropna(subset=["latitude", "longitude"])
    if not valid_coords.empty:
        center_lat = float(valid_coords["latitude"].mean())
        center_lon = float(valid_coords["longitude"].mean())
        # Default zoom level based on coordinate span
        lat_span = valid_coords["latitude"].max() - valid_coords["latitude"].min()
        lon_span = valid_coords["longitude"].max() - valid_coords["longitude"].min()
        max_span = max(lat_span, lon_span)
        if max_span < 1.0:
            zoom_start = 10
        elif max_span < 5.0:
            zoom_start = 7
        elif max_span < 20.0:
            zoom_start = 5
        else:
            zoom_start = 4
    else:
        center_lat, center_lon = 37.0, -95.0
        zoom_start = 4

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles="CartoDB positron",
        control_scale=True,
    )

    # Alternative base layers
    folium.TileLayer("OpenStreetMap", name="Street Map").add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite Imagery",
    ).add_to(m)

    Fullscreen(position="topleft").add_to(m)

    # Feature groups for priority categories
    fg_high = folium.FeatureGroup(name="High Priority (Score >= 70)", show=True)
    fg_elevated = folium.FeatureGroup(name="Elevated (Score 45-69)", show=True)
    fg_routine = folium.FeatureGroup(name="Routine (Score < 45)", show=True)
    fg_flagged = folium.FeatureGroup(name="Flagged / Incomplete", show=False)

    records_to_display = df.head(max_markers)

    for _, row in records_to_display.iterrows():
        lat = row.get("latitude")
        lon = row.get("longitude")
        if pd.isna(lat) or pd.isna(lon):
            continue

        try:
            lat_f = float(lat)
            lon_f = float(lon)
            if not (-90.0 <= lat_f <= 90.0 and -180.0 <= lon_f <= 180.0):
                continue
        except (ValueError, TypeError):
            continue

        rec_id = str(row.get("record_id", ""))
        score = row.get("priority_score")
        score_val = float(score) if score is not None and not pd.isna(score) else 0.0
        is_valid = bool(row.get("is_valid", True))
        frp = row.get("frp")
        conf = row.get("confidence")
        sat = str(row.get("satellite", "Unknown"))
        inst = str(row.get("instrument", "Unknown"))
        dt_str = str(row.get("datetime_utc", "N/A"))
        repeat = row.get("repeat_count", 0)
        expl = str(row.get("explanation", ""))

        color = get_marker_color(score, is_valid)
        radius = calculate_marker_radius(frp)

        # Highlight if selected
        is_selected = selected_record_id and (rec_id == selected_record_id)
        stroke_color = "#000000" if is_selected else color
        stroke_weight = 3 if is_selected else 1
        fill_opacity = 0.9 if is_selected else 0.75

        # Format popup HTML
        badge_bg = color
        popup_html = f"""
        <div style="font-family: Arial, sans-serif; font-size: 12px; width: 280px; line-height: 1.4;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="font-weight: bold; font-size: 13px;">ID: {html.escape(rec_id)}</span>
                <span style="background-color: {badge_bg}; color: white; padding: 2px 7px; border-radius: 4px; font-weight: bold; font-size: 11px;">
                    Score: {score_val:.1f}
                </span>
            </div>
            <div style="margin-bottom: 4px;"><b>Coordinates:</b> {lat_f:.4f}°, {lon_f:.4f}°</div>
            <div style="margin-bottom: 4px;"><b>Time (UTC):</b> {html.escape(dt_str)}</div>
            <div style="margin-bottom: 4px;"><b>Sensor:</b> {html.escape(inst)} ({html.escape(sat)})</div>
            <div style="margin-bottom: 4px;"><b>FRP:</b> {frp if frp is not None and not pd.isna(frp) else 'N/A'} MW</div>
            <div style="margin-bottom: 4px;"><b>Confidence:</b> {html.escape(str(conf))}</div>
            <div style="margin-bottom: 6px;"><b>Local Repeats:</b> {repeat}</div>
            <div style="background-color: #f1f3f4; padding: 6px; border-radius: 4px; font-size: 11px; color: #202124;">
                {html.escape(expl[:220])}{'...' if len(expl) > 220 else ''}
            </div>
        </div>
        """

        marker = folium.CircleMarker(
            location=[lat_f, lon_f],
            radius=radius,
            color=stroke_color,
            weight=stroke_weight,
            fill=True,
            fill_color=color,
            fill_opacity=fill_opacity,
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=f"{rec_id} | Score: {score_val:.1f} | FRP: {frp} MW",
        )

        if not is_valid or row.get("validation_status") == "INVALID":
            marker.add_to(fg_flagged)
        elif score_val >= 70.0:
            marker.add_to(fg_high)
        elif score_val >= 45.0:
            marker.add_to(fg_elevated)
        else:
            marker.add_to(fg_routine)

    fg_high.add_to(m)
    fg_elevated.add_to(m)
    fg_routine.add_to(m)
    fg_flagged.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def create_timeline_chart(df: pd.DataFrame) -> go.Figure:
    """Plot observation frequency over time with priority breakdown."""
    if df.empty or "datetime_utc" not in df.columns:
        fig = go.Figure()
        fig.update_layout(title="No Observation Data Available", template="plotly_white")
        return fig

    valid_times = df.dropna(subset=["datetime_utc"]).copy()
    if valid_times.empty:
        fig = go.Figure()
        fig.update_layout(title="No Valid Timestamps Available", template="plotly_white")
        return fig

    # Map priority category
    def assign_category(score):
        if pd.isna(score):
            return "Flagged / Null"
        s = float(score)
        if s >= 70.0:
            return "High (>= 70)"
        elif s >= 45.0:
            return "Elevated (45-69)"
        return "Routine (< 45)"

    valid_times["Priority Tier"] = valid_times["priority_score"].apply(assign_category)
    valid_times["acq_hour"] = pd.to_datetime(valid_times["datetime_utc"]).dt.floor("3h")

    grouped = (
        valid_times.groupby(["acq_hour", "Priority Tier"])
        .size()
        .reset_index(name="Detections")
    )

    color_map = {
        "High (>= 70)": COLOR_HIGH,
        "Elevated (45-69)": COLOR_ELEVATED,
        "Routine (< 45)": COLOR_ROUTINE,
        "Flagged / Null": COLOR_INVALID,
    }

    fig = px.bar(
        grouped,
        x="acq_hour",
        y="Detections",
        color="Priority Tier",
        color_discrete_map=color_map,
        title="Active Fire Observations Over Time (3-Hour Bins)",
        labels={"acq_hour": "Observation Time (UTC)", "Detections": "Count"},
        template="plotly_white",
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=320,
    )
    return fig


def create_confidence_distribution(df: pd.DataFrame) -> go.Figure:
    """Plot distribution of normalized detection confidence across satellite sensors."""
    if df.empty or "confidence_norm" not in df.columns:
        fig = go.Figure()
        fig.update_layout(title="Confidence Data Unavailable", template="plotly_white")
        return fig

    valid_conf = df.dropna(subset=["confidence_norm"]).copy()
    if valid_conf.empty:
        fig = go.Figure()
        fig.update_layout(title="No Confidence Values to Display", template="plotly_white")
        return fig

    fig = px.histogram(
        valid_conf,
        x="confidence_norm",
        color="instrument",
        nbins=20,
        barmode="overlay",
        opacity=0.75,
        title="Detection Confidence Distribution by Instrument",
        labels={"confidence_norm": "Normalized Confidence (0-100%)", "instrument": "Sensor"},
        color_discrete_sequence=["#1976d2", "#e65100", "#388e3c"],
        template="plotly_white",
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=300,
    )
    return fig


def create_frp_distribution(df: pd.DataFrame) -> go.Figure:
    """Plot Fire Radiative Power (FRP) distribution with log density."""
    if df.empty or "frp" not in df.columns:
        fig = go.Figure()
        fig.update_layout(title="FRP Data Unavailable", template="plotly_white")
        return fig

    valid_frp = df.dropna(subset=["frp"]).copy()
    valid_frp = valid_frp[valid_frp["frp"] >= 0]

    if valid_frp.empty:
        fig = go.Figure()
        fig.update_layout(title="No FRP Data to Display", template="plotly_white")
        return fig

    fig = px.box(
        valid_frp,
        x="instrument",
        y="frp",
        color="instrument",
        points="outliers",
        title="Fire Radiative Power (MW) Distribution",
        labels={"frp": "FRP (MegaWatts)", "instrument": "Sensor"},
        color_discrete_sequence=["#0288d1", "#f57c00", "#7b1fa2"],
        template="plotly_white",
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False,
        height=300,
        yaxis=dict(type="log", title="FRP (MW, log scale)"),
    )
    return fig


def create_observation_breakdown_chart(row: pd.Series) -> go.Figure:
    """Plot individual observation priority score component contribution breakdown."""
    components = ["Recency (30%)", "Confidence (30%)", "FRP (25%)", "Repeats (15%)"]
    scores = [
        row.get("recency_score") or 0.0,
        row.get("confidence_score") or 0.0,
        row.get("frp_score") or 0.0,
        row.get("repeat_score") or 0.0,
    ]

    colors = ["#1976d2", "#0288d1", "#f57c00", "#388e3c"]

    fig = go.Figure(
        data=[
            go.Bar(
                x=components,
                y=scores,
                text=[f"{s:.1f}/100" for s in scores],
                textposition="auto",
                marker_color=colors,
            )
        ]
    )

    total_score = row.get("priority_score", 0.0)
    fig.update_layout(
        title=f"Score Breakdown for {row.get('record_id', '')} (Total: {total_score:.1f}/100)",
        yaxis=dict(range=[0, 105], title="Component Score (0-100)"),
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20),
        height=260,
    )
    return fig
