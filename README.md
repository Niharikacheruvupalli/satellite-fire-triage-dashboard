# Satellite-Derived Active Fire Triage Dashboard

An interactive analyst decision-support dashboard for ingesting, validating, ranking, and visualizing satellite active fire observations (MODIS and VIIRS).

> [!IMPORTANT]
> **Operational Scope & Disclaimer**: This system calculates an **observation-priority score** (0–100) to assist human analysts in triaging high-interest satellite detections. It does **not** confirm ground-truth fire events, predict fire spread velocity, or replace official wildfire emergency dispatch or containment management systems.

---

## 🏛️ Architecture Overview

The system is organized into modular Python components with explicit type hints, error handling, and separation of concerns:

```
satellite-fire-triage/
├── app.py                   # Streamlit web application & multi-view dashboard
├── data_loader.py           # CSV and NASA FIRMS API ingestion & schema normalization
├── validation.py            # Coordinate bounds, sensor schemas, and duplicate audit
├── ranking.py               # 0-100 explainable priority scoring & missing-data renormalization
├── visualization.py         # Folium interactive map & Plotly analytical charts
├── evaluation.py            # Benchmark evaluation on labelled synthetic ground-truth
├── requirements.txt         # Project dependencies
├── README.md                # Documentation and run instructions
├── data/
│   ├── synthetic_fires_modis_viirs.csv  # Realistic bundled dataset with validation edge cases
│   └── evaluation_labeled_sample.csv    # Labelled benchmark sample for evaluation
└── tests/
    ├── test_data_loader.py  # Tests for CSV parser, timestamp conversion, API handler
    ├── test_validation.py   # Tests for coordinate checks, negative FRP, duplicates
    ├── test_ranking.py      # Tests for scoring math, exponential decay, renormalization
    └── test_evaluation.py   # Tests for synthetic evaluation metrics & limitations
```

---

## 🚀 Key Features

### 1. Data Ingestion & Schema Normalization
- Supports file uploads of standard NASA FIRMS CSVs and custom fire observation tables.
- Bundled synthetic dataset with realistic clusters (California Sierra Foothills, Pacific Northwest, Northern Boreal, Kansas Agricultural) and deliberate validation edge cases.
- Optional NASA FIRMS REST API client reading credentials from `FIRMS_MAP_KEY` environment variable. Never hardcodes credentials. Gracefully handles HTTP timeouts, 429 rate limits, and empty responses.
- Normalizes column aliases and instrument differences across MODIS (numeric 0–100% confidence) and VIIRS (`l`, `n`, `h` categorical confidence).

### 2. Rigorous Data Validation (No Silent Removal)
- Validates latitude $\in [-90, 90]$ and longitude $\in [-180, 180]$.
- Validates timestamp parsing into timezone-aware UTC datetime.
- Confirms non-negative Fire Radiative Power (FRP $\ge 0\text{ MW}$) and flags extreme thermal anomalies ($> 5000\text{ MW}$).
- Identifies spatial-temporal duplicates (matching sensor within $\approx 100\text{ m}$ and matching timestamp).
- **Preserves all records**: Each observation is tagged with a validation status (`VALID`, `WARNING`, `INVALID`) and a detailed audit list of issues.

### 3. Transparent, Explainable Observation-Priority Score (0–100)
Observations are prioritized for human analyst review using a transparent heuristic:
- **Recency of Observation (30%)**: Exponential half-life decay ($\approx 12\text{h}$) relative to latest observation.
- **Detection Confidence (30%)**: Sensor-reported confidence normalized to 0–100%.
- **Fire Radiative Power / FRP (25%)**: Logarithmic scaling mapping $0\text{ MW} \to 0$ and $250+\text{ MW} \to 100$.
- **Repeat Cluster Detections (15%)**: Corroboration score based on neighboring detections within $5\text{ km}$ and $48\text{ hours}$.

**Dynamic Weight Renormalization**: If a feature is unrecorded (e.g., missing FRP or confidence), the missing feature is **not** treated as zero evidence. The remaining weights are automatically renormalized to sum to 100%, and the record is flagged with `score_incomplete = True`.

**Plain-Language Rationale**: Every record receives an interpretable sentence detailing why it was prioritized.

### 4. Interactive Geospatial Dashboard
- **Folium Map**: Circle markers color-coded by priority tier (🔴 High $\ge 70$, 🟠 Elevated 45–69, 🟢 Routine $< 45$, ⚪ Flagged) and sized proportionally by FRP. Includes satellite tile switcher.
- **Plotly Analytics**: 3-hour bin observation timeline, sensor confidence distributions, and log-scaled FRP boxplots.
- **Synchronized Filters**: Region presets, satellite selection, FRP slider, priority threshold, and day/night toggles update all views simultaneously.
- **Observation Inspector**: Detail drawer showing component score bar chart, metadata, and plain-language explanation.
- **Data Export**: One-click download of filtered records with all triage scores and reasons as CSV.

### 5. Benchmark Evaluation & Limitations
- Evaluates ranking performance (Precision@K, Recall@K, F1-Score) on a labelled synthetic ground-truth benchmark (`evaluation_labeled_sample.csv`).
- Explicitly clarifies that heuristic ranking is distinct from a trained machine learning model.
- Includes an extensive educational and operational guide covering satellite revisit gaps, cloud mask obscuration, non-wildfire false positives, and sensor footprint distortion.

---

## 🛠️ Setup and Installation

### Prerequisites
- Python 3.11+

### Installation
1. Clone or navigate to the project directory:
   ```bash
   cd satellite-fire-triage
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Running the Application

Launch the Streamlit dashboard:
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

*(Optional) To use the live NASA FIRMS API, set your API key in your terminal or in the sidebar:*
```bash
export FIRMS_MAP_KEY="your_api_key_here"  # Linux / macOS
$env:FIRMS_MAP_KEY="your_api_key_here"    # PowerShell
```

---

## 🧪 Running Automated Tests

Run the full pytest suite:
```bash
pytest -v tests/
```
All 19 test cases verify:
- Field normalization and timestamp conversions
- Validation rules and duplicate detection without silent drops
- Heuristic scoring, exponential recency, and weight renormalization
- Benchmark evaluation metrics and operational limitations coverage
