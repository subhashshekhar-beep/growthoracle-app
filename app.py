# ============================
# app.py — GrowthOracle AI — Next Gen (Full)
# PART 1/5: Imports, Config, Logger, Validation Core
# ============================
import os, io, re, sys, math, json, time, logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd
import streamlit as st

# Optional deps with better error handling
try:
    import yaml
except ImportError:
    yaml = None

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except ImportError:
    _HAS_ST = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.io import to_html
    _HAS_PLOTLY = True
except ImportError:
    px = None
    go = None
    to_html = None
    _HAS_PLOTLY = False

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    _HAS_STM = True
except ImportError:
    _HAS_STM = False

# ---- Page ----
st.set_page_config(
    page_title="GrowthOracle AI — Next Gen",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)
st.title("GrowthOracle AI — Next Gen")
st.caption("Time-aware insights • English SEO-title cannibalization • Interactive analytics • Explainable recommendations")

# ---- Logger ----
@st.cache_resource
def get_logger(level=logging.INFO):
    logger = logging.getLogger("growthoracle")
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = get_logger()

# ---- Defaults / Config ----
_DEFAULT_CONFIG = {
    "thresholds": {
        "striking_distance_min": 11, "striking_distance_max": 20,
        "ctr_deficit_pct": 1.0, "similarity_threshold": 0.60,
        "min_impressions": 100, "min_clicks": 10
    },
    "presets": {
        "Conservative": {"striking_distance_min": 15, "striking_distance_max": 25, "ctr_deficit_pct": 2.0, "similarity_threshold": 0.7, "min_impressions": 300, "min_clicks": 30},
        "Standard":     {"striking_distance_min": 11, "striking_distance_max": 20, "ctr_deficit_pct": 1.0, "similarity_threshold": 0.6, "min_impressions": 100, "min_clicks": 10},
        "Aggressive":   {"striking_distance_min": 8,  "striking_distance_max": 18, "ctr_deficit_pct": 0.5, "similarity_threshold": 0.5, "min_impressions": 50,  "min_clicks": 5}
    },
    "expected_ctr_by_rank": {1: 0.28, 2: 0.15, 3: 0.11, 4: 0.08, 5: 0.07, 6: 0.05, 7: 0.045, 8: 0.038, 9: 0.032},
    "performance": {"sample_row_limit": 350_000, "seed": 42},
    "defaults": {"date_lookback_days": 60}
}

@st.cache_resource
def load_config():
    cfg = _DEFAULT_CONFIG.copy()
    if yaml is not None:
        for candidate in ["config.yaml", "growthoracle.yaml", "settings.yaml"]:
            if os.path.exists(candidate):
                try:
                    with open(candidate, "r", encoding="utf-8") as f:
                        user_cfg = yaml.safe_load(f) or {}
                    # Deep merge for nested dictionaries
                    for k, v in user_cfg.items():
                        if isinstance(v, dict) and k in cfg and isinstance(cfg[k], dict):
                            cfg[k].update(v)
                        else:
                            cfg[k] = v
                    logger.info(f"Loaded configuration from {candidate}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load {candidate}: {e}")
    return cfg

CONFIG = load_config()

# ---- Validation Core ----
@dataclass
class ValidationMessage:
    category: str  # "Critical" | "Warning" | "Info"
    code: str
    message: str
    context: Dict[str, Any] = field(default_factory=dict)

class ValidationCollector:
    def __init__(self):
        self.messages: List[ValidationMessage] = []
        self.exceptions: List[Dict[str, Any]] = []
        self.checkpoints: List[Dict[str, Any]] = []
        self.lineage_notes: List[str] = []

    def add(self, category: str, code: str, message: str, **ctx):
        self.messages.append(ValidationMessage(category, code, message, ctx))

    def add_exc(self, where: str, exc: Exception):
        self.exceptions.append({
            "where": where, "type": type(exc).__name__, "message": str(exc),
            "traceback": "\n".join(str(exc).splitlines()[-5:]) if str(exc) else "No traceback"
        })

    def checkpoint(self, name: str, **data):
        self.checkpoints.append({"name": name, **data})

    def quality_score(self) -> float:
        crit = sum(1 for m in self.messages if m.category == "Critical")
        warn = sum(1 for m in self.messages if m.category == "Warning")
        info = sum(1 for m in self.messages if m.category == "Info")
        score = 100 - (25 * crit + 8 * warn + 1 * info)
        return float(max(0, min(100, score)))

    def to_dataframe(self) -> pd.DataFrame:
        if not self.messages:
            return pd.DataFrame(columns=["category", "code", "message", "context"])
        return pd.DataFrame([{
            "category": m.category,
            "code": m.code,
            "message": m.message,
            "context": json.dumps(m.context, ensure_ascii=False)
        } for m in self.messages])

PROCESS_LOG: List[Dict[str, Any]] = []

def log_event(event: str, **kws):
    entry = {"ts": pd.Timestamp.utcnow().isoformat(), "event": event, **kws}
    PROCESS_LOG.append(entry)
    logger.debug(f"[LOG] {event} | {kws}")

def coerce_numeric(series, name: str, vc: ValidationCollector, clamp: Optional[Tuple[float, float]] = None) -> pd.Series:
    """Safely convert series to numeric with validation logging"""
    if series is None or len(series) == 0:
        return pd.Series([], dtype=float)

    s = pd.to_numeric(series, errors="coerce")
    bad = s.isna().sum()
    if bad > 0:
        vc.add("Warning", "NUM_COERCE", f"Non-numeric values coerced to NaN in {name}", bad_rows=int(bad))

    if clamp and len(s) > 0:
        lo, hi = clamp
        # Use .loc to avoid SettingWithCopyWarning
        s_clamped = s.copy()
        out_of_bounds_mask = (s_clamped < lo) | (s_clamped > hi)
        before = out_of_bounds_mask.sum()
        if before > 0:
            s_clamped.loc[out_of_bounds_mask] = s_clamped.clip(lo, hi)
            vc.add("Info", "NUM_CLAMP", f"{name} clipped to bounds", lo=lo, hi=hi, affected=int(before))
            s = s_clamped
    return s

def safe_dt_parse(col: pd.Series, name: str, vc: ValidationCollector) -> pd.Series:
    """Safely parse datetime series with validation logging"""
    if col is None or len(col) == 0:
        return pd.Series([], dtype='datetime64[ns, UTC]')

    parsed = pd.to_datetime(col, errors="coerce", utc=True)
    bad = parsed.isna().sum()
    if bad > 0:
        vc.add("Warning", "DATE_PARSE", f"Unparseable datetime in {name}", bad_rows=int(bad))
    return parsed

def detect_date_cols(df: pd.DataFrame) -> List[str]:
    """Detect potential date columns in dataframe"""
    if df is None or df.empty:
        return []
    return [c for c in df.columns if any(k in c.lower() for k in ["date", "dt", "time", "timestamp", "publish"])]

DEFAULT_VALIDATION_STRICTNESS = "Standard"
MERGE_STRATEGY = {"gsc_x_prod": "left", "ga4_align": "left"}

def add_lineage(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Add data lineage information"""
    if df is None:
        return None
    df = df.copy()
    df["_source"] = source
    return df

# Initialize session state
def init_state_defaults():
    """Initialize session state with defaults"""
    if "config" not in st.session_state:
        st.session_state.config = CONFIG
    if "thresholds" not in st.session_state:
        st.session_state.thresholds = CONFIG["thresholds"].copy()
    if "date_range" not in st.session_state:
        end = date.today()
        start = end - timedelta(days=CONFIG["defaults"]["date_lookback_days"])
        st.session_state.date_range = (start, end)
    if "log_level" not in st.session_state:
        st.session_state.log_level = "INFO"
    if "strictness" not in st.session_state:
        st.session_state.strictness = "Standard"

init_state_defaults()
# ============================
# PART 2/5: Templates, Strong Readers, Date Standardizer, Mapping, UI State
# ============================

def _make_template_production():
    """Create production data template"""
    return pd.DataFrame({
        "Msid": [101, 102, 103],
        "Title": ["Budget 2025 highlights explained", "IPL 2025 schedule & squads", "Monsoon updates: city-by-city guide"],
        "Path": ["/business/budget-2025/highlights", "/sports/cricket/ipl-2025/schedule", "/news/monsoon/guide"],
        "Publish Time": ["2025-08-01 09:15:00", "2025-08-10 18:30:00", "2025-09-01 07:00:00"]
    })

def _make_template_ga4():
    """Create GA4 data template"""
    return pd.DataFrame({
        "customEvent:msid": [101, 101, 102, 102, 103],
        "date": ["2025-08-01", "2025-08-02", "2025-08-10", "2025-08-11", "2025-09-01"],
        "screenPageViews": [5000, 6000, 15000, 12000, 7000],
        "totalUsers": [4000, 4500, 10000, 8000, 5200],
        "userEngagementDuration": [52.3, 48.2, 41.0, 44.7, 63.1],
        "bounceRate": [0.42, 0.45, 0.51, 0.49, 0.38]
    })

def _make_template_gsc():
    """Create GSC data template"""
    return pd.DataFrame({
        "Date": ["2025-08-01", "2025-08-02", "2025-08-10", "2025-08-11", "2025-09-01"],
        "Page": [
            "https://example.com/business/budget-2025/highlights/101.cms",
            "https://example.com/business/budget-2025/highlights/101.cms",
            "https://example.com/sports/cricket/ipl-2025/schedule/102.cms",
            "https://example.com/sports/cricket/ipl-2025/schedule/102.cms",
            "https://example.com/news/monsoon/guide/103.cms"
        ],
        "Query": ["budget 2025", "budget highlights", "ipl 2025 schedule", "ipl squads", "monsoon city guide"],
        "Clicks": [200, 240, 1200, 1100, 300],
        "Impressions": [5000, 5500, 40000, 38000, 7000],
        "CTR": [0.04, 0.0436, 0.03, 0.0289, 0.04286],
        "Position": [8.2, 8.0, 12.3, 11.7, 9.1]
    })

def download_df_button(df: pd.DataFrame, filename: str, label: str):
    """Create download button for dataframe"""
    if df is None or df.empty:
        st.warning(f"No data to download for {label}")
        return

    try:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=label,
            data=csv,
            file_name=filename,
            mime="text/csv",
            use_container_width=True
        )
    except Exception as e:
        st.error(f"Failed to create download: {e}")

def read_csv_safely(upload, name: str, vc: ValidationCollector, sample_rows: int = 1000) -> Optional[pd.DataFrame]:
    """Safely read CSV with multiple encoding attempts"""
    if upload is None:
        vc.add("Critical", "NO_FILE", f"{name} file not provided")
        return None

    try_encodings = [None, "utf-8", "utf-8-sig", "latin-1", "cp1252"]
    last_err = None

    for enc in try_encodings:
        try:
            upload.seek(0)
            df = pd.read_csv(upload, encoding=enc) if enc else pd.read_csv(upload)

            if df.empty or df.shape[1] == 0:
                vc.add("Critical", "EMPTY_CSV", f"{name} appears empty")
                return None

            # Check for problematic headers
            nullish_headers = sum(1 for c in df.columns if pd.isna(c) or str(c).strip().lower() in ("unnamed: 0", "", "nan"))
            if nullish_headers > 0:
                vc.add("Warning", "HEADER_SUSPECT", f"{name} had unnamed/blank headers", count=int(nullish_headers))

            vc.checkpoint(f"{name}_read", rows=int(min(len(df), sample_rows)), cols=int(df.shape[1]), encoding=enc or "auto")
            return add_lineage(df, name)

        except Exception as e:
            last_err = e
            continue

    vc.add("Critical", "CSV_ENCODING", f"Failed to read {name} with common encodings", last_error=str(last_err))
    vc.add_exc(f"read_csv:{name}", last_err or Exception("Unknown encoding error"))
    return None

# Sidebar configuration
with st.sidebar:
    st.subheader("Configuration")
    preset = st.selectbox("Presets", ["Standard", "Conservative", "Aggressive"], index=0)

    if st.button("Apply Preset"):
        if preset in st.session_state.config["presets"]:
            st.session_state.thresholds.update(st.session_state.config["presets"][preset])
            st.success(f"Applied {preset} preset")
        else:
            st.error(f"Preset {preset} not found in configuration")

    t = st.session_state.thresholds
    c1, c2 = st.columns(2)
    with c1:
        t["striking_distance_min"] = st.slider("Striking Distance — Min Position", 5, 50, t["striking_distance_min"])
        t["ctr_deficit_pct"] = st.slider("CTR Deficit Threshold (%)", 0.5, 10.0, float(t["ctr_deficit_pct"]), step=0.1)
        t["min_impressions"] = st.number_input("Min Impressions", min_value=0, value=int(t["min_impressions"]), step=50)
    with c2:
        t["striking_distance_max"] = st.slider("Striking Distance — Max Position", 5, 50, t["striking_distance_max"])
        t["similarity_threshold"] = st.slider("Content Similarity Threshold", 0.30, 0.90, float(t["similarity_threshold"]), step=0.05)
        t["min_clicks"] = st.number_input("Min Clicks", min_value=0, value=int(t["min_clicks"]), step=5)

    st.markdown("---")
    st.subheader("Analysis Period")
    start_def, end_def = st.session_state.date_range
    start_date = st.date_input("Start Date", value=start_def)
    end_date = st.date_input("End Date", value=end_def)

    if start_date > end_date:
        st.warning("Start date is after end date. Swapping.")
        start_date, end_date = end_date, start_date

    st.session_state.date_range = (start_date, end_date)

    st.markdown("---")
    st.subheader("Logging & Merge Options")
    st.session_state.log_level = st.selectbox("Log Level", ["DEBUG", "INFO", "WARNING", "ERROR"], index=1)
    get_logger(getattr(logging, st.session_state.log_level))

    strictness = st.selectbox("Validation Strictness", ["Strict", "Standard", "Lenient"], index=1)
    st.session_state.strictness = strictness

    ms1 = st.selectbox("GSC × Production Join", ["left", "inner"], index=0,
                      help="Left keeps all GSC rows; inner keeps only MSIDs present in Production")
    ms2 = st.selectbox("Attach GA4 on (msid,date)", ["left", "inner"], index=0)

    MERGE_STRATEGY["gsc_x_prod"] = ms1
    MERGE_STRATEGY["ga4_align"] = ms2

    st.markdown("---")
    st.subheader("About")
    st.caption("GrowthOracle AI v2.0 | SEO & Content Intelligence Platform")

# Main app steps
st.markdown("### Onboarding & Data Ingestion")
step = st.radio("Steps", [
    "1) Get CSV Templates",
    "2) Upload & Map Columns",
    "3) Validate & Process",
    "4) Configure & Analyze"
], horizontal=True)

if step == "1) Get CSV Templates":
    st.info("Download sample CSV templates to understand required structure.")
    colA, colB, colC = st.columns(3)

    with colA:
        df = _make_template_production()
        st.dataframe(df, use_container_width=True, hide_index=True)
        download_df_button(df, "template_production.csv", "Download Production Template")

    with colB:
        df = _make_template_ga4()
        st.dataframe(df, use_container_width=True, hide_index=True)
        download_df_button(df, "template_ga4.csv", "Download GA4 Template")

    with colC:
        df = _make_template_gsc()
        st.dataframe(df, use_container_width=True, hide_index=True)
        download_df_button(df, "template_gsc.csv", "Download GSC Template")

    st.stop()

# Step 2: Uploads
st.subheader("Upload Your Data Files")
with st.container():
    col1, col2, col3 = st.columns(3)
    with col1:
        prod_file = st.file_uploader("Production Data (CSV)", type=["csv"], key="prod_csv")
        if prod_file:
            st.success(f"✓ Production: {prod_file.name}")
    with col2:
        ga4_file = st.file_uploader("GA4 Data (CSV)", type=["csv"], key="ga4_csv")
        if ga4_file:
            st.success(f"✓ GA4: {ga4_file.name}")
    with col3:
        gsc_file = st.file_uploader("GSC Data (CSV)", type=["csv"], key="gsc_csv")
        if gsc_file:
            st.success(f"✓ GSC: {gsc_file.name}")

# Check if all files are uploaded
if not all([prod_file, ga4_file, gsc_file]):
    st.warning("Please upload all three CSV files to proceed")
    st.stop()

# Read raw files with strong reader for preview/mapping stage
vc_read = ValidationCollector()
prod_df_raw = read_csv_safely(prod_file, "Production", vc_read)
ga4_df_raw = read_csv_safely(ga4_file, "GA4", vc_read)
gsc_df_raw = read_csv_safely(gsc_file, "GSC", vc_read)

if any(df is None or df.empty for df in [prod_df_raw, ga4_df_raw, gsc_df_raw]):
    st.error("One or more uploaded files appear empty/unreadable. See Validation section below.")
    st.dataframe(vc_read.to_dataframe(), use_container_width=True, hide_index=True)
    st.stop()
    # ============================
# PART 3/5: Onboarding Steps, Mapping UI, Validation UI, Robust Processing
# ============================

def _guess_colmap(prod_df, ga4_df, gsc_df):
    """Guess column mappings based on common patterns"""
    if prod_df is None or ga4_df is None or gsc_df is None:
        return {}, {}, {}

    prod_map = {
        "msid": "Msid" if "Msid" in prod_df.columns else next((c for c in prod_df.columns if c.lower() == "msid"), None),
        "title": "Title" if "Title" in prod_df.columns else next((c for c in prod_df.columns if "title" in c.lower()), None),
        "path": "Path" if "Path" in prod_df.columns else next((c for c in prod_df.columns if "path" in c.lower()), None),
        "publish": "Publish Time" if "Publish Time" in prod_df.columns else next((c for c in prod_df.columns if "publish" in c.lower()), None),
    }

    ga4_map = {
        "msid": "customEvent:msid" if "customEvent:msid" in ga4_df.columns else next((c for c in ga4_df.columns if "msid" in c.lower()), None),
        "date": "date" if "date" in ga4_df.columns else next((c for c in ga4_df.columns if c.lower() == "date"), None),
        "pageviews": "screenPageViews" if "screenPageViews" in ga4_df.columns else next((c for c in ga4_df.columns if "pageview" in c.lower()), None),
        "users": "totalUsers" if "totalUsers" in ga4_df.columns else next((c for c in ga4_df.columns if "users" in c.lower()), None),
        "engagement": "userEngagementDuration" if "userEngagementDuration" in ga4_df.columns else next((c for c in ga4_df.columns if "engagement" in c.lower()), None),
        "bounce": "bounceRate" if "bounceRate" in ga4_df.columns else next((c for c in ga4_df.columns if "bounce" in c.lower()), None),
    }

    gsc_map = {
        "date": "Date" if "Date" in gsc_df.columns else next((c for c in gsc_df.columns if c.lower() == "date"), None),
        "page": "Page" if "Page" in gsc_df.columns else next((c for c in gsc_df.columns if "page" in c.lower()), None),
        "query": "Query" if "Query" in gsc_df.columns else next((c for c in gsc_df.columns if "query" in c.lower()), None),
        "clicks": "Clicks" if "Clicks" in gsc_df.columns else next((c for c in gsc_df.columns if "clicks" in c.lower()), None),
        "impr": "Impressions" if "Impressions" in gsc_df.columns else next((c for c in gsc_df.columns if "impr" in c.lower()), None),
        "ctr": "CTR" if "CTR" in gsc_df.columns else next((c for c in gsc_df.columns if "ctr" in c.lower()), None),
        "pos": "Position" if "Position" in gsc_df.columns else next((c for c in gsc_df.columns if "position" in c.lower()), None),
    }

    return prod_map, ga4_map, gsc_map

def guess_colmap_enhanced(prod_df, ga4_df, gsc_df):
    """Enhanced column mapping with date detection"""
    prod_map, ga4_map, gsc_map = _guess_colmap(prod_df, ga4_df, gsc_df)

    # Enhanced date detection
    if prod_df is not None:
        prod_dates = [c for c in detect_date_cols(prod_df) if "publish" in c.lower() or "time" in c.lower()]
        if prod_dates and not prod_map.get("publish"):
            prod_map["publish"] = prod_dates[0]

    if ga4_df is not None and not ga4_map.get("date"):
        for c in detect_date_cols(ga4_df):
            if c.lower() == "date":
                ga4_map["date"] = c
                break

    if gsc_df is not None and not gsc_map.get("date"):
        for c in detect_date_cols(gsc_df):
            if c.lower() == "date":
                gsc_map["date"] = c
                break

    return prod_map, ga4_map, gsc_map

def validate_columns_presence(prod_map, ga4_map, gsc_map, vc: ValidationCollector):
    """Validate that required columns are mapped"""
    req_prod = ["msid"]
    req_ga4 = ["msid"]
    req_gsc = ["date", "page", "query", "clicks", "impr", "pos"] # Clicks is now required

    missing = []
    for k in req_prod:
        if not prod_map.get(k):
            missing.append(f"Production: {k}")
    for k in req_ga4:
        if not ga4_map.get(k):
            missing.append(f"GA4: {k}")
    for k in req_gsc:
        if not gsc_map.get(k):
            missing.append(f"GSC: {k}")

    if missing:
        vc.add("Critical", "MISSING_COLMAP", "Missing/ambiguous mappings for required columns", items=missing)
    return missing

if step == "2) Upload & Map Columns":
    prod_map_guess, ga4_map_guess, gsc_map_guess = guess_colmap_enhanced(prod_df_raw, ga4_df_raw, gsc_df_raw)
    st.subheader("Column Mapping")
    st.caption("We guessed likely columns. Adjust if needed.")

    with st.expander("Production Mapping", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        prod_map = {}
        prod_map["msid"] = c1.selectbox(
            "MSID", prod_df_raw.columns,
            index=prod_df_raw.columns.get_loc(prod_map_guess["msid"]) if prod_map_guess.get("msid") in prod_df_raw.columns else 0
        )
        prod_map["title"] = c2.selectbox(
            "Title", prod_df_raw.columns,
            index=prod_df_raw.columns.get_loc(prod_map_guess["title"]) if prod_map_guess.get("title") in prod_df_raw.columns else 0
        )
        prod_map["path"] = c3.selectbox(
            "Path", prod_df_raw.columns,
            index=prod_df_raw.columns.get_loc(prod_map_guess["path"]) if prod_map_guess.get("path") in prod_df_raw.columns else 0
        )
        prod_map["publish"] = c4.selectbox(
            "Publish Time", prod_df_raw.columns,
            index=prod_df_raw.columns.get_loc(prod_map_guess["publish"]) if prod_map_guess.get("publish") in prod_df_raw.columns else 0
        )

    with st.expander("GA4 Mapping", expanded=True):
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        ga4_map = {}
        ga4_map["msid"] = c1.selectbox(
            "MSID (GA4)", ga4_df_raw.columns,
            index=ga4_df_raw.columns.get_loc(ga4_map_guess["msid"]) if ga4_map_guess.get("msid") in ga4_df_raw.columns else 0
        )
        ga4_map["date"] = c2.selectbox(
            "Date (GA4)", ga4_df_raw.columns,
            index=ga4_df_raw.columns.get_loc(ga4_map_guess["date"]) if ga4_map_guess.get("date") in ga4_df_raw.columns else 0
        )
        ga4_map["pageviews"] = c3.selectbox(
            "Pageviews", ga4_df_raw.columns,
            index=ga4_df_raw.columns.get_loc(ga4_map_guess["pageviews"]) if ga4_map_guess.get("pageviews") in ga4_df_raw.columns else 0
        )
        ga4_map["users"] = c4.selectbox(
            "Users", ga4_df_raw.columns,
            index=ga4_df_raw.columns.get_loc(ga4_map_guess["users"]) if ga4_map_guess.get("users") in ga4_df_raw.columns else 0
        )
        ga4_map["engagement"] = c5.selectbox(
            "Engagement Duration", ga4_df_raw.columns,
            index=ga4_df_raw.columns.get_loc(ga4_map_guess["engagement"]) if ga4_map_guess.get("engagement") in ga4_df_raw.columns else 0
        )
        ga4_map["bounce"] = c6.selectbox(
            "Bounce Rate", ga4_df_raw.columns,
            index=ga4_df_raw.columns.get_loc(ga4_map_guess["bounce"]) if ga4_map_guess.get("bounce") in ga4_df_raw.columns else 0
        )

    with st.expander("GSC Mapping", expanded=True):
        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        gsc_map = {}
        gsc_map["date"] = c1.selectbox(
            "Date (GSC)", gsc_df_raw.columns,
            index=gsc_df_raw.columns.get_loc(gsc_map_guess["date"]) if gsc_map_guess.get("date") in gsc_df_raw.columns else 0
        )
        gsc_map["page"] = c2.selectbox(
            "Page URL", gsc_df_raw.columns,
            index=gsc_df_raw.columns.get_loc(gsc_map_guess["page"]) if gsc_map_guess.get("page") in gsc_df_raw.columns else 0
        )
        gsc_map["query"] = c3.selectbox(
            "Query", gsc_df_raw.columns,
            index=gsc_df_raw.columns.get_loc(gsc_map_guess["query"]) if gsc_map_guess.get("query") in gsc_df_raw.columns else 0
        )
        gsc_map["clicks"] = c4.selectbox( # FIXED: Added Clicks mapping
            "Clicks", gsc_df_raw.columns,
            index=gsc_df_raw.columns.get_loc(gsc_map_guess["clicks"]) if gsc_map_guess.get("clicks") in gsc_df_raw.columns else 0
        )
        gsc_map["impr"] = c5.selectbox(
            "Impressions", gsc_df_raw.columns,
            index=gsc_df_raw.columns.get_loc(gsc_map_guess["impr"]) if gsc_map_guess.get("impr") in gsc_df_raw.columns else 0
        )
        gsc_map["ctr"] = c6.selectbox(
            "CTR", gsc_df_raw.columns,
            index=gsc_df_raw.columns.get_loc(gsc_map_guess["ctr"]) if gsc_map_guess.get("ctr") in gsc_df_raw.columns else 0
        )
        gsc_map["pos"] = c7.selectbox(
            "Position", gsc_df_raw.columns,
            index=gsc_df_raw.columns.get_loc(gsc_map_guess["pos"]) if gsc_map_guess.get("pos") in gsc_df_raw.columns else 0
        )

    # Validate mappings
    missing = validate_columns_presence(prod_map, ga4_map, gsc_map, vc_read)
    rep_df = vc_read.to_dataframe()

    if not rep_df.empty:
        st.markdown("**Preliminary Reader/Mapping Warnings**")
        st.dataframe(rep_df, use_container_width=True, hide_index=True)

    if missing:
        st.error("Critical mapping issues detected. Please fix before proceeding.")
    else:
        st.session_state.mapping = {"prod": prod_map, "ga4": ga4_map, "gsc": gsc_map}
        st.success("Column mapping saved. Proceed to **Step 3**.")

    st.stop()

# Step 3: Validate & Process
if "mapping" not in st.session_state:
    st.warning("Please complete **Step 2** (column mapping) first.")
    st.stop()

prod_map = st.session_state.mapping["prod"]
ga4_map = st.session_state.mapping["ga4"]
gsc_map = st.session_state.mapping["gsc"]

def run_validation_pipeline(prod_df_raw, ga4_df_raw, gsc_df_raw, prod_map, ga4_map, gsc_map):
    """Run comprehensive validation pipeline"""
    vc = ValidationCollector()

    # Validate column mappings
    missing_map = validate_columns_presence(prod_map, ga4_map, gsc_map, vc)

    # Validate each dataframe
    for name, df in [("Production", prod_df_raw), ("GA4", ga4_df_raw), ("GSC", gsc_df_raw)]:
        if df is None or df.empty:
            vc.add("Critical", "EMPTY_FILE", f"{name} is empty or unreadable")
            continue

        if df.shape[1] < 2:
            vc.add("Critical", "TOO_FEW_COLS", f"{name} has too few cols", cols=int(df.shape[1]))

        if df.duplicated().any():
            vc.add("Info", "DUP_ROWS", f"{name} contained fully duplicated rows", rows=int(df.duplicated().sum()))

        cand = detect_date_cols(df)
        if cand:
            vc.add("Info", "DATE_CANDIDATES", f"Possible date columns in {name}", columns=cand[:6])

    # Check MSID consistency between Production and GSC
    try:
        if prod_df_raw is not None and gsc_df_raw is not None:
            p_m = set(pd.to_numeric(prod_df_raw[prod_map["msid"]], errors="coerce").dropna().astype("int64"))

            def msid_from_url(u):
                if isinstance(u, str):
                    m = re.search(r"(\d+)\.cms", u)
                    return int(m.group(1)) if m else None
                return None

            g_m = set(pd.to_numeric(gsc_df_raw[gsc_map["page"]].apply(msid_from_url), errors="coerce").dropna().astype("int64"))
            only_p, only_g = len(p_m - g_m), len(g_m - p_m)

            if only_p:
                vc.add("Info", "MSID_ONLY_PROD", "MSIDs appear only in Production", count=int(only_p))
            if only_g:
                vc.add("Warning", "MSID_ONLY_GSC", "MSIDs appear only in GSC", count=int(only_g))
    except Exception as e:
        vc.add_exc("preview_msid_consistency", e)

    return vc

# ============================
# PART 3 Cont'd: Validation UI and Robust Processing
# ============================

st.subheader("Data Validation Report")
vc0 = run_validation_pipeline(prod_df_raw, ga4_df_raw, gsc_df_raw, prod_map, ga4_map, gsc_map)
rep_df = vc0.to_dataframe()

if rep_df.empty:
    st.success("No issues detected in preliminary checks ✅")
else:
    tabs = st.tabs(["Critical", "Warning", "Info"])
    for i, cat in enumerate(["Critical", "Warning", "Info"]):
        with tabs[i]:
            sub = rep_df[rep_df["category"] == cat]
            if not sub.empty:
                st.dataframe(sub, use_container_width=True, hide_index=True)
            else:
                st.info(f"No {cat} issues")

    st.caption(f"Data Quality Score (pre-processing): **{vc0.quality_score():.0f} / 100**")
    st.download_button(
        "Download Validation Report (CSV)",
        data=rep_df.to_csv(index=False).encode("utf-8"),
        file_name=f"validation_report_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

def standardize_dates_early(prod_df, ga4_df, gsc_df, mappings, vc: ValidationCollector):
    """Standardize date formats early in processing pipeline"""
    def _ensure_utc(ts: pd.Timestamp) -> pd.Timestamp:
        if pd.isna(ts): return ts
        try: return ts.tz_convert("UTC")
        except:
            try: return ts.tz_localize("UTC")
            except: return ts

    def normalize_date_only(df, col_name, out_name):
        if df is not None and col_name in df.columns:
            dt = safe_dt_parse(df[col_name], col_name, vc)
            df[out_name] = dt.dt.date
            if dt.notna().any():
                maxd, mind = dt.max(), dt.min()
                if pd.notna(maxd) and _ensure_utc(maxd) > pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1):
                    vc.add("Warning", "FUTURE_DATE", f"{out_name} has future dates", sample=str(maxd))
                if pd.notna(mind) and _ensure_utc(mind) < pd.Timestamp(2020, 1, 1, tz="UTC"):
                    vc.add("Info", "OLD_DATE", f"{out_name} includes <2020 dates", earliest=str(mind))

    p = prod_df.copy() if prod_df is not None else None
    if p is not None and mappings["prod"].get("publish"):
        p["Publish Time"] = safe_dt_parse(p[mappings["prod"]["publish"]], "Publish Time", vc)

    g4 = ga4_df.copy() if ga4_df is not None else None
    if g4 is not None and mappings["ga4"].get("date"):
        normalize_date_only(g4, mappings["ga4"]["date"], "date")

    gs = gsc_df.copy() if gsc_df is not None else None
    if gs is not None and mappings["gsc"].get("date"):
        normalize_date_only(gs, mappings["gsc"]["date"], "date")

    return p, g4, gs

@st.cache_data(show_spinner=False, max_entries=3)
def process_uploaded_files_complete(prod_df_raw, ga4_df_raw, gsc_df_raw, prod_map, ga4_map, gsc_map,
                                   vc_serialized: Optional[str] = None,
                                   merge_strategy: Optional[Dict[str, str]] = None):
    """COMPLETE processing pipeline that processes ALL data"""
    vc = ValidationCollector()
    if vc_serialized:
        try:
            messages = json.loads(vc_serialized)
            for item in messages:
                ctx = item.get("context", {})
                if isinstance(ctx, str):
                    try: ctx = json.loads(ctx)
                    except: ctx = {}
                vc.add(item["category"], item["code"], item["message"], **ctx)
        except Exception as e:
            vc.add("Warning", "VC_LOAD_FAIL", f"Failed to load previous validation: {e}")

    ms = merge_strategy or MERGE_STRATEGY
    prod_df = prod_df_raw.copy() if prod_df_raw is not None else None
    ga4_df = ga4_df_raw.copy() if ga4_df_raw is not None else None
    gsc_df = gsc_df_raw.copy() if gsc_df_raw is not None else None

    # --- 1. RENAME COLUMNS to standard names ---
    rename_maps = {
        "prod": {v: k for k, v in prod_map.items()},
        "ga4": {v: k for k, v in ga4_map.items()},
        "gsc": {v: k for k, v in gsc_map.items()}
    }
    # Standard names
    std_names = {
        "prod": {"msid": "msid", "title": "Title", "path": "Path", "publish": "Publish Time"},
        "ga4": {"msid": "msid", "date": "date", "pageviews": "screenPageViews", "users": "totalUsers", "engagement": "userEngagementDuration", "bounce": "bounceRate"},
        "gsc": {"date": "date", "page": "page_url", "query": "Query", "clicks": "Clicks", "impr": "Impressions", "ctr": "CTR", "pos": "Position"}
    }
    try:
        if prod_df is not None: prod_df.rename(columns={prod_map[k]: v for k, v in std_names["prod"].items() if prod_map.get(k)}, inplace=True)
        if ga4_df is not None: ga4_df.rename(columns={ga4_map[k]: v for k, v in std_names["ga4"].items() if ga4_map.get(k)}, inplace=True)
        if gsc_df is not None: gsc_df.rename(columns={gsc_map[k]: v for k, v in std_names["gsc"].items() if gsc_map.get(k)}, inplace=True)
    except Exception as e:
        vc.add("Critical", "RENAME_FAIL", f"Failed during column renaming: {e}")
        return None, vc

    # --- 2. EARLY DATE & MSID STANDARDIZATION ---
    prod_df, ga4_df, gsc_df = standardize_dates_early(prod_df, ga4_df, gsc_df, {"prod": std_names["prod"], "ga4": std_names["ga4"], "gsc": std_names["gsc"]}, vc)

    for df, name in [(prod_df, "Production"), (ga4_df, "GA4")]:
        if df is not None and "msid" in df.columns:
            df["msid"] = pd.to_numeric(df["msid"], errors="coerce")
            if df["msid"].isna().any(): vc.add("Warning", "MSID_BAD", f"Non-numeric MSIDs in {name}", count=int(df['msid'].isna().sum()))
            df.dropna(subset=["msid"], inplace=True)
            if not df.empty: df["msid"] = df["msid"].astype("int64")

    if gsc_df is not None and "page_url" in gsc_df.columns:
        gsc_df["msid"] = gsc_df["page_url"].str.extract(r'(\d+)\.cms').iloc[:, 0]
        gsc_df["msid"] = pd.to_numeric(gsc_df["msid"], errors="coerce")
        if gsc_df["msid"].isna().any(): vc.add("Warning", "MSID_EXTRACT_FAIL", "Could not extract MSID from some GSC URLs", count=int(gsc_df['msid'].isna().sum()))
        gsc_df.dropna(subset=["msid"], inplace=True)
        if not gsc_df.empty: gsc_df["msid"] = gsc_df["msid"].astype("int64")

    # --- 3. ROBUST NUMERIC CONVERSION (CRITICAL FIX) ---
    if gsc_df is not None:
        for col, clamp in [("Clicks", (0, None)), ("Impressions", (0, None)), ("Position", (1, 100))]:
            if col in gsc_df.columns:
                gsc_df[col] = coerce_numeric(gsc_df[col], f"GSC.{col}", vc, clamp=clamp)
        if "CTR" in gsc_df.columns:
            # Handle percentage CTRs
            if gsc_df["CTR"].dtype == 'object' and gsc_df["CTR"].str.contains('%').any():
                gsc_df["CTR"] = pd.to_numeric(gsc_df["CTR"].str.replace('%', ''), errors='coerce') / 100.0
            gsc_df["CTR"] = coerce_numeric(gsc_df["CTR"], "GSC.CTR", vc, clamp=(0, 1))
        elif "Clicks" in gsc_df.columns and "Impressions" in gsc_df.columns:
            # Calculate CTR if not provided
            vc.add("Info", "CTR_CALCULATED", "CTR column calculated from Clicks/Impressions")
            gsc_df["CTR"] = (gsc_df["Clicks"] / gsc_df["Impressions"].replace(0, np.nan)).fillna(0)

    if ga4_df is not None:
        for col in ["screenPageViews", "totalUsers", "userEngagementDuration", "bounceRate"]:
            if col in ga4_df.columns:
                ga4_df[col] = coerce_numeric(ga4_df[col], f"GA4.{col}", vc)

    # --- 4. MERGE DATA ---
    if gsc_df is None or prod_df is None or gsc_df.empty or prod_df.empty:
        vc.add("Critical", "MERGE_PREP_FAIL", "Cannot merge due to missing GSC or Production data.")
        return None, vc

    prod_cols = [c for c in ["msid", "Title", "Path", "Publish Time"] if c in prod_df.columns]
    merged_1 = pd.merge(gsc_df, prod_df[prod_cols].drop_duplicates(subset=["msid"]), on="msid", how=ms.get("gsc_x_prod", "left"))
    vc.checkpoint("merge_gsc_prod", after_m1=len(merged_1))

    if ga4_df is not None and not ga4_df.empty and "date" in ga4_df.columns:
        numeric_cols = [c for c in ["screenPageViews", "totalUsers", "userEngagementDuration", "bounceRate"] if c in ga4_df.columns]
        ga4_daily = ga4_df.groupby(["msid", "date"], as_index=False)[numeric_cols].sum(min_count=1)
        master_df = pd.merge(merged_1, ga4_daily, on=["msid", "date"], how=ms.get("ga4_align", "left"))
    else:
        master_df = merged_1
        vc.add("Info", "NO_GA4_MERGE", "GA4 data not available or missing 'date' for merge")
    vc.checkpoint("merge_ga4", after_master=len(master_df))

    # --- 5. FINAL CLEANING AND FEATURE ENGINEERING ---
    if master_df is not None and not master_df.empty:
        if "Path" in master_df.columns:
            cats = master_df["Path"].str.strip('/').str.split('/', n=2, expand=True)
            master_df["L1_Category"] = cats[0].fillna("Uncategorized")
            master_df["L2_Category"] = cats[1].fillna("General")
        else:
            master_df["L1_Category"] = "Uncategorized"
            master_df["L2_Category"] = "General"

        if "Title" in master_df.columns:
            drop_n = master_df["Title"].isna().sum()
            if drop_n > 0:
                vc.add("Warning", "TITLE_MISSING", "Rows lacking Title dropped", rows=int(drop_n))
                master_df.dropna(subset=["Title"], inplace=True)

        master_df["_lineage"] = "GSC→PROD→GA4"

    return master_df, vc


# --- RUN PROCESSING ---
vc_serialized = rep_df.to_json(orient="records") if not rep_df.empty else "[]"
with st.spinner("Processing & merging datasets... This may take a moment."):
    master_df, vc_after = process_uploaded_files_complete(
        prod_df_raw, ga4_df_raw, gsc_df_raw,
        prod_map, ga4_map, gsc_map,
        vc_serialized=vc_serialized,
        merge_strategy=MERGE_STRATEGY
    )

if master_df is None or master_df.empty:
    st.error("Data processing failed critically. Please check the validation report and your file mappings.")
    if vc_after:
        st.dataframe(vc_after.to_dataframe(), use_container_width=True, hide_index=True)
    st.stop()

# Post-merge validation report
st.subheader("Post-merge Data Quality")
post_df = vc_after.to_dataframe()
if not post_df.empty:
    st.dataframe(post_df, use_container_width=True, hide_index=True)
    st.caption(f"Data Quality Score (post-merge): **{vc_after.quality_score():.0f} / 100**")
    st.download_button(
        "Download Post-merge Report (CSV)",
        data=post_df.to_csv(index=False).encode("utf-8"),
        file_name=f"postmerge_report_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

st.success(f"✅ Master dataset created: {master_df.shape[0]:,} rows × {master_df.shape[1]} columns")

if "date" in master_df.columns:
    try:
        date_col = pd.to_datetime(master_df["date"], errors="coerce")
        if date_col.notna().any():
            min_date, max_date = date_col.min().date(), date_col.max().date()
            st.caption(f"Date range in master data: **{min_date}** to **{max_date}**")
    except Exception: pass

if master_df.shape[0] > CONFIG["performance"]["sample_row_limit"]:
    st.info(f"Large dataset detected. For interactive analysis, a sample of {CONFIG['performance']['sample_row_limit']:,} rows will be used.")
    analysis_df = master_df.sample(
        n=CONFIG["performance"]["sample_row_limit"],
        random_state=CONFIG["performance"]["seed"]
    )
else:
    analysis_df = master_df

st.subheader("Data Preview (First 10 rows)")
st.dataframe(master_df.head(10), use_container_width=True, hide_index=True)

with st.expander("Data Summary", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", f"{len(master_df):,}")
    with col2:
        st.metric("Total Columns", f"{len(master_df.columns)}")
    with col3:
        if "msid" in master_df.columns:
            st.metric("Unique Articles", f"{master_df['msid'].nunique():,}")

if step == "3) Validate & Process":
    st.success("Data processing complete! Move to **Step 4) Configure & Analyze** to generate insights.")
    st.stop()
    # ============================
# PART 4/5: Core Analysis Modules
# ============================

# Date filtering
def filter_by_date(df, start_date, end_date):
    if df is None or df.empty or "date" not in df.columns:
        return df

    df_copy = df.copy()
    try:
        df_copy["date"] = pd.to_datetime(df_copy["date"], errors="coerce").dt.date
        mask = (df_copy["date"] >= start_date) & (df_copy["date"] <= end_date)
        filtered = df_copy[mask].copy()
        st.info(f"Date filter applied: {len(filtered):,} rows from {start_date} to {end_date}")
        return filtered
    except Exception as e:
        st.warning(f"Date filtering failed: {e}")
        return df_copy

# Apply date filter
start_date, end_date = st.session_state.date_range
filtered_df = filter_by_date(analysis_df, start_date, end_date)

TH = st.session_state.thresholds
EXPECTED_CTR = CONFIG["expected_ctr_by_rank"]

def run_module_safely(label: str, fn, *args, **kwargs):
    """Execute analysis module with comprehensive error handling"""
    try:
        with st.spinner(f"Running {label}..."):
            result = fn(*args, **kwargs)
        return result
    except Exception as e:
        error_msg = f"Module '{label}' encountered an issue. Details: {type(e).__name__}: {e}"
        st.warning(error_msg)
        logger.error(f"[ModuleFail] {label}: {e}", exc_info=True)
        return None

def export_plot_html(fig, name):
    """Export Plotly figure to HTML"""
    if to_html is None or fig is None:
        st.info("Plotly HTML export not available.")
        return
    try:
        html_str = to_html(fig, include_plotlyjs="cdn", full_html=True)
        st.download_button(
            f"Export {name} (HTML)",
            data=html_str.encode("utf-8"),
            file_name=f"{name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.html",
            mime="text/html"
        )
    except Exception as e:
        st.warning(f"Failed to export plot: {e}")

def _pick_col(df, candidates):
    """Safely pick first available column from candidates"""
    if df is None: return None
    for candidate in candidates:
        if candidate in df.columns: return candidate
    return None

def engagement_mismatches(df):
    """Identify engagement vs search performance mismatches"""
    if df is None or df.empty:
        return ["No data available for analysis"]

    d = df.copy()
    insights = []

    if "Position" in d.columns and "CTR" in d.columns:
        pos_ctr_data = d[["msid", "Position", "CTR", "Title"]].dropna()
        if not pos_ctr_data.empty:
            good_pos_low_ctr = pos_ctr_data[(pos_ctr_data["Position"] <= 10) & (pos_ctr_data["CTR"] < 0.03)]
            for _, row in good_pos_low_ctr.head(2).iterrows():
                insights.append(f"""### ⚠️ Low CTR at Good Position
**MSID:** `{row.get('msid', 'N/A')}` | **Position:** {row['Position']:.1f} | **CTR:** {row['CTR']:.2%}
**Title:** {str(row.get('Title', 'Unknown'))[:80]}...
**Recommendation:** Test more compelling titles and meta descriptions to convert high visibility into more clicks.""")

            high_ctr_poor_position = pos_ctr_data[(pos_ctr_data["Position"] > 15) & (pos_ctr_data["CTR"] > 0.05)]
            for _, row in high_ctr_poor_position.head(2).iterrows():
                insights.append(f"""### 💎 High CTR at Poor Position (Hidden Gem)
**MSID:** `{row.get('msid', 'N/A')}` | **Position:** {row['Position']:.1f} | **CTR:** {row['CTR']:.2%}
**Title:** {str(row.get('Title', 'Unknown'))[:80]}...
**Recommendation:** This content resonates well. Invest in on-page SEO and link building to improve its ranking.""")

    if "bounceRate" in d.columns and "Position" in d.columns:
        bounce_data = d[["msid", "bounceRate", "Title", "Position"]].dropna()
        if not bounce_data.empty:
            high_bounce_good_pos = bounce_data[(bounce_data["bounceRate"] > 0.7) & (bounce_data["Position"] <= 15)]
            for _, row in high_bounce_good_pos.head(2).iterrows():
                insights.append(f"""### 🚨 High Bounce Rate at Good Position
**MSID:** `{row.get('msid', 'N/A')}` | **Position:** {row['Position']:.1f} | **Bounce Rate:** {row['bounceRate']:.1%}
**Title:** {str(row.get('Title', 'Unknown'))[:80]}...
**Recommendation:** Content may not match search intent. Review content, improve page speed, and enhance readability.""")

    if not insights:
        insights.append("No specific engagement-search mismatches detected. Content appears well-balanced.")
    return insights

def scatter_engagement_vs_search(df):
    """Create engagement vs search scatter plot"""
    if df is None or df.empty:
        st.info("No data available for scatter plot")
        return

    engagement_col = _pick_col(df, ["userEngagementDuration", "totalUsers"])
    search_col = _pick_col(df, ["CTR", "Position"])
    size_col = _pick_col(df, ["Clicks", "screenPageViews", "Impressions"])

    if not engagement_col or not search_col or not size_col:
        st.info("Insufficient metrics for a meaningful scatter plot (need engagement, search, and size metrics).")
        return

    plot_data = df[["msid", "Title", "L1_Category", engagement_col, search_col, size_col]].dropna()
    if plot_data.empty:
        st.info("No complete data available for scatter plot after cleaning.")
        return

    if _HAS_PLOTLY:
        try:
            fig = px.scatter(
                plot_data,
                x=search_col,
                y=engagement_col,
                size=size_col,
                color="L1_Category",
                hover_data=["msid", "Title"],
                title=f"Engagement ({engagement_col}) vs. Search Performance ({search_col})",
                labels={k: k.replace("_", " ").title() for k in [engagement_col, search_col, size_col]},
                size_max=60
            )
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")
            export_plot_html(fig, "engagement_vs_search")
        except Exception as e:
            st.error(f"Failed to create scatter plot: {e}")
    else:
        st.scatter_chart(plot_data, x=search_col, y=engagement_col, size=size_col, color="L1_Category")

def category_heatmap(df, value_col, title):
    """Create category performance heatmap"""
    if not _HAS_PLOTLY: st.info("Heatmap requires Plotly."); return
    if df is None or df.empty or value_col not in df.columns:
        st.info(f"No data for heatmap (value column: {value_col})."); return

    try:
        agg_data = df.groupby(["L1_Category", "L2_Category"]).agg(
            value=(value_col, "sum")
        ).reset_index()
        if agg_data.empty: st.info("No data after aggregation for heatmap."); return

        fig = px.density_heatmap(
            agg_data, x="L1_Category", y="L2_Category", z="value",
            color_continuous_scale="Viridis", title=title,
            labels={"value": value_col}
        )
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")
        export_plot_html(fig, f"heatmap_{value_col}")
    except Exception as e: st.error(f"Failed to create heatmap: {e}")

def analyze_category_performance(df):
    """Analyze category-level performance"""
    if df is None or df.empty: return pd.DataFrame()

    d = df.copy()
    agg_dict = {"msid": pd.NamedAgg(column="msid", aggfunc="nunique")}
    rename_map = {"msid": "total_articles"}

    metric_map = {
        "Clicks": "total_gsc_clicks",
        "screenPageViews": "total_pageviews",
        "userEngagementDuration": "avg_engagement_s",
        "Impressions": "total_impressions"
    }
    agg_funcs = {"sum": ["Clicks", "screenPageViews", "Impressions"], "mean": ["userEngagementDuration"]}

    for agg_func, cols in agg_funcs.items():
        for col in cols:
            if col in d.columns:
                agg_dict[col] = pd.NamedAgg(column=col, aggfunc=agg_func)
                rename_map[col] = metric_map[col]

    if len(agg_dict) <= 1:
        st.warning("No performance metrics found for category analysis."); return pd.DataFrame()

    try:
        grouped = d.groupby(["L1_Category", "L2_Category"]).agg(**agg_dict).reset_index().rename(columns=rename_map)
        return grouped.sort_values(by=list(rename_map.values())[1], ascending=False) # Sort by first metric
    except Exception as e:
        st.error(f"Category analysis failed: {e}"); return pd.DataFrame()


def forecast_series(daily_series, periods=14):
    """Generate time series forecasts"""
    if daily_series is None or len(daily_series) < 7: return None
    try:
        daily_series = daily_series.asfreq("D").fillna(method="ffill").fillna(0)
        if _HAS_STM and len(daily_series) >= 14:
            model = ExponentialSmoothing(daily_series, trend="add", seasonal="add", seasonal_periods=7).fit()
            forecast = model.forecast(periods)
            # Simple confidence interval
            std_err = np.std(model.resid) * np.sqrt(np.arange(1, periods + 1))
            return pd.DataFrame({"date": forecast.index, "forecast": forecast, "low": forecast - 1.96 * std_err, "high": forecast + 1.96 * std_err})
        # Fallback
        last_val = daily_series.rolling(7).mean().iloc[-1]
        future_dates = pd.date_range(start=daily_series.index.max() + pd.Timedelta(days=1), periods=periods)
        return pd.DataFrame({"date": future_dates, "forecast": [last_val] * periods, "low": last_val * 0.8, "high": last_val * 1.2})
    except Exception as e: st.error(f"Forecasting failed: {e}"); return None

def time_series_trends(df, metric_col, title):
    """Create time series trend visualization"""
    if df is None or df.empty or "date" not in df.columns or metric_col not in df.columns:
        st.info(f"Time series analysis requires date and '{metric_col}'."); return

    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date", metric_col])
    daily_data = d.groupby("date")[metric_col].sum()

    if daily_data.empty: st.info(f"No data for {metric_col} time series."); return

    if _HAS_PLOTLY:
        try:
            fig = px.line(daily_data, title=title, labels={"value": metric_col, "date": "Date"})
            fig.update_layout(xaxis_rangeslider_visible=True)
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")
            export_plot_html(fig, f"timeseries_{metric_col}")
        except Exception as e: st.error(f"Failed to create time series chart: {e}")
    else: st.line_chart(daily_data)


# ============================
# MAIN ANALYSIS UI
# ============================
st.header("📊 Advanced Analytics & Insights")

# Module 4: Engagement vs Search Mismatch
st.subheader("Engagement vs. Search Performance Mismatch")
st.caption("Identify content with high engagement but poor search performance (Hidden Gems) and vice-versa.")

engagement_cards = run_module_safely("Engagement Mismatch Analysis", engagement_mismatches, filtered_df)
if isinstance(engagement_cards, list):
    for card in engagement_cards:
        st.markdown(card)

st.subheader("Engagement vs. Search Scatter Analysis")
run_module_safely("Engagement Scatter Plot", scatter_engagement_vs_search, filtered_df)

st.divider()
# ============================
# PART 5/5: Complete Analysis & Export
# ============================

# Module 5: Category Performance
st.subheader("Category Performance Analysis")
st.caption("Analyze the performance of your content categories across key metrics.")

category_results = run_module_safely("Category Performance", analyze_category_performance, filtered_df)

if isinstance(category_results, pd.DataFrame) and not category_results.empty:
    st.dataframe(category_results, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Category Traffic Distribution")
        traffic_col = _pick_col(filtered_df, ["Clicks", "screenPageViews", "Impressions"])
        if traffic_col:
            run_module_safely("Traffic Heatmap", category_heatmap, filtered_df, traffic_col, f"Category Heatmap by {traffic_col}")
        else: st.info("No traffic metrics (Clicks, Pageviews) for heatmap.")

    with col2:
        st.subheader("Top Categories by Performance")
        perf_col = _pick_col(category_results, ["total_gsc_clicks", "total_pageviews"])
        if perf_col and _HAS_PLOTLY:
            top_cats = category_results.nlargest(10, perf_col)
            fig = px.bar(top_cats, x="L2_Category", y=perf_col, title=f"Top 10 Categories by {perf_col}", color="L1_Category")
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")
        elif perf_col:
             st.bar_chart(category_results.nlargest(10, perf_col).set_index("L2_Category")[perf_col])
        else: st.info("No performance metrics to rank categories.")
else:
    st.info("Category analysis could not be completed. Check data and mappings.")

st.divider()

# Module 6: Trends & Forecasting
st.subheader("Trends & Forecasting")
st.caption("Analyze historical trends and generate performance forecasts.")

if "date" in filtered_df.columns and not filtered_df.empty:
    ts_data = filtered_df.copy()
    primary_metric = _pick_col(ts_data, ["Clicks", "screenPageViews", "totalUsers", "Impressions"])

    if primary_metric:
        st.subheader(f"Overall Trend: {primary_metric}")
        run_module_safely(f"{primary_metric} Time Series", time_series_trends, ts_data, primary_metric, f"{primary_metric} Over Time")

        daily_series = ts_data.groupby(pd.to_datetime(ts_data['date']))[primary_metric].sum()
        if len(daily_series) >= 7:
            st.subheader(f"14-Day Forecast for {primary_metric}")
            forecast_data = forecast_series(daily_series, periods=14)
            if forecast_data is not None and _HAS_PLOTLY:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=daily_series.index, y=daily_series.values, name="Historical Data"))
                fig.add_trace(go.Scatter(x=forecast_data["date"], y=forecast_data["forecast"], name="Forecast", line=dict(dash="dash")))
                fig.add_trace(go.Scatter(x=forecast_data["date"], y=forecast_data["high"], fill=None, mode='lines', line_color='rgba(255,0,0,0.2)', name="Upper Bound"))
                fig.add_trace(go.Scatter(x=forecast_data["date"], y=forecast_data["low"], fill='tonexty', mode='lines', line_color='rgba(255,0,0,0.2)', name="Lower Bound"))
                st.plotly_chart(fig, use_container_width=True, theme="streamlit")
                export_plot_html(fig, f"forecast_{primary_metric}")
    else: st.info("No suitable metrics (Clicks, Pageviews, etc.) found for trend analysis.")
else: st.info("Date column required for trend analysis and forecasting.")


# ============================
# EXPORT & SUMMARY
# ============================
st.divider()
st.subheader("📤 Export & Summary")

col1, col2, col3 = st.columns(3)
with col1:
    download_df_button(filtered_df, f"growthoracle_analysis_{pd.Timestamp.now().strftime('%Y%m%d')}.csv", "Download Analysis Data")
with col2:
    if isinstance(category_results, pd.DataFrame):
        download_df_button(category_results, f"category_performance_{pd.Timestamp.now().strftime('%Y%m%d')}.csv", "Download Category Analysis")

with col3:
    with st.expander("Data Summary", expanded=True):
        st.metric("Total Rows Analyzed", f"{len(filtered_df):,}")
        if "msid" in filtered_df.columns:
            st.metric("Unique Articles", f"{filtered_df['msid'].nunique():,}")
        if "Clicks" in filtered_df.columns:
            total_clicks = filtered_df["Clicks"].sum()
            # FIXED: This will now work as 'Clicks' is numeric
            st.metric("Total GSC Clicks", f"{int(total_clicks):,}")


st.divider()
st.subheader("🎯 Key Recommendations")
if isinstance(engagement_cards, list) and len(engagement_cards) > 1:
    st.success("**Priority Actions Based on Your Data:**")
    for card in engagement_cards[:3]: # Show top 3
        st.markdown(f"- {card.split('**Recommendation:**')[-1].strip()}")
else:
    st.info("Process your data to generate personalized recommendations.")

# Footer
st.markdown("---")
st.caption("GrowthOracle AI v2.0 | End of Report")
