# ============================
# app.py — GrowthOracle AI — Next Gen (Full)
# PART 1/5: Imports, Config, Logger, Validation Core
# ============================
import os, io, re, sys, math, json, time, logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta, date
from math import sqrt

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
st.caption("Time-aware insights • Interactive analytics • Explainable recommendations")

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
    # Expected CTR curve for rank positions (1..9 provided; we’ll extrapolate beyond)
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
    if series is None or len(series) == 0:
        return pd.Series([], dtype=float)
    s = pd.to_numeric(series, errors="coerce")
    bad = s.isna().sum()
    if bad > 0:
        vc.add("Warning", "NUM_COERCE", f"Non-numeric values coerced to NaN in {name}", bad_rows=int(bad))
    if clamp and len(s) > 0:
        lo, hi = clamp
        s_clamped = s.copy()
        out_of_bounds_mask = ((s_clamped < lo) | (s_clamped > hi)) if hi is not None else (s_clamped < lo)
        before = out_of_bounds_mask.sum()
        if before > 0:
            s_clamped.loc[out_of_bounds_mask] = s_clamped.clip(lower=lo, upper=hi)
            vc.add("Info", "NUM_CLAMP", f"{name} clipped to bounds", lo=lo, hi=hi, affected=int(before))
            s = s_clamped
    return s

def safe_dt_parse(col: pd.Series, name: str, vc: ValidationCollector) -> pd.Series:
    if col is None or len(col) == 0:
        return pd.Series([], dtype='datetime64[ns, UTC]')
    parsed = pd.to_datetime(col, errors="coerce", utc=True)
    bad = parsed.isna().sum()
    if bad > 0:
        vc.add("Warning", "DATE_PARSE", f"Unparseable datetime in {name}", bad_rows=int(bad))
    return parsed

def detect_date_cols(df: pd.DataFrame) -> List[str]:
    if df is None or df.empty:
        return []
    return [c for c in df.columns if any(k in c.lower() for k in ["date", "dt", "time", "timestamp", "publish"])]

DEFAULT_VALIDATION_STRICTNESS = "Standard"
MERGE_STRATEGY = {"gsc_x_prod": "left", "ga4_align": "left"}

def add_lineage(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if df is None:
        return None
    df = df.copy()
    df["_source"] = source
    return df

# Initialize session state
def init_state_defaults():
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
    return pd.DataFrame({
        "Msid": [101, 102, 103],
        "Title": ["Budget 2025 highlights explained", "IPL 2025 schedule & squads", "Monsoon updates: city-by-city guide"],
        "Path": ["/business/budget-2025/highlights", "/sports/cricket/ipl-2025/schedule", "/news/monsoon/guide"],
        "Publish Time": ["2025-08-01 09:15:00", "2025-08-10 18:30:00", "2025-09-01 07:00:00"]
    })

def _make_template_ga4():
    return pd.DataFrame({
        "customEvent:msid": [101, 101, 102, 102, 103],
        "date": ["2025-08-01", "2025-08-02", "2025-08-10", "2025-08-11", "2025-09-01"],
        "screenPageViews": [5000, 6000, 15000, 12000, 7000],
        "totalUsers": [4000, 4500, 10000, 8000, 5200],
        "userEngagementDuration": [52.3, 48.2, 41.0, 44.7, 63.1],
        "bounceRate": [0.42, 0.45, 0.51, 0.49, 0.38]
    })

def _make_template_gsc():
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

if not all([prod_file, ga4_file, gsc_file]):
    st.warning("Please upload all three CSV files to proceed")
    st.stop()

# Read raw files
vc_read = ValidationCollector()
prod_df_raw = read_csv_safely(prod_file, "Production", vc_read)
ga4_df_raw = read_csv_safely(ga4_file, "GA4", vc_read)
gsc_df_raw = read_csv_safely(gsc_file, "GSC", vc_read)
if any(df is None or df.empty for df in [prod_df_raw, ga4_df_raw, gsc_df_raw]):
    st.error("One or more uploaded files appear empty/unreadable. See Validation section below.")
    st.dataframe(vc_read.to_dataframe(), use_container_width=True, hide_index=True)
    st.stop()

# Mapping helpers
def _guess_colmap(prod_df, ga4_df, gsc_df):
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
    prod_map, ga4_map, gsc_map = _guess_colmap(prod_df, ga4_df, gsc_df)
    if prod_df is not None:
        prod_dates = [c for c in detect_date_cols(prod_df) if "publish" in c.lower() or "time" in c.lower()]
        if prod_dates and not prod_map.get("publish"):
            prod_map["publish"] = prod_dates[0]
    if ga4_df is not None and not ga4_map.get("date"):
        for c in detect_date_cols(ga4_df):
            if c.lower() == "date":
                ga4_map["date"] = c; break
    if gsc_df is not None and not gsc_map.get("date"):
        for c in detect_date_cols(gsc_df):
            if c.lower() == "date":
                gsc_map["date"] = c; break
    return prod_map, ga4_map, gsc_map

def validate_columns_presence(prod_map, ga4_map, gsc_map, vc: ValidationCollector):
    req_prod = ["msid"]
    req_ga4 = ["msid"]
    req_gsc = ["date", "page", "query", "clicks", "impr", "pos"]
    missing = []
    for k in req_prod:
        if not prod_map.get(k): missing.append(f"Production: {k}")
    for k in req_ga4:
        if not ga4_map.get(k): missing.append(f"GA4: {k}")
    for k in req_gsc:
        if not gsc_map.get(k): missing.append(f"GSC: {k}")
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
        prod_map["msid"] = c1.selectbox("MSID", prod_df_raw.columns,
            index=prod_df_raw.columns.get_loc(prod_map_guess["msid"]) if prod_map_guess.get("msid") in prod_df_raw.columns else 0)
        prod_map["title"] = c2.selectbox("Title", prod_df_raw.columns,
            index=prod_df_raw.columns.get_loc(prod_map_guess["title"]) if prod_map_guess.get("title") in prod_df_raw.columns else 0)
        prod_map["path"] = c3.selectbox("Path", prod_df_raw.columns,
            index=prod_df_raw.columns.get_loc(prod_map_guess["path"]) if prod_map_guess.get("path") in prod_df_raw.columns else 0)
        prod_map["publish"] = c4.selectbox("Publish Time", prod_df_raw.columns,
            index=prod_df_raw.columns.get_loc(prod_map_guess["publish"]) if prod_map_guess.get("publish") in prod_df_raw.columns else 0)

    with st.expander("GA4 Mapping", expanded=True):
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        ga4_map = {}
        ga4_map["msid"] = c1.selectbox("MSID (GA4)", ga4_df_raw.columns,
            index=ga4_df_raw.columns.get_loc(ga4_map_guess["msid"]) if ga4_map_guess.get("msid") in ga4_df_raw.columns else 0)
        ga4_map["date"] = c2.selectbox("Date (GA4)", ga4_df_raw.columns,
            index=ga4_df_raw.columns.get_loc(ga4_map_guess["date"]) if ga4_map_guess.get("date") in ga4_df_raw.columns else 0)
        ga4_map["pageviews"] = c3.selectbox("Pageviews", ga4_df_raw.columns,
            index=ga4_df_raw.columns.get_loc(ga4_map_guess["pageviews"]) if ga4_map_guess.get("pageviews") in ga4_df_raw.columns else 0)
        ga4_map["users"] = c4.selectbox("Users", ga4_df_raw.columns,
            index=ga4_df_raw.columns.get_loc(ga4_map_guess["users"]) if ga4_map_guess.get("users") in ga4_df_raw.columns else 0)
        ga4_map["engagement"] = c5.selectbox("Engagement Duration", ga4_df_raw.columns,
            index=ga4_df_raw.columns.get_loc(ga4_map_guess["engagement"]) if ga4_map_guess.get("engagement") in ga4_df_raw.columns else 0)
        ga4_map["bounce"] = c6.selectbox("Bounce Rate", ga4_df_raw.columns,
            index=ga4_df_raw.columns.get_loc(ga4_map_guess["bounce"]) if ga4_map_guess.get("bounce") in ga4_df_raw.columns else 0)

    with st.expander("GSC Mapping", expanded=True):
        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        gsc_map = {}
        gsc_map["date"] = c1.selectbox("Date (GSC)", gsc_df_raw.columns,
            index=gsc_df_raw.columns.get_loc(gsc_map_guess["date"]) if gsc_map_guess.get("date") in gsc_df_raw.columns else 0)
        gsc_map["page"] = c2.selectbox("Page URL", gsc_df_raw.columns,
            index=gsc_df_raw.columns.get_loc(gsc_map_guess["page"]) if gsc_map_guess.get("page") in gsc_df_raw.columns else 0)
        gsc_map["query"] = c3.selectbox("Query", gsc_df_raw.columns,
            index=gsc_df_raw.columns.get_loc(gsc_map_guess["query"]) if gsc_map_guess.get("query") in gsc_df_raw.columns else 0)
        gsc_map["clicks"] = c4.selectbox("Clicks", gsc_df_raw.columns,
            index=gsc_df_raw.columns.get_loc(gsc_map_guess["clicks"]) if gsc_map_guess.get("clicks") in gsc_df_raw.columns else 0)
        gsc_map["impr"] = c5.selectbox("Impressions", gsc_df_raw.columns,
            index=gsc_df_raw.columns.get_loc(gsc_map_guess["impr"]) if gsc_map_guess.get("impr") in gsc_df_raw.columns else 0)
        gsc_map["ctr"] = c6.selectbox("CTR", gsc_df_raw.columns,
            index=gsc_df_raw.columns.get_loc(gsc_map_guess["ctr"]) if gsc_map_guess.get("ctr") in gsc_df_raw.columns else 0)
        gsc_map["pos"] = c7.selectbox("Position", gsc_df_raw.columns,
            index=gsc_df_raw.columns.get_loc(gsc_map_guess["pos"]) if gsc_map_guess.get("pos") in gsc_df_raw.columns else 0)

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
# ============================
# PART 3/5: Validation UI, Robust Processing & Post-merge
# ============================

if "mapping" not in st.session_state:
    st.warning("Please complete **Step 2** (column mapping) first.")
    st.stop()

prod_map = st.session_state.mapping["prod"]
ga4_map = st.session_state.mapping["ga4"]
gsc_map = st.session_state.mapping["gsc"]

def run_validation_pipeline(prod_df_raw, ga4_df_raw, gsc_df_raw, prod_map, ga4_map, gsc_map):
    vc = ValidationCollector()
    missing_map = validate_columns_presence(prod_map, ga4_map, gsc_map, vc)

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
            if only_p: vc.add("Info", "MSID_ONLY_PROD", "MSIDs appear only in Production", count=int(only_p))
            if only_g: vc.add("Warning", "MSID_ONLY_GSC", "MSIDs appear only in GSC", count=int(only_g))
    except Exception as e:
        vc.add_exc("preview_msid_consistency", e)
    return vc

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

    # 1) Rename to standard names
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

    # 2) Early date & msid standardization
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

    # 3) Robust numeric conversion
        # --- 3. ROBUST NUMERIC CONVERSION (IMPROVED) ---
    if gsc_df is not None:
        # 3a) Pre-clean text numbers: remove thousands separators and NBSPs
        def _clean_num_str(s: pd.Series) -> pd.Series:
            return (
                s.astype(str)
                 .str.replace(r"[\u2009\u00A0,\s]", "", regex=True)  # thin space, NBSP, commas, spaces
                 .str.replace(r"^-+$", "", regex=True)              # dashes to empty
                 .str.strip()
            )

        for col in ["Clicks", "Impressions", "Position"]:
            if col in gsc_df.columns and gsc_df[col].dtype == "object":
                gsc_df[col] = _clean_num_str(gsc_df[col])

        # CTR may arrive like "3.4%" or "3,4 %"
        if "CTR" in gsc_df.columns and gsc_df["CTR"].dtype == "object":
            tmp = _clean_num_str(gsc_df["CTR"])
            tmp = tmp.str.replace("%", "", regex=False)
            gsc_df["CTR"] = pd.to_numeric(tmp, errors="coerce") / 100.0

        # 3b) Now coerce to numeric + clamp
        for col, clamp in [("Clicks", (0, None)), ("Impressions", (0, None)), ("Position", (1, 100))]:
            if col in gsc_df.columns:
                gsc_df[col] = coerce_numeric(gsc_df[col], f"GSC.{col}", vc, clamp=clamp)

        # If CTR didn’t exist, compute it
        if "CTR" in gsc_df.columns:
            gsc_df["CTR"] = coerce_numeric(gsc_df["CTR"], "GSC.CTR", vc, clamp=(0, 1))
        elif "Clicks" in gsc_df.columns and "Impressions" in gsc_df.columns:
            vc.add("Info", "CTR_CALCULATED", "CTR column calculated from Clicks/Impressions")
            gsc_df["CTR"] = (gsc_df["Clicks"] / gsc_df["Impressions"].replace(0, np.nan)).fillna(0)

    # 4) Merge
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

    # 5) Final cleaning & features
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
    except Exception:
        pass

if master_df.shape[0] > CONFIG["performance"]["sample_row_limit"]:
    st.info(f"Large dataset detected. For interactive analysis, a sample of {CONFIG['performance']['sample_row_limit']:,} rows will be used.")
    analysis_df = master_df.sample(
        n=CONFIG["performance"]["sample_row_limit"],
        random_state=CONFIG["performance"]["seed"]
    )
else:
    analysis_df = master_df

# --- Data Preview + Full CSV download ---
st.subheader("Data Preview (First 10 rows)")
col_prev, col_btn = st.columns([0.80, 0.20])

with col_btn:
    # Button to download the ENTIRE merged dataset
    download_df_button(
        master_df,
        f"master_merged_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        "Download Full Merged dataset (CSV)"
    )

# (Optional) make the first-10 preview friendlier by showing 0 instead of None for key metrics
preview_df = master_df.head(10).copy()
for _c in ["Clicks","Impressions"]:
    if _c in preview_df.columns:
        preview_df[_c] = pd.to_numeric(preview_df[_c], errors="coerce").fillna(0)

with col_prev:
    st.dataframe(preview_df, use_container_width=True, hide_index=True)


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

def _pick_col(df, candidates):
    if df is None: return None
    for candidate in candidates:
        if candidate in df.columns: return candidate
    return None

def _expected_ctr_for_pos(pos: float) -> float:
    """Extrapolate expected CTR for a given average position."""
    if pd.isna(pos): return np.nan
    p = max(1, min(50, float(pos)))
    base = EXPECTED_CTR.get(int(min(9, round(p))), EXPECTED_CTR[9])
    if p <= 9:
        return base
    # gentle decay beyond rank 9
    return base * (9.0 / p) ** 0.5

def engagement_mismatches(df):
    """Card-style insights (unchanged logic for short, human-friendly takeaways)."""
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

def build_engagement_mismatch_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return ALL rows that meet mismatch conditions, for CSV download."""
    if df is None or df.empty: return pd.DataFrame()
    d = df.copy()
    for col in ["Clicks","Impressions","CTR","Position","userEngagementDuration","bounceRate","screenPageViews","totalUsers"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")
    # compute expected CTR by avg position if available
    d["expected_ctr"] = d["Position"].apply(_expected_ctr_for_pos) if "Position" in d.columns else np.nan
    d["ctr_deficit"] = (d["expected_ctr"] - d["CTR"]) if all(c in d.columns for c in ["expected_ctr","CTR"]) else np.nan

    tags = []
    for _, row in d.iterrows():
        tag = None
        pos = row.get("Position", np.nan)
        ctr = row.get("CTR", np.nan)
        impr = row.get("Impressions", 0)
        br = row.get("bounceRate", np.nan)
        exp = row.get("expected_ctr", np.nan)
        # Apply min impressions to cut noise
        if not pd.isna(impr) and impr < TH["min_impressions"]:
            tags.append(None); continue
        # Cases
        if not pd.isna(pos) and not pd.isna(ctr) and not pd.isna(exp):
            deficit_pct = (exp - ctr) / exp if exp and exp > 0 else np.nan
            if pos <= 10 and not pd.isna(deficit_pct) and deficit_pct >= (TH["ctr_deficit_pct"]/100.0):
                tag = "Low CTR @ Good Position"
        if tag is None and not pd.isna(pos) and not pd.isna(ctr):
            if pos > 15 and ctr > 0.05:
                tag = "Hidden Gem: High CTR @ Poor Position"
        if tag is None and not pd.isna(br) and not pd.isna(pos):
            if br > 0.70 and pos <= 15:
                tag = "High Bounce @ Good Position"
        tags.append(tag)
    d["Mismatch_Tag"] = tags
    d = d[~d["Mismatch_Tag"].isna()].copy()

    # Keep useful columns for export
    keep_cols = [c for c in [
        "date","msid","Title","Path","L1_Category","L2_Category","Query",
        "Position","CTR","expected_ctr","ctr_deficit","Clicks","Impressions",
        "screenPageViews","totalUsers","userEngagementDuration","bounceRate"
    ] if c in d.columns]
    return d[["Mismatch_Tag"] + keep_cols].sort_values(["Mismatch_Tag","msid"])

def scatter_engagement_vs_search(df: pd.DataFrame):
    """Clean 'Engagement vs Search' with two simple views: Scatter (bubbles) and Scatter + marginals."""
    if df is None or df.empty:
        st.info("No data available for scatter plot")
        return

    d = df.copy()

    # Available fields
    y_opts    = [c for c in ["userEngagementDuration", "totalUsers", "screenPageViews"] if c in d.columns]
    x_opts    = [c for c in ["CTR", "Position"] if c in d.columns]
    size_opts = [c for c in ["Clicks", "Impressions", "screenPageViews", "totalUsers"] if c in d.columns]
    color_opts= [c for c in ["L1_Category", "L2_Category"] if c in d.columns]

    if not y_opts or not x_opts or not size_opts:
        st.info("Need engagement (Y), search (X), and a size metric.")
        return

    # Controls
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        x = st.selectbox("Search (X)", x_opts, index=0, help="CTR or Position (left=better when Position).")
    with c2:
        y = st.selectbox("Engagement (Y)", y_opts, index=0, help="Pick the engagement proxy.")
    with c3:
        size_col = st.selectbox("Bubble size", size_opts, index=0, help="Bigger bubble = more impact.")
    with c4:
        color_col = st.selectbox("Color by", color_opts, index=0)

    f1, f2, f3 = st.columns(3)
    with f1:
        min_impr = st.number_input("Min Impressions", min_value=0, value=200, step=50) if "Impressions" in d.columns else 0
    with f2:
        min_clicks = st.number_input("Min Clicks", min_value=0, value=5, step=5) if "Clicks" in d.columns else 0
    with f3:
        gran = st.radio("Granularity", ["Raw (date×query×page)", "One per page (weighted)"], index=1, horizontal=True)

    # Clean numerics quickly
    for c in list({x, y, size_col} | {"CTR","Position","Clicks","Impressions"}):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    # Filters
    if "Impressions" in d.columns and min_impr > 0:
        d = d[d["Impressions"] >= min_impr]
    if "Clicks" in d.columns and min_clicks > 0:
        d = d[d["Clicks"] >= min_clicks]

    # Aggregate to one dot per page (weighted by Impressions→best, else Clicks)
    if gran.startswith("One per page"):
        w = "Impressions" if "Impressions" in d.columns else ("Clicks" if "Clicks" in d.columns else None)
        keys = ["msid","Title","L1_Category","L2_Category"]
        for k in keys:
            if k not in d.columns:
                d[k] = np.nan
        if w:
            d["_w"] = pd.to_numeric(d[w], errors="coerce").fillna(0)
            grp = d.groupby(keys, as_index=False).agg(
                size_val=(size_col, "sum"),
                w_sum=("_w","sum"),
                clicks_sum=("Clicks","sum") if "Clicks" in d.columns else ("_w","sum"),
                impr_sum=("Impressions","sum") if "Impressions" in d.columns else ("_w","sum"),
                pos_w=("Position", lambda s: np.average(pd.to_numeric(s, errors='coerce'), weights=d.loc[s.index, "_w"]) if "Position" in d.columns else np.nan),
                y_w=(y,  lambda s: np.average(pd.to_numeric(s, errors='coerce'), weights=d.loc[s.index, "_w"]))
            )
            if x == "CTR" and "clicks_sum" in grp.columns and "impr_sum" in grp.columns:
                grp["CTR"] = np.where(grp["impr_sum"] > 0, grp["clicks_sum"]/grp["impr_sum"], np.nan)
            elif x == "Position" and "pos_w" in grp.columns:
                grp["Position"] = grp["pos_w"]
            grp[y] = grp["y_w"]
            plot_data = grp.rename(columns={"size_val": size_col}).drop(columns=["w_sum","clicks_sum","impr_sum","pos_w","y_w"], errors="ignore")
        else:
            plot_data = d.groupby(keys, as_index=False).agg(**{x:(x,"mean"), y:(y,"mean"), size_col:(size_col,"sum")})
    else:
        base_cols = ["msid","Title",color_col,x,y,size_col]
        extra = [c for c in ["Query","date"] if c in d.columns]
        plot_data = d[[c for c in base_cols+extra if c in d.columns]].copy()

    plot_data = plot_data.dropna(subset=[x, y, size_col])
    if plot_data.empty:
        st.info("No points left after filters.")
        return

    # Limit points for readability
    max_points = st.slider("Max points", 500, 10000, 3000, step=500)
    if len(plot_data) > max_points:
        plot_data = plot_data.nlargest(max_points, size_col)

    # Two views only
    view_mode = st.radio("View", ["Scatter (bubbles)", "Scatter + marginals"], index=0, horizontal=True)

    if not _HAS_PLOTLY:
        st.scatter_chart(plot_data, x=x, y=y)
        return

    # Build figure
    if view_mode == "Scatter (bubbles)":
        fig = px.scatter(
            plot_data,
            x=x, y=y, size=size_col, color=color_col,
            hover_data=[c for c in ["msid","Title","Query","date"] if c in plot_data.columns],
            size_max=60, opacity=0.8,
            title=f"Engagement ({y}) vs Search ({x})"
        )
    else:
        fig = px.scatter(
            plot_data,
            x=x, y=y, size=size_col, color=color_col, opacity=0.75, size_max=60,
            hover_data=[c for c in ["msid","Title","Query","date"] if c in plot_data.columns],
            marginal_x="histogram", marginal_y="violin",
            title=f"Engagement ({y}) vs Search ({x}) — with distributions"
        )

    # --- Directional guides (axis titles + corner tags + caption) ---
    if x == "Position":
        fig.update_xaxes(autorange="reversed", title="Position")
        dircap = f"Left = better rank, Up = higher {y}"
        good_x, bad_x = 0.05, 0.95  # paper coords
    else:
        fig.update_xaxes(title="CTR")
        dircap = f"Right = higher CTR, Up = higher {y}"
        good_x, bad_x = 0.95, 0.05
    fig.update_yaxes(title=f"{y} (higher ↑)")

    # Render + export + data download
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    if "export_plot_html" in globals():
        export_plot_html(fig, "engagement_vs_search")
    download_df_button(
        plot_data,
        f"scatter_data_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        "Download plotted points (CSV)"
    )


# --- Utility: Export Plotly figure as downloadable HTML (NEW) ---
def export_plot_html(fig, name: str):
    """Show a download button to export a Plotly fig as a standalone HTML file."""
    if to_html is None or fig is None:
        st.info("Plotly HTML export not available.")
        return
    try:
        html_str = to_html(fig, include_plotlyjs="cdn", full_html=True)
        st.download_button(
            label=f"Export {name} (HTML)",
            data=html_str.encode("utf-8"),
            file_name=f"{name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.html",
            mime="text/html",
            use_container_width=True
        )
    except Exception as e:
        st.warning(f"Failed to export plot: {e}")

def category_treemap(cat_df: pd.DataFrame, metric_choice: str, per_article: bool):
    if not _HAS_PLOTLY:
        st.info("Treemap requires Plotly.")
        return
    if cat_df is None or cat_df.empty:
        st.info("No category data.")
        return

    pretty, col, asc = _resolve_cat_metric(metric_choice, per_article)

    df = cat_df.copy()
    # Area value
    if metric_choice == "Avg Position":
        # Area by volume (impressions) when viewing a ratio-like metric
        if "total_impressions" in df.columns:
            df["metric_value"] = pd.to_numeric(df["total_impressions"], errors="coerce").fillna(0)
        else:
            df["metric_value"] = 1
        color_col = "avg_position_weighted"
        color_title = "Avg Position (lower better)"
        color_scale = "RdYlGn_r"  # reversed (green = low number)
    else:
        df["metric_value"] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        color_col = "ctr_weighted" if "ctr_weighted" in df.columns else "metric_value"
        color_title = "Weighted CTR"
        color_scale = "RdYlGn"

    fig = px.treemap(
        df,
        path=["L1_Category","L2_Category"],
        values="metric_value",
        color=color_col,
        color_continuous_scale=color_scale,
        hover_data={
            "total_articles": True,
            "total_gsc_clicks": True,
            "total_impressions": True,
            "total_pageviews": True,
            "total_users": True,
            "avg_position_weighted": ":.2f",
            "ctr_weighted": ":.2%"
        },
        title=f"Treemap — {pretty} (color = {color_title})"
    )
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    if "export_plot_html" in globals():
        export_plot_html(fig, f"treemap_{col}")

    download_df_button(
        df[["L1_Category","L2_Category","total_articles","total_gsc_clicks","total_impressions",
            "total_pageviews","total_users","avg_position_weighted","ctr_weighted"]],
        f"treemap_data_{col}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        "Download treemap data (CSV)"
    )

def category_heatmap(cat_df: pd.DataFrame, metric_choice: str, per_article: bool):
    if not _HAS_PLOTLY:
        st.info("Heatmap requires Plotly.")
        return
    if cat_df is None or cat_df.empty:
        st.info("No category data.")
        return

    pretty, col, asc = _resolve_cat_metric(metric_choice, per_article)
    df = cat_df.copy()

    # Pivot: rows=L2, cols=L1 (keeps labels readable)
    try:
        pv = df.pivot_table(
            index="L2_Category", columns="L1_Category",
            values=col, aggfunc="sum"
        ).fillna(0)
    except Exception:
        st.info("Not enough category variety for a heatmap.")
        return

    # Build fig
    color_scale = "RdYlGn_r" if metric_choice == "Avg Position" else "RdYlGn"
    fig = px.imshow(
        pv,
        labels=dict(x="L1 Category", y="L2 Category", color=pretty),
        color_continuous_scale=color_scale,
        title=f"Heatmap — {pretty}"
    )
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    if "export_plot_html" in globals():
        export_plot_html(fig, f"heatmap_{col}")

    # Download the matrix used
    pv_reset = pv.reset_index()
    download_df_button(
        pv_reset,
        f"heatmap_matrix_{col}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        "Download heatmap matrix (CSV)"
    )

def _series_mode(series: pd.Series):
    try:
        m = series.mode()
        return m.iloc[0] if not m.empty else None
    except Exception:
        return None

def analyze_category_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate category-level performance + weighted metrics."""
    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy()
    # Ensure categories exist
    for c in ["L1_Category", "L2_Category"]:
        if c not in d.columns:
            d[c] = "Uncategorized"

    # Numericize
    for c in ["Clicks","Impressions","screenPageViews","totalUsers","userEngagementDuration","Position","CTR"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    # Fallback CTR if missing
    if "CTR" not in d.columns and {"Clicks","Impressions"}.issubset(d.columns):
        d["CTR"] = (d["Clicks"] / d["Impressions"].replace(0, np.nan)).fillna(0)

    # Weights for weighted means
    w = "Impressions" if "Impressions" in d.columns else ("Clicks" if "Clicks" in d.columns else None)
    if w is None:
        d["_w"] = 1.0
    else:
        d["_w"] = d[w].fillna(0)

    # Group
    agg = d.groupby(["L1_Category","L2_Category"], as_index=False).agg(
        total_articles=("msid","nunique"),
        total_gsc_clicks=("Clicks","sum") if "Clicks" in d.columns else ("msid","count"),
        total_impressions=("Impressions","sum") if "Impressions" in d.columns else ("msid","count"),
        total_pageviews=("screenPageViews","sum") if "screenPageViews" in d.columns else ("msid","count"),
        total_users=("totalUsers","sum") if "totalUsers" in d.columns else ("msid","count"),
        avg_engagement_s=("userEngagementDuration","mean") if "userEngagementDuration" in d.columns else ("msid","count"),
    )

    # Weighted CTR (ΣClicks / ΣImpressions)
    if {"total_gsc_clicks","total_impressions"}.issubset(agg.columns):
        agg["ctr_weighted"] = np.where(
            agg["total_impressions"] > 0,
            agg["total_gsc_clicks"] / agg["total_impressions"],
            np.nan
        )
    else:
        agg["ctr_weighted"] = np.nan

    # Weighted Position (by Impressions→best, else Clicks→else equal)
    if "Position" in d.columns:
        wp = d.groupby(["L1_Category","L2_Category"]).apply(
            lambda g: np.average(g["Position"].dropna(), weights=g["_w"].loc[g["Position"].dropna().index])
                      if g["Position"].notna().any() and g["_w"].sum() > 0 else np.nan
        ).reset_index(name="avg_position_weighted")
        agg = agg.merge(wp, on=["L1_Category","L2_Category"], how="left")
    else:
        agg["avg_position_weighted"] = np.nan

    # Per-article efficiency
    agg["users_per_article"] = np.where(agg["total_articles"]>0, agg["total_users"]/agg["total_articles"], np.nan)
    agg["pvs_per_article"]   = np.where(agg["total_articles"]>0, agg["total_pageviews"]/agg["total_articles"], np.nan)
    agg["clicks_per_article"]= np.where(agg["total_articles"]>0, agg["total_gsc_clicks"]/agg["total_articles"], np.nan)
    agg["impr_per_article"]  = np.where(agg["total_articles"]>0, agg["total_impressions"]/agg["total_articles"], np.nan)

    return agg
def _resolve_cat_metric(metric_choice: str, per_article: bool) -> Tuple[str, str, bool]:
    """
    Returns (pretty_label, df_column_name, ascending_sort)
    ascending_sort=True for 'Avg Position' (lower is better), else False.
    """
    if per_article and metric_choice in {"Users","Page Views","Clicks","Impressions"}:
        col = {
            "Users":"users_per_article",
            "Page Views":"pvs_per_article",
            "Clicks":"clicks_per_article",
            "Impressions":"impr_per_article",
        }[metric_choice]
        return f"{metric_choice} per Article", col, False

    mapping = {
        "Users": ("total_users", False),
        "Page Views": ("total_pageviews", False),
        "Clicks": ("total_gsc_clicks", False),
        "Impressions": ("total_impressions", False),
        "Avg Position": ("avg_position_weighted", True),  # lower is better
    }
    pretty = metric_choice
    col, asc = mapping.get(metric_choice, ("total_gsc_clicks", False))
    return pretty, col, asc


def forecast_series(daily_series, periods=14):
    if daily_series is None or len(daily_series) < 7: return None
    try:
        daily_series = daily_series.asfreq("D").fillna(method="ffill").fillna(0)
        if _HAS_STM and len(daily_series) >= 14:
            model = ExponentialSmoothing(daily_series, trend="add", seasonal="add", seasonal_periods=7).fit()
            forecast = model.forecast(periods)
            std_err = np.std(model.resid) * np.sqrt(np.arange(1, periods + 1))
            return pd.DataFrame({"date": forecast.index, "forecast": forecast, "low": forecast - 1.96 * std_err, "high": forecast + 1.96 * std_err})
        last_val = daily_series.rolling(7).mean().iloc[-1]
        future_dates = pd.date_range(start=daily_series.index.max() + pd.Timedelta(days=1), periods=periods)
        return pd.DataFrame({"date": future_dates, "forecast": [last_val] * periods, "low": last_val * 0.8, "high": last_val * 1.2})
    except Exception as e:
        st.error(f"Forecasting failed: {e}"); return None

def time_series_trends(df, metric_col, title):
    """Generic line chart (we'll skip calling this for Clicks as requested)."""
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
        except Exception as e:
            st.error(f"Failed to create time series chart: {e}")
    else:
        st.line_chart(daily_data)

# --- Growth Efficiency (Resources → Outcomes) ---

def compute_category_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate inputs (articles) and outputs and compute efficiency per article."""
    if df is None or df.empty: return pd.DataFrame()

    d = df.copy()
    # Ensure numeric
    for col in ["totalUsers","screenPageViews","Clicks","Impressions","userEngagementDuration","bounceRate","Position"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")

    g = d.groupby(["L1_Category","L2_Category"]).agg(
        total_articles = pd.NamedAgg(column="msid", aggfunc=lambda s: pd.Series(s).nunique()),
        total_users    = pd.NamedAgg(column="totalUsers", aggfunc="sum") if "totalUsers" in d.columns else pd.NamedAgg(column="msid", aggfunc=lambda s: np.nan),
        total_pvs      = pd.NamedAgg(column="screenPageViews", aggfunc="sum") if "screenPageViews" in d.columns else pd.NamedAgg(column="msid", aggfunc=lambda s: np.nan),
        total_clicks   = pd.NamedAgg(column="Clicks", aggfunc="sum") if "Clicks" in d.columns else pd.NamedAgg(column="msid", aggfunc=lambda s: np.nan),
        total_impr     = pd.NamedAgg(column="Impressions", aggfunc="sum") if "Impressions" in d.columns else pd.NamedAgg(column="msid", aggfunc=lambda s: np.nan),
        avg_eng_s      = pd.NamedAgg(column="userEngagementDuration", aggfunc="mean") if "userEngagementDuration" in d.columns else pd.NamedAgg(column="msid", aggfunc=lambda s: np.nan),
        avg_bounce     = pd.NamedAgg(column="bounceRate", aggfunc="mean") if "bounceRate" in d.columns else pd.NamedAgg(column="msid", aggfunc=lambda s: np.nan),
        avg_position   = pd.NamedAgg(column="Position", aggfunc="mean") if "Position" in d.columns else pd.NamedAgg(column="msid", aggfunc=lambda s: np.nan)
    ).reset_index()

    # Per-article efficiency (guard against div/0)
    g["users_per_article"] = g["total_users"]  / g["total_articles"].replace(0, np.nan)
    g["pvs_per_article"]   = g["total_pvs"]    / g["total_articles"].replace(0, np.nan)
    g["clicks_per_article"]= g["total_clicks"] / g["total_articles"].replace(0, np.nan)
    g["impr_per_article"]  = g["total_impr"]   / g["total_articles"].replace(0, np.nan)

    # Tidy NA
    g = g.replace([np.inf, -np.inf], np.nan)

    return g

def plot_efficiency_quadrant(cat_df: pd.DataFrame, outcome: str, y_mode: str = "Total") -> None:
    """
    outcome ∈ {"total_users","total_pvs","total_clicks","total_impr"}
    y_mode: "Total" or "Per Article"
    """
    if not _HAS_PLOTLY:
        st.info("Plotly required for the quadrant chart."); return
    if cat_df is None or cat_df.empty or "total_articles" not in cat_df.columns:
        st.info("No category efficiency data to plot."); return
    if outcome not in cat_df.columns and f"{outcome.split('_',1)[1]}_per_article" not in cat_df.columns:
        st.info("Selected outcome not available."); return

    df = cat_df.copy()
    x = "total_articles"
    if y_mode == "Total":
        y = outcome
        y_label = outcome.replace("_"," ").title()
    else:
        # map total_users -> users_per_article, etc.
        per_map = {
            "total_users": "users_per_article",
            "total_pvs": "pvs_per_article",
            "total_clicks": "clicks_per_article",
            "total_impr": "impr_per_article",
        }
        y = per_map.get(outcome)
        y_label = y.replace("_"," ").title() if y else "Per Article"

    df = df.dropna(subset=[x, y])
    if df.empty:
        st.info("Nothing to display after cleaning."); return

    # Medians for quadrant split
    x_med = df[x].median()
    y_med = df[y].median()

    fig = px.scatter(
        df, x=x, y=y, color="L1_Category", size="total_clicks" if "total_clicks" in df.columns else None,
        hover_data=["L1_Category","L2_Category","total_articles","total_users","total_pvs","users_per_article","pvs_per_article"],
        title=f"Resources → Outcomes Quadrant ({'Total' if y_mode=='Total' else 'Efficiency'})"
    )

    # Add median lines
    fig.add_hline(y=y_med, line_dash="dash", opacity=0.4)
    fig.add_vline(x=x_med, line_dash="dash", opacity=0.4)

    # Annotations for quadrants
    fig.update_layout(margin=dict(l=10,r=10,t=50,b=10))
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")

def opportunity_lists(cat_df: pd.DataFrame, outcome: str, y_mode: str = "Total") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (under_invested, over_invested) tables with potential gain/excess vs median production."""
    if cat_df is None or cat_df.empty: return pd.DataFrame(), pd.DataFrame()

    df = cat_df.copy()
    x = "total_articles"
    if y_mode == "Total":
        y = outcome
        y_per_article_col = {
            "total_users": "users_per_article",
            "total_pvs": "pvs_per_article",
            "total_clicks": "clicks_per_article",
            "total_impr": "impr_per_article",
        }[outcome]
    else:
        # If plotting efficiency, still compute potential using per-article metric
        per_map = {
            "total_users": "users_per_article",
            "total_pvs": "pvs_per_article",
            "total_clicks": "clicks_per_article",
            "total_impr": "impr_per_article",
        }
        y = per_map[outcome]
        y_per_article_col = y

    df = df.dropna(subset=[x, y, y_per_article_col])
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    x_med = df[x].median()
    y_med = df[y].median()

    # Tags
    df["tag"] = np.where((df[x] < x_med) & (df[y] >= y_med), "Under-invested", 
                  np.where((df[x] >= x_med) & (df[y] < y_med), "Over-invested", "Other"))

    # Potential: move each category to median production at current efficiency
    df["delta_articles_to_median"] = (x_med - df[x]).clip(lower=0)  # only for under-invested
    df["potential_gain"] = (df["delta_articles_to_median"] * df[y_per_article_col]).round(0)

    under = (df[df["tag"] == "Under-invested"]
             .sort_values(["potential_gain", y_per_article_col], ascending=False)
             [["L1_Category","L2_Category","total_articles",y,y_per_article_col,"delta_articles_to_median","potential_gain"]]
             .rename(columns={
                 y: ("Outcome (Y)" if y_mode=="Total" else y_label if 'y_label' in locals() else "Outcome"),
                 y_per_article_col: "Outcome per Article"
             }))

    over = (df[df["tag"] == "Over-invested"]
            .assign(excess_articles_vs_median=(df[x] - x_med).clip(lower=0))
            .sort_values(["excess_articles_vs_median", x], ascending=False)
            [["L1_Category","L2_Category","total_articles","excess_articles_vs_median", y_per_article_col]]
            .rename(columns={y_per_article_col: "Outcome per Article"}))

    return under, over

# ============================
# PART 5/5: Complete Analysis UI & Exports
# ============================
st.header("📊 Advanced Analytics & Insights")

# Module: Engagement vs Search Mismatch — cards + full CSV
st.subheader("Module 1 : Engagement vs. Search Performance Mismatch")
st.caption("Identify 'Hidden Gems' (high CTR at poor positions), low CTR at good positions, and high bounce at good positions.")

engagement_cards = engagement_mismatches(filtered_df)
if isinstance(engagement_cards, list):
    for card in engagement_cards:
        st.markdown(card)

# Full mismatch table + CSV export
mismatch_df = build_engagement_mismatch_table(filtered_df)
if mismatch_df is not None and not mismatch_df.empty:
    st.info(f"Found **{len(mismatch_df):,}** mismatch rows.")
    with st.expander("Preview mismatch rows (first 200)", expanded=False):
        st.dataframe(mismatch_df.head(200), use_container_width=True, hide_index=True)
    download_df_button(mismatch_df, f"engagement_search_mismatch_full_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                       "Download ALL Mismatch Rows (CSV)")
else:
    st.info("No mismatch rows matched your thresholds and filters.")

# Cleaner Scatter
st.subheader("Module 2 : Engagement vs. Search Scatter Analysis")
scatter_engagement_vs_search(filtered_df)

st.divider()

# Module 5: Category Performance
st.subheader("Category Performance Analysis")

# Build / refresh the aggregate once
category_results = analyze_category_performance(filtered_df)
if not isinstance(category_results, pd.DataFrame) or category_results.empty:
    st.info("Category analysis could not be completed. Check data and mappings.")
    st.stop()

# Shared controls (like Module 4)
ctrl1, ctrl2, ctrl3 = st.columns([0.45, 0.25, 0.30])
with ctrl1:
    metric_choice = st.selectbox(
        "Metric",
        ["Users","Clicks","Page Views","Impressions","Avg Position"],
        index=1  # default Clicks
    )
with ctrl2:
    per_article = st.checkbox("Per Article", value=False, help="Efficiency view (disabled for Avg Position)")
    if metric_choice == "Avg Position":
        per_article = False
with ctrl3:
    topn = st.slider("Top N for ranking", 5, 30, 10, step=1)

pretty, perf_col, sort_asc = _resolve_cat_metric(metric_choice, per_article)

# Show the aggregate table (optional)
with st.expander("Category table (full)", expanded=False):
    st.dataframe(category_results, use_container_width=True, hide_index=True)
    download_df_button(
        category_results,
        f"category_aggregate_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        "Download category table (CSV)"
    )

# Visualizations
col1, col2 = st.columns(2)
with col1:
    st.subheader("Category Traffic Distribution")
    st.caption("Treemap (recommended)")
    category_treemap(category_results, metric_choice, per_article)
    st.caption("Heatmap")
    category_heatmap(category_results, metric_choice, per_article)

with col2:
    st.subheader("Top Categories by Performance")
    # Rank by selected metric
    if perf_col in category_results.columns:
        top_cats = (category_results
                    .dropna(subset=[perf_col])
                    .sort_values(perf_col, ascending=sort_asc)
                    .head(topn))
        # Friendly axis note
        if metric_choice == "Avg Position":
            st.caption("Lower = better (avg rank).")
        else:
            st.caption("Higher = better.")

        if _HAS_PLOTLY:
            fig = px.bar(
                top_cats.sort_values(perf_col, ascending=not sort_asc),
                x=perf_col, y="L2_Category",
                color="L1_Category",
                orientation="h",
                title=f"Top {topn} by {pretty}"
            )
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")
            if "export_plot_html" in globals():
                export_plot_html(fig, f"top_categories_{perf_col}")
        else:
            st.bar_chart(top_cats.set_index("L2_Category")[perf_col])

        download_df_button(
            top_cats[["L1_Category","L2_Category", perf_col]],
            f"top_categories_{perf_col}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            "Download Top categories (CSV)"
        )
    else:
        st.info(f"No data for metric: {pretty}")


# Module 6: Trends & Forecasting
st.subheader("Module 4: Trends & Forecasting")
st.caption("We’ve removed the 'Clicks over time' chart per your request. Choose a different trend metric below. The forecast shows dashed line with a shaded 95% interval.")

if "date" in filtered_df.columns and not filtered_df.empty:
    ts_data = filtered_df.copy()
    # Prefer non-Clicks metric for the trend line
    trend_metric_choices = [c for c in ["totalUsers", "screenPageViews", "Impressions"] if c in ts_data.columns]
    selected_trend_metric = st.selectbox("Trend metric (Clicks excluded)", trend_metric_choices, index=0 if trend_metric_choices else 0)

    if selected_trend_metric:
        st.subheader(f"Overall Trend: {selected_trend_metric}")
        time_series_trends(ts_data, selected_trend_metric, f"{selected_trend_metric} Over Time")

    # Forecast: use a sensible primary metric (prefer users/pageviews; fallback to impressions; only if available)
    primary_metric = _pick_col(ts_data, ["totalUsers", "screenPageViews", "Impressions"])
    if primary_metric:
        daily_series = ts_data.groupby(pd.to_datetime(ts_data['date']))[primary_metric].sum()
        if len(daily_series) >= 7:
            st.subheader(f"14-Day Forecast for {primary_metric}")
            forecast_data = forecast_series(daily_series, periods=14)
            if forecast_data is not None and _HAS_PLOTLY:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=daily_series.index, y=daily_series.values, name="Historical"))
                fig.add_trace(go.Scatter(x=forecast_data["date"], y=forecast_data["forecast"], name="Forecast", line=dict(dash="dash")))
                fig.add_trace(go.Scatter(x=forecast_data["date"], y=forecast_data["high"], fill=None, mode='lines', line_color='rgba(200,0,0,0.25)', name="Upper 95%"))
                fig.add_trace(go.Scatter(x=forecast_data["date"], y=forecast_data["low"], fill='tonexty', mode='lines', line_color='rgba(200,0,0,0.25)', name="Lower 95%"))
                fig.update_layout(margin=dict(l=10,r=10,t=40,b=10), legend=dict(orientation="h"))
                st.plotly_chart(fig, use_container_width=True, theme="streamlit")
                # concise explanation
                st.caption("Dashed line = forecast, shaded band = 95% interval. Based on 7-day seasonal smoothing when available.")
                # metrics
                total_fc = float(forecast_data["forecast"].sum())
                total_lo = float(forecast_data["low"].sum())
                total_hi = float(forecast_data["high"].sum())
                m1, m2, m3 = st.columns(3)
                m1.metric("Projected next 14-day total", f"{total_fc:,.0f}")
                m2.metric("Low (95%)", f"{total_lo:,.0f}")
                m3.metric("High (95%)", f"{total_hi:,.0f}")

                # CSV export
                fc_df = forecast_data.copy()
                fc_df["date"] = fc_df["date"].dt.strftime("%Y-%m-%d")
                download_df_button(fc_df, f"forecast_{primary_metric}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                                   f"Download {primary_metric} Forecast (CSV)")
    else:
        st.info("No suitable metrics (Users/Pageviews/Impressions) found for forecasting.")
else:
    st.info("Date column required for trend analysis and forecasting.")

# --- Growth Efficiency (Resources → Outcomes) ---
st.divider()
st.subheader("📈 Module 5: Growth Efficiency — Resources → Outcomes")

cat_eff = compute_category_efficiency(filtered_df)
if cat_eff is None or cat_eff.empty:
    st.info("No category efficiency data available.")
else:
    # Choose the outcome proxy
    outcome_choice = st.selectbox(
        "Outcome metric",
        [c for c in ["total_users","total_pvs","total_clicks","total_impr"] if c in cat_eff.columns],
        format_func=lambda x: {
            "total_users":"Users",
            "total_pvs":"Pageviews",
            "total_clicks":"GSC Clicks",
            "total_impr":"GSC Impressions"
        }[x]
    )

    y_mode = st.radio("Y-axis", ["Total","Per Article"], index=0, horizontal=True)

    # Quadrant chart
    plot_efficiency_quadrant(cat_eff, outcome_choice, y_mode=y_mode)

    # Leaderboard
    with st.expander("Efficiency Table (downloadable)", expanded=False):
        show_cols = ["L1_Category","L2_Category","total_articles",
                     "total_users","total_pvs","total_clicks","total_impr",
                     "users_per_article","pvs_per_article","clicks_per_article","impr_per_article",
                     "avg_eng_s","avg_bounce","avg_position"]
        show_cols = [c for c in show_cols if c in cat_eff.columns]
        st.dataframe(cat_eff[show_cols].sort_values("pvs_per_article" if "pvs_per_article" in cat_eff.columns else "users_per_article", ascending=False),
                     use_container_width=True, hide_index=True)
        download_df_button(cat_eff[show_cols], f"growth_efficiency_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                           "Download Growth Efficiency (CSV)")

    # Action lists
    under, over = opportunity_lists(cat_eff, outcome_choice, y_mode=y_mode)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🚀 Under-invested Winners (scale production)")
        if not under.empty:
            st.dataframe(under.head(15), use_container_width=True, hide_index=True)
            download_df_button(under, f"under_invested_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                               "Download Under-invested (CSV)")
        else:
            st.info("None detected at current thresholds.")
    with c2:
        st.markdown("### 🧰 Over-invested Laggards (fix or reduce)")
        if not over.empty:
            st.dataframe(over.head(15), use_container_width=True, hide_index=True)
            download_df_button(over, f"over_invested_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                               "Download Over-invested (CSV)")
        else:
            st.info("None detected at current thresholds.")


# ============================
# EXPORT & SUMMARY
# ============================
st.divider()
st.subheader("📤 Export & Summary")

col1, col2, col3, col4 = st.columns(4)
with col1:
    download_df_button(filtered_df, f"growthoracle_analysis_{pd.Timestamp.now().strftime('%Y%m%d')}.csv", "Download Analysis Data")
with col2:
    if isinstance(category_results, pd.DataFrame) and not category_results.empty:
        download_df_button(category_results, f"category_performance_{pd.Timestamp.now().strftime('%Y%m%d')}.csv", "Download Category Analysis")
with col3:
    if mismatch_df is not None and not mismatch_df.empty:
        download_df_button(mismatch_df, f"engagement_search_mismatch_full_{pd.Timestamp.now().strftime('%Y%m%d')}.csv", "Download Mismatch Rows (CSV)")
with col4:
    if "Clicks" in filtered_df.columns:
        total_clicks = pd.to_numeric(filtered_df["Clicks"], errors="coerce").fillna(0).sum()
        st.metric("Total GSC Clicks (analyzed)", f"{int(total_clicks):,}")

st.divider()
st.subheader("🎯 Key Recommendations")
if isinstance(engagement_cards, list) and len(engagement_cards) > 1:
    st.success("**Priority Actions Based on Your Data:**")
    for card in engagement_cards[:3]:
        st.markdown(f"- {card.split('**Recommendation:**')[-1].strip()}")
else:
    st.info("Process your data to generate personalized recommendations.")

# Footer
st.markdown("---")
st.caption("GrowthOracle AI v2.0 | End of Report")







