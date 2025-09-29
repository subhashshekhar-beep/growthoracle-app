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
        warn = sum(1 for m in self.messages if m.category == "Warning")  # FIXED: removed duplicate "m in"
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
        before = ((s < lo) | (s > hi)).sum()
        if before > 0:
            s = s.clip(lo, hi)
            vc.add("Info", "NUM_CLAMP", f"{name} clipped to bounds", lo=lo, hi=hi, affected=int(before))
    return s

def safe_dt_parse(col: pd.Series, name: str, vc: ValidationCollector) -> pd.Series:
    """Safely parse datetime series with validation logging"""
    if col is None or len(col) == 0:
        return pd.Series([], dtype='datetime64[ns]')
    
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
    req_gsc = ["date", "page", "query", "impr", "pos"]
    
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
        vc.add("Critical", "MISSING_COLMAP", "Missing/ambiguous mappings", items=missing)
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
        c1, c2, c3, c4, c5 = st.columns(5)
        ga4_map = {}
        ga4_map["msid"] = c1.selectbox(
            "MSID", ga4_df_raw.columns, 
            index=ga4_df_raw.columns.get_loc(ga4_map_guess["msid"]) if ga4_map_guess.get("msid") in ga4_df_raw.columns else 0
        )
        ga4_map["date"] = c2.selectbox(
            "Date", ga4_df_raw.columns, 
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
        ga4_map["bounce"] = st.selectbox(
            "Bounce Rate", ga4_df_raw.columns, 
            index=ga4_df_raw.columns.get_loc(ga4_map_guess["bounce"]) if ga4_map_guess.get("bounce") in ga4_df_raw.columns else 0
        )

    with st.expander("GSC Mapping", expanded=True):
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        gsc_map = {}
        gsc_map["date"] = c1.selectbox(
            "Date", gsc_df_raw.columns, 
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
        gsc_map["impr"] = c4.selectbox(
            "Impressions", gsc_df_raw.columns, 
            index=gsc_df_raw.columns.get_loc(gsc_map_guess["impr"]) if gsc_map_guess.get("impr") in gsc_df_raw.columns else 0
        )
        gsc_map["ctr"] = c5.selectbox(
            "CTR", gsc_df_raw.columns, 
            index=gsc_df_raw.columns.get_loc(gsc_map_guess["ctr"]) if gsc_map_guess.get("ctr") in gsc_df_raw.columns else 0
        )
        gsc_map["pos"] = c6.selectbox(
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
# PART 3/5: Onboarding Steps, Mapping UI, Validation UI, Robust Processing
# ============================

# [Keep all the previous code until the validation report section...]

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

# Define the standardize_dates_early function (it was missing)
def standardize_dates_early(prod_df, ga4_df, gsc_df, mappings, vc: ValidationCollector):
    """Standardize date formats early in processing pipeline"""
    
    def _ensure_utc(ts: pd.Timestamp) -> pd.Timestamp:
        if pd.isna(ts):
            return ts
        try:
            return ts.tz_convert("UTC")
        except Exception:
            try:
                return ts.tz_localize("UTC")
            except Exception:
                return ts

    def normalize_date_only(df, col_name, out_name):
        if df is not None and col_name in df.columns:
            dt = safe_dt_parse(df[col_name], col_name, vc)
            if dt is not None and len(dt) > 0:
                df[out_name] = dt.dt.date
                if dt.notna().any():
                    maxd, mind = dt.max(), dt.min()
                    maxd_utc, mind_utc = _ensure_utc(maxd), _ensure_utc(mind)
                    now_utc = pd.Timestamp.now(tz="UTC")
                    if (pd.notna(maxd_utc)) and (maxd_utc > now_utc + pd.Timedelta(days=1)):
                        vc.add("Warning", "FUTURE_DATE", f"{out_name} has future dates", sample=str(maxd_utc))
                    if (pd.notna(mind_utc)) and (mind_utc < pd.Timestamp(2020, 1, 1, tz="UTC")):
                        vc.add("Info", "OLD_DATE", f"{out_name} includes <2020 dates", earliest=str(mind_utc))

    # Process production data
    p = prod_df.copy() if prod_df is not None else None
    if p is not None and mappings["prod"].get("publish") and mappings["prod"]["publish"] in p.columns:
        p["Publish Time"] = safe_dt_parse(p[mappings["prod"]["publish"]], "Publish Time", vc)
    elif p is not None:
        for cand in detect_date_cols(p):
            if "publish" in cand.lower():
                p["Publish Time"] = safe_dt_parse(p[cand], cand, vc)
                vc.add("Info", "DATE_DETECT", "Detected publish column in Production", column=cand)
                break

    # Process GA4 data
    g4 = ga4_df.copy() if ga4_df is not None else None
    if g4 is not None:
        g4_date_col = mappings["ga4"].get("date") or "date"
        if g4_date_col in g4.columns:
            normalize_date_only(g4, g4_date_col, "date")

    # Process GSC data
    gs = gsc_df.copy() if gsc_df is not None else None
    if gs is not None:
        gs_date_col = mappings["gsc"].get("date") or "Date"
        if gs_date_col in gs.columns:
            normalize_date_only(gs, gs_date_col, "date")

    return p, g4, gs

@st.cache_data(show_spinner=False, max_entries=3)
def process_uploaded_files_complete(prod_df_raw, ga4_df_raw, gsc_df_raw, prod_map, ga4_map, gsc_map,
                                   vc_serialized: Optional[str] = None,
                                   merge_strategy: Optional[Dict[str, str]] = None):
    """COMPLETE processing pipeline that processes ALL data"""
    vc = ValidationCollector()
    
    # Load previous validation messages if provided
    if vc_serialized:
        try:
            messages = json.loads(vc_serialized)
            for item in messages:
                ctx = item.get("context", {})
                if isinstance(ctx, str):
                    try:
                        ctx = json.loads(ctx)
                    except:
                        ctx = {}
                vc.add(item["category"], item["code"], item["message"], **ctx)
        except Exception as e:
            vc.add("Warning", "VC_LOAD_FAIL", f"Failed to load previous validation: {e}")
    
    ms = merge_strategy or MERGE_STRATEGY

    # Create safe copies
    prod_df = prod_df_raw.copy() if prod_df_raw is not None else None
    ga4_df = ga4_df_raw.copy() if ga4_df_raw is not None else None
    gsc_df = gsc_df_raw.copy() if gsc_df_raw is not None else None

    # Validate and rename Production columns
    if prod_df is None or prod_map.get("msid") not in prod_df.columns:
        vc.add("Critical", "MISSING_KEY", "Production missing MSID column", want=prod_map.get("msid"))
        return None, vc
        
    try:
        prod_df.rename(columns={prod_map["msid"]: "msid"}, inplace=True)
        if prod_map.get("title") in prod_df.columns:
            prod_df.rename(columns={prod_map["title"]: "Title"}, inplace=True)
        if prod_map.get("path") in prod_df.columns:
            prod_df.rename(columns={prod_map["path"]: "Path"}, inplace=True)
        if prod_map.get("publish") in prod_df.columns:
            prod_df.rename(columns={prod_map["publish"]: "Publish Time"}, inplace=True)
    except Exception as e:
        vc.add("Critical", "PROD_RENAME_FAIL", f"Failed to rename production columns: {e}")
        return None, vc

    # Validate and rename GA4 columns
    if ga4_df is None or ga4_map.get("msid") not in ga4_df.columns:
        vc.add("Critical", "MISSING_KEY", "GA4 missing MSID column", want=ga4_map.get("msid"))
        return None, vc
        
    try:
        ga4_df.rename(columns={ga4_map["msid"]: "msid"}, inplace=True)
        if ga4_map.get("date") in ga4_df.columns:
            ga4_df.rename(columns={ga4_map["date"]: "date"}, inplace=True)
        if ga4_map.get("pageviews") in ga4_df.columns:
            ga4_df.rename(columns={ga4_map["pageviews"]: "screenPageViews"}, inplace=True)
        if ga4_map.get("users") in ga4_df.columns:
            ga4_df.rename(columns={ga4_map["users"]: "totalUsers"}, inplace=True)
        if ga4_map.get("engagement") in ga4_df.columns:
            ga4_df.rename(columns={ga4_map["engagement"]: "userEngagementDuration"}, inplace=True)
        if ga4_map.get("bounce") in ga4_df.columns:
            ga4_df.rename(columns={ga4_map["bounce"]: "bounceRate"}, inplace=True)
    except Exception as e:
        vc.add("Critical", "GA4_RENAME_FAIL", f"Failed to rename GA4 columns: {e}")
        return None, vc

    # Validate and rename GSC columns
    if gsc_df is None:
        vc.add("Critical", "MISSING_DATA", "GSC data is missing")
        return None, vc
        
    try:
        gsc_ren = {
            gsc_map["date"]: "date", 
            gsc_map["page"]: "page_url", 
            gsc_map["query"]: "Query",
            gsc_map["impr"]: "Impressions", 
            gsc_map.get("ctr", "CTR"): "CTR", 
            gsc_map["pos"]: "Position"
        }
        
        for k in list(gsc_ren.keys()):
            if k not in gsc_df.columns:
                vc.add("Critical", "MISSING_COL", f"GSC missing required column '{k}'")
                return None, vc
                
        gsc_df.rename(columns=gsc_ren, inplace=True)
    except Exception as e:
        vc.add("Critical", "GSC_RENAME_FAIL", f"Failed to rename GSC columns: {e}")
        return None, vc

    # Early date standardization
    try:
        prod_df, ga4_df, gsc_df = standardize_dates_early(
            prod_df, ga4_df, gsc_df,
            {"prod": prod_map, "ga4": ga4_map, "gsc": gsc_map}, 
            vc
        )
    except Exception as e:
        vc.add("Warning", "DATE_STD_FAIL", f"Date standardization had issues: {e}")

    # Convert MSIDs to numeric
    try:
        for df, col in [(prod_df, "msid"), (ga4_df, "msid")]:
            if df is not None and len(df) > 0:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                bad = df[col].isna().sum()
                if bad > 0:
                    vc.add("Warning", "MSID_BAD", "Non-numeric MSIDs dropped", rows=int(bad))
                df.dropna(subset=[col], inplace=True)
                if len(df) > 0:
                    df[col] = df[col].astype("int64")
    except Exception as e:
        vc.add("Warning", "MSID_CONV_FAIL", f"MSID conversion failed: {e}")

    # Extract MSID from GSC URLs
    try:
        if gsc_df is not None and len(gsc_df) > 0 and "msid" not in gsc_df.columns:
            def _extract_msid_from_url(url):
                if pd.isna(url):
                    return None
                try:
                    m = re.search(r"/(\d+)\.cms", str(url))
                    return int(m.group(1)) if m else None
                except:
                    return None
                    
            gsc_df["msid"] = gsc_df["page_url"].apply(_extract_msid_from_url)
            missing = gsc_df["msid"].isna().sum()
            if missing > 0:
                vc.add("Warning", "MSID_FROM_URL", "Some GSC rows lacked MSID in URL", unresolved=int(missing))
            gsc_df.dropna(subset=["msid"], inplace=True)
            if len(gsc_df) > 0:
                gsc_df["msid"] = gsc_df["msid"].astype("int64")
    except Exception as e:
        vc.add("Warning", "MSID_EXTRACT_FAIL", f"MSID extraction failed: {e}")

    # Numeric coercions for GSC
    try:
        if gsc_df is not None and len(gsc_df) > 0:
            if "Impressions" in gsc_df.columns:
                gsc_df["Impressions"] = coerce_numeric(gsc_df["Impressions"], "GSC.Impressions", vc, clamp=(0, float("inf")))
            
            if "CTR" in gsc_df.columns:
                if gsc_df["CTR"].dtype == object:
                    tmp = gsc_df["CTR"].astype(str).str.replace("%", "", regex=False)
                    ctr_val = pd.to_numeric(tmp, errors="coerce")
                    if len(ctr_val) > 0 and (ctr_val > 1.0).any():
                        ctr_val = ctr_val / 100.0
                        vc.add("Info", "CTR_SCALE", "CTR parsed as percentage (÷100)")
                    gsc_df["CTR"] = ctr_val
                else:
                    gsc_df["CTR"] = pd.to_numeric(gsc_df["CTR"], errors="coerce")
                
                if len(gsc_df) > 0:
                    out_of_bounds = ((gsc_df["CTR"] < 0) | (gsc_df["CTR"] > 1)).sum()
                    if out_of_bounds > 0:
                        vc.add("Warning", "CTR_CLAMP", "CTR values clamped to [0,1]", rows=int(out_of_bounds))
                        gsc_df["CTR"] = gsc_df["CTR"].clip(0, 1)
            
            if "Position" in gsc_df.columns:
                gsc_df["Position"] = coerce_numeric(gsc_df["Position"], "GSC.Position", vc, clamp=(1, 100))
    except Exception as e:
        vc.add("Warning", "GSC_NUMERIC_FAIL", f"GSC numeric conversion failed: {e}")

    # Parse categories from Path
    try:
        if prod_df is not None and len(prod_df) > 0 and "Path" in prod_df.columns:
            def parse_path(path_str):
                if not isinstance(path_str, str):
                    return ("Uncategorized", "Uncategorized")
                s = path_str.strip().strip("/")
                if not s:
                    return ("Uncategorized", "Uncategorized")
                parts = [p for p in s.split("/") if p and p.strip()]
                if len(parts) == 0:
                    return ("Uncategorized", "Uncategorized")
                elif len(parts) == 1:
                    return (parts[0], "General")
                else:
                    return (parts[0], parts[1])
                    
            cat_tuples = prod_df["Path"].apply(parse_path)
            prod_df[["L1_Category", "L2_Category"]] = pd.DataFrame(cat_tuples.tolist(), index=prod_df.index)
        else:
            if prod_df is not None and len(prod_df) > 0:
                prod_df["L1_Category"] = "Uncategorized"
                prod_df["L2_Category"] = "Uncategorized"
    except Exception as e:
        vc.add("Warning", "CATEGORY_PARSE_FAIL", f"Category parsing failed: {e}")
        if prod_df is not None and len(prod_df) > 0:
            prod_df["L1_Category"] = "Uncategorized"
            prod_df["L2_Category"] = "Uncategorized"

    # Merge GSC with Production
    before_counts = {
        "prod": len(prod_df) if prod_df is not None else 0,
        "ga4": len(ga4_df) if ga4_df is not None else 0, 
        "gsc": len(gsc_df) if gsc_df is not None else 0
    }
    
    try:
        if gsc_df is not None and len(gsc_df) > 0 and prod_df is not None and len(prod_df) > 0:
            # Get all available columns from production
            prod_cols = ["msid"]
            for col in ["Title", "Path", "Publish Time", "L1_Category", "L2_Category"]:
                if col in prod_df.columns:
                    prod_cols.append(col)
            
            merged_1 = pd.merge(
                gsc_df,
                prod_df[prod_cols].drop_duplicates(subset=["msid"]),
                on="msid", 
                how=ms.get("gsc_x_prod", "left")
            )
            vc.checkpoint("merge_gsc_prod", before=before_counts, after_m1=len(merged_1))
        else:
            vc.add("Critical", "MERGE_FAIL", "Cannot merge - missing GSC or Production data")
            return None, vc
    except Exception as e:
        vc.add("Critical", "MERGE_GSC_PROD_FAIL", f"GSC-Production merge failed: {e}")
        return None, vc

    # Prepare GA4 daily aggregates
    try:
        if ga4_df is not None and len(ga4_df) > 0 and "date" in ga4_df.columns:
            # Get numeric columns for aggregation
            numeric_cols = []
            for col in ["screenPageViews", "totalUsers", "userEngagementDuration", "bounceRate"]:
                if col in ga4_df.columns:
                    numeric_cols.append(col)
            
            if numeric_cols:
                ga4_daily = ga4_df.groupby(["msid", "date"], as_index=False)[numeric_cols].sum(min_count=1)
            else:
                ga4_daily = ga4_df.copy()
        else:
            ga4_daily = ga4_df.copy() if ga4_df is not None else pd.DataFrame()
            if "date" not in ga4_daily.columns:
                ga4_daily["date"] = pd.NaT
            vc.add("Info", "GA4_NO_DATE", "GA4 had no date; set NaT")
    except Exception as e:
        vc.add("Warning", "GA4_AGG_FAIL", f"GA4 aggregation failed: {e}")
        ga4_daily = pd.DataFrame()

    # Final merge with GA4
    try:
        if ga4_daily is not None and len(ga4_daily) > 0:
            master_df = pd.merge(merged_1, ga4_daily, on=["msid", "date"], how=ms.get("ga4_align", "left"))
        else:
            master_df = merged_1.copy()
            vc.add("Info", "NO_GA4_MERGE", "GA4 data not available for merge")
            
        vc.checkpoint("merge_ga4", after_master=len(master_df))
    except Exception as e:
        vc.add("Warning", "FINAL_MERGE_FAIL", f"Final merge with GA4 failed: {e}")
        master_df = merged_1.copy()

    # Final data cleaning
    try:
        if master_df is not None and len(master_df) > 0:
            # Deduplicate
            subset_cols = [c for c in ["date", "msid", "Query"] if c in master_df.columns]
            if len(subset_cols) >= 2:  # Need at least 2 columns for meaningful dedup
                dup_before = master_df.duplicated(subset=subset_cols).sum()
                if dup_before > 0:
                    vc.add("Info", "DEDUP", "Duplicate rows removed", count=int(dup_before))
                    master_df = master_df.drop_duplicates(subset=subset_cols, keep="first").reset_index(drop=True)

            # Impute missing values
            if "CTR" in master_df.columns:
                master_df["CTR"] = master_df["CTR"].fillna(0.0)
                
            if "Position" in master_df.columns:
                miss = master_df["Position"].isna().sum()
                if miss > 0:
                    master_df["Position"] = master_df["Position"].fillna(50.0)
                    vc.add("Info", "POSITION_IMPUTE", "Missing Position imputed to 50.0", rows=int(miss))

            # Remove rows without titles
            if "Title" in master_df.columns:
                drop_title_n = master_df["Title"].isna().sum()
                if drop_title_n > 0:
                    vc.add("Warning", "TITLE_MISSING", "Rows lacking Title dropped", rows=int(drop_title_n))
                    master_df = master_df.dropna(subset=["Title"])

            master_df["_lineage"] = "GSC→PROD→GA4"
        
    except Exception as e:
        vc.add("Warning", "CLEANING_FAIL", f"Final cleaning failed: {e}")

    return master_df, vc

# FIXED: Define vc_serialized before using it
vc_serialized = rep_df.to_json(orient="records") if not rep_df.empty else "[]"

# Process data with COMPLETE processing function
with st.spinner("Processing & merging datasets..."):
    master_df, vc_after = process_uploaded_files_complete(
        prod_df_raw, ga4_df_raw, gsc_df_raw, 
        prod_map, ga4_map, gsc_map,
        vc_serialized=vc_serialized,  # Now this variable is defined
        merge_strategy=MERGE_STRATEGY
    )

if master_df is None or master_df.empty:
    st.error("Data processing failed. Please check your files and try again.")
    if vc_after and not vc_after.to_dataframe().empty:
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

# Master data preview - NOW SHOWS ALL DATA
st.success(f"✅ Master dataset created: {master_df.shape[0]:,} rows × {master_df.shape[1]} columns")

# Show date range info
if "date" in master_df.columns:
    try:
        date_col = pd.to_datetime(master_df["date"], errors="coerce")
        if date_col.notna().any():
            min_date = date_col.min().date()
            max_date = date_col.max().date()
            st.caption(f"Date range: **{min_date}** to **{max_date}**")
    except Exception:
        pass

# Sampling for large datasets (but processes ALL data first)
if master_df.shape[0] > CONFIG["performance"]["sample_row_limit"]:
    st.info(f"Large dataset detected. For interactive analysis, sampling {CONFIG['performance']['sample_row_limit']:,} rows.")
    analysis_df = master_df.sample(
        min(CONFIG["performance"]["sample_row_limit"], len(master_df)), 
        random_state=CONFIG["performance"]["seed"]
    )
else:
    analysis_df = master_df  # Use all data if under limit

# Preview data
st.subheader("Data Preview (First 10 rows)")
st.dataframe(master_df.head(10), use_container_width=True, hide_index=True)

# Show data summary
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
# PART 4/5: Core Analysis Modules
# ============================

# Date filtering
# ============================
# PART 4/5: Core Analysis Modules - USING REAL DATA
# ============================

# Date filtering
def filter_by_date(df, start_date, end_date):
    """Filter dataframe by date range using REAL data"""
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

# Apply date filter to REAL data
start_date, end_date = st.session_state.date_range
filtered_df = filter_by_date(master_df, start_date, end_date)

TH = st.session_state.thresholds
EXPECTED_CTR = CONFIG["expected_ctr_by_rank"]

# Safe module execution
def run_module_safely(label: str, fn, *args, **kwargs):
    """Execute analysis module with comprehensive error handling"""
    try:
        with st.spinner(f"Running {label}..."):
            result = fn(*args, **kwargs)
        return result
    except Exception as e:
        error_msg = f"Module '{label}' encountered an issue and was skipped. Details: {type(e).__name__}: {e}"
        st.warning(error_msg)
        logger.error(f"[ModuleFail] {label}: {e}")
        return None

# Plotly HTML exporter
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

# Column resolution helper
def _pick_col(df, candidates):
    """Safely pick first available column from candidates"""
    if df is None:
        return None
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None

# Engagement vs Search Mismatch Analysis - USING REAL DATA
def engagement_mismatches(df):
    """Identify engagement vs search performance mismatches using REAL data"""
    if df is None or df.empty:
        return ["No data available for analysis"]
    
    d = df.copy()
    
    # Check what columns we actually have in the REAL data
    available_cols = d.columns.tolist()
    st.info(f"Available columns for analysis: {', '.join(available_cols)}")
    
    # Use REAL data to create insights
    insights = []
    
    # Analysis 1: Position vs CTR analysis
    if "Position" in d.columns and "CTR" in d.columns:
        # Remove rows with missing data
        pos_ctr_data = d[["msid", "Position", "CTR", "Title"]].dropna()
        
        if not pos_ctr_data.empty:
            # Find pages with good position but low CTR (opportunities)
            good_position_low_ctr = pos_ctr_data[(pos_ctr_data["Position"] <= 10) & (pos_ctr_data["CTR"] < 0.03)]
            if not good_position_low_ctr.empty:
                for _, row in good_position_low_ctr.head(3).iterrows():
                    insight = f"""### ⚠️ Low CTR at Good Position
**MSID:** `{row.get('msid', 'N/A')}`  
**Position:** {row['Position']:.1f} | **CTR:** {row['CTR']:.2%}  
**Title:** {str(row.get('Title', 'Unknown'))[:80]}...

**Recommendations:**  
- **Title Optimization:** Test more compelling titles  
- **Meta Description:** Improve snippet appeal  
- **Rich Results:** Implement schema markup  

*Goal: Convert high positions into more clicks*"""
                    insights.append(insight)
            
            # Find pages with high CTR but poor position (hidden gems)
            high_ctr_poor_position = pos_ctr_data[(pos_ctr_data["Position"] > 10) & (pos_ctr_data["CTR"] > 0.05)]
            if not high_ctr_poor_position.empty:
                for _, row in high_ctr_poor_position.head(2).iterrows():
                    insight = f"""### 💎 High CTR but Poor Position
**MSID:** `{row.get('msid', 'N/A')}`  
**Position:** {row['Position']:.1f} | **CTR:** {row['CTR']:.2%}  
**Title:** {str(row.get('Title', 'Unknown'))[:80]}...

**Recommendations:**  
- **Content Quality:** Improve depth and quality  
- **Backlinks:** Build authoritative links  
- **User Signals:** Enhance user experience  

*Goal: Improve rankings for high-CTR content*"""
                    insights.append(insight)
    
    # Analysis 2: Engagement duration analysis
    if "userEngagementDuration" in d.columns:
        engagement_data = d[["msid", "userEngagementDuration", "Title"]].dropna()
        if not engagement_data.empty:
            high_engagement = engagement_data.nlargest(2, "userEngagementDuration")
            for _, row in high_engagement.iterrows():
                insight = f"""### 💎 High Engagement Content
**MSID:** `{row.get('msid', 'N/A')}`  
**Avg. Duration:** {row['userEngagementDuration']:.1f}s  
**Title:** {str(row.get('Title', 'Unknown'))[:80]}...

**Recommendations:**  
- **Content Expansion:** Add more depth to this popular topic  
- **Internal Linking:** Link from high-traffic pages  
- **Update Frequency:** Keep this content current  

*Goal: Leverage engagement to improve rankings*"""
                insights.append(insight)
    
    # Analysis 3: Bounce rate analysis
    if "bounceRate" in d.columns:
        bounce_data = d[["msid", "bounceRate", "Title", "Position"]].dropna()
        if not bounce_data.empty:
            high_bounce_good_position = bounce_data[(bounce_data["bounceRate"] > 0.7) & (bounce_data["Position"] <= 15)]
            if not high_bounce_good_position.empty:
                for _, row in high_bounce_good_position.head(2).iterrows():
                    insight = f"""### 🚨 High Bounce Rate at Good Position
**MSID:** `{row.get('msid', 'N/A')}`  
**Position:** {row['Position']:.1f} | **Bounce Rate:** {row['bounceRate']:.1%}  
**Title:** {str(row.get('Title', 'Unknown'))[:80]}...

**Recommendations:**  
- **Content Match:** Ensure content matches search intent  
- **Page Speed:** Improve loading times  
- **Readability:** Enhance content structure  

*Goal: Reduce bounce rate to improve rankings*"""
                    insights.append(insight)
    
    if not insights:
        insights.append("No specific engagement-search mismatches detected. Your content appears well-balanced based on the available metrics.")
    
    return insights

def scatter_engagement_vs_search(df):
    """Create engagement vs search scatter plot using REAL data"""
    if df is None or df.empty:
        st.info("No data available for scatter plot")
        return
        
    d = df.copy()
    
    # Use actual available columns
    engagement_col = _pick_col(d, ["userEngagementDuration", "bounceRate"])
    search_col = _pick_col(d, ["Position", "CTR", "Clicks"])
    
    if not engagement_col or not search_col:
        st.info(f"Need both engagement ({engagement_col}) and search ({search_col}) metrics for scatter plot")
        return
    
    # Prepare data for scatter plot
    plot_data = d[["msid", "Title", "L1_Category", "L2_Category", engagement_col, search_col]].dropna()
    
    if plot_data.empty:
        st.info("No complete data available for scatter plot after cleaning")
        return
    
    if _HAS_PLOTLY:
        try:
            fig = px.scatter(
                plot_data, 
                x=engagement_col, 
                y=search_col,
                color="L1_Category" if "L1_Category" in plot_data.columns else None,
                hover_data=["msid", "Title"],
                title=f"Real Data: {engagement_col} vs {search_col}",
                labels={
                    engagement_col: engagement_col,
                    search_col: search_col
                }
            )
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")
            export_plot_html(fig, "engagement_vs_search")
            
            # Show data summary
            st.caption(f"Showing {len(plot_data):,} data points from your actual dataset")
        except Exception as e:
            st.error(f"Failed to create scatter plot: {e}")
    else:
        st.scatter_chart(plot_data, x=engagement_col, y=search_col)

# Category Performance Analysis - USING REAL DATA
def category_heatmap(df, value_col, title):
    """Create category performance heatmap using REAL data"""
    if not _HAS_PLOTLY:
        st.info("Heatmap visualization requires Plotly")
        return
        
    if df is None or df.empty:
        st.info("No data available for heatmap")
        return
        
    d = df.copy()
    
    # Ensure category columns exist
    if "L1_Category" not in d.columns:
        d["L1_Category"] = "Uncategorized"
    if "L2_Category" not in d.columns:
        d["L2_Category"] = "Uncategorized"
    if value_col not in d.columns:
        st.info(f"Column '{value_col}' not found in your data")
        return
    
    try:
        # Aggregate REAL data
        agg_data = d.groupby(["L1_Category", "L2_Category"]).agg(
            value=(value_col, "sum")
        ).reset_index()
        
        if agg_data.empty:
            st.info(f"No {value_col} data available for heatmap after aggregation")
            return
            
        fig = px.density_heatmap(
            agg_data, 
            x="L1_Category", 
            y="L2_Category", 
            z="value", 
            color_continuous_scale="Viridis",
            title=f"{title} - Your Real Data",
            labels={"value": value_col}
        )
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")
        export_plot_html(fig, f"heatmap_{value_col}")
        
        # Show data summary
        st.caption(f"Showing category distribution based on {len(d):,} rows of your actual data")
        
    except Exception as e:
        st.error(f"Failed to create heatmap: {e}")

def analyze_category_performance(df):
    """Analyze category-level performance using REAL data"""
    if df is None or df.empty:
        st.warning("No data available for category analysis")
        return pd.DataFrame()
    
    d = df.copy()
    
    # Ensure category columns exist
    if "L1_Category" not in d.columns:
        d["L1_Category"] = "Uncategorized"
    if "L2_Category" not in d.columns:
        d["L2_Category"] = "Uncategorized"
    if "msid" not in d.columns:
        d["msid"] = range(len(d))
    
    # Identify available metrics in REAL data
    engagement_col = _pick_col(d, ["userEngagementDuration", "engagement_duration"])
    pageviews_col = _pick_col(d, ["screenPageViews", "pageviews"])
    users_col = _pick_col(d, ["totalUsers", "users"])
    clicks_col = _pick_col(d, ["Clicks", "clicks"])
    impressions_col = _pick_col(d, ["Impressions", "impr"])
    
    if not any([engagement_col, pageviews_col, users_col, clicks_col, impressions_col]):
        st.warning("No performance metrics found for category analysis")
        return pd.DataFrame()
    
    # Prepare numeric columns
    numeric_cols = [c for c in [engagement_col, pageviews_col, users_col, clicks_col, impressions_col] if c]
    for col in numeric_cols:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    
    # Build aggregation dictionary based on available columns
    agg_dict = {"msid": "nunique"}  # Count unique articles
    
    # Add available metrics to aggregation
    if pageviews_col:
        agg_dict[pageviews_col] = "sum"
    elif users_col:
        agg_dict[users_col] = "sum"
        
    if engagement_col:
        agg_dict[engagement_col] = "mean"
        
    if clicks_col:
        agg_dict[clicks_col] = "sum"
        
    if impressions_col:
        agg_dict[impressions_col] = "sum"
    
    # Group by categories using REAL data
    try:
        grouped = (d.groupby(["L1_Category", "L2_Category"])
                     .agg(agg_dict)
                     .reset_index())
        
        # Rename columns for clarity
        column_rename = {"msid": "total_articles"}
        if pageviews_col:
            column_rename[pageviews_col] = "total_traffic"
        elif users_col:
            column_rename[users_col] = "total_traffic"
        if engagement_col:
            column_rename[engagement_col] = "avg_engagement_duration"
        if clicks_col:
            column_rename[clicks_col] = "total_gsc_clicks"
        if impressions_col:
            column_rename[impressions_col] = "total_impressions"
            
        grouped = grouped.rename(columns=column_rename)
        
        # Calculate performance indices if we have traffic data
        if "total_traffic" in grouped.columns and len(grouped) > 0:
            site_avg_traffic = grouped["total_traffic"].mean()
            if site_avg_traffic > 0:
                grouped["traffic_index"] = grouped["total_traffic"] / site_avg_traffic
            else:
                grouped["traffic_index"] = 0
                
        if "avg_engagement_duration" in grouped.columns and len(grouped) > 0:
            site_avg_eng = grouped["avg_engagement_duration"].mean()
            if site_avg_eng > 0:
                grouped["engagement_index"] = grouped["avg_engagement_duration"] / site_avg_eng
            else:
                grouped["engagement_index"] = 0
        
        # Classify quadrants if we have both indices
        def classify_quadrant(row):
            if "traffic_index" not in row or "engagement_index" not in row:
                return "N/A"
                
            traffic_high = row["traffic_index"] >= 1.0
            engagement_high = row["engagement_index"] >= 1.0
            
            if traffic_high and engagement_high:
                return "Stars"
            elif not traffic_high and engagement_high:
                return "Hidden Gems" 
            elif traffic_high and not engagement_high:
                return "Workhorses"
            else:
                return "Underperformers"
                
        if "traffic_index" in grouped.columns and "engagement_index" in grouped.columns:
            grouped["quadrant"] = grouped.apply(classify_quadrant, axis=1)
        else:
            grouped["quadrant"] = "N/A"
            
        st.success(f"✅ Category analysis completed on {len(d):,} rows of real data")
        return grouped
        
    except Exception as e:
        st.error(f"Category analysis failed: {e}")
        return pd.DataFrame()

# Trends & Forecasting - USING REAL DATA
def forecast_series(daily_series, periods=14):
    """Generate time series forecasts using REAL data"""
    if daily_series is None or len(daily_series) < 7:
        st.warning("Need at least 7 days of data for forecasting")
        return None
        
    try:
        # Ensure daily frequency and handle missing values
        daily_series = daily_series.asfreq("D").fillna(method="ffill").fillna(0)
        
        if _HAS_STM and len(daily_series) >= 14:
            try:
                # Use Holt-Winters exponential smoothing
                seasonal_periods = min(7, len(daily_series))
                model = ExponentialSmoothing(
                    daily_series, 
                    trend="add", 
                    seasonal="add", 
                    seasonal_periods=seasonal_periods
                )
                fit = model.fit(optimized=True)
                forecast = fit.forecast(periods)
                
                # Calculate confidence intervals
                residuals = daily_series - fit.fittedvalues
                std_dev = residuals.std()
                lower = forecast - 1.96 * std_dev
                upper = forecast + 1.96 * std_dev
                
                return pd.DataFrame({
                    "date": forecast.index, 
                    "forecast": forecast.values, 
                    "low": lower.values, 
                    "high": upper.values
                })
                
            except Exception as e:
                st.info(f"Advanced forecasting unavailable, using simple method: {e}")
        
        # Fallback: simple moving average
        moving_avg = daily_series.rolling(window=7, min_periods=1).mean()
        last_value = moving_avg.iloc[-1] if not moving_avg.empty else daily_series.mean()
        
        future_dates = pd.date_range(
            start=daily_series.index.max() + pd.Timedelta(days=1), 
            periods=periods, 
            freq="D"
        )
        
        # Simple linear projection based on recent trend
        recent_trend = daily_series.tail(7).pct_change().mean()
        forecast_values = [last_value * (1 + recent_trend) ** i for i in range(periods)]
        
        return pd.DataFrame({
            "date": future_dates,
            "forecast": forecast_values,
            "low": [v * 0.8 for v in forecast_values],  # 20% lower bound
            "high": [v * 1.2 for v in forecast_values]  # 20% upper bound
        })
        
    except Exception as e:
        st.error(f"Forecasting failed: {e}")
        return None

def time_series_trends(df, metric_col, title):
    """Create time series trend visualization using REAL data"""
    if df is None or df.empty or "date" not in df.columns:
        st.info("Time series analysis requires date column and data")
        return
        
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"])
    
    if d.empty or metric_col not in d.columns:
        st.info(f"No valid {metric_col} data available for time series")
        return
        
    # Prepare metric
    d[metric_col] = pd.to_numeric(d[metric_col], errors="coerce").fillna(0)
    
    # Aggregate by date using REAL data
    daily_data = d.groupby("date")[metric_col].sum().reset_index()
    
    if daily_data.empty:
        st.info(f"No {metric_col} data available after aggregation")
        return
    
    if _HAS_PLOTLY:
        try:
            fig = px.line(
                daily_data, 
                x="date", 
                y=metric_col, 
                title=f"{title} - Your Real Data",
                labels={metric_col: metric_col.replace("_", " ").title()}
            )
            
            # Add range slider for better navigation
            fig.update_layout(
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=7, label="1w", step="day", stepmode="backward"),
                            dict(count=1, label="1m", step="month", stepmode="backward"),
                            dict(step="all")
                        ])
                    ),
                    rangeslider=dict(visible=True),
                    type="date"
                )
            )
            
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")
            export_plot_html(fig, f"timeseries_{metric_col}")
            
            # Show data summary
            date_range = f"{daily_data['date'].min().strftime('%Y-%m-%d')} to {daily_data['date'].max().strftime('%Y-%m-%d')}"
            st.caption(f"Showing {len(daily_data)} days of real {metric_col} data from {date_range}")
            
        except Exception as e:
            st.error(f"Failed to create time series chart: {e}")
    else:
        st.line_chart(daily_data.set_index("date")[metric_col])

# ============================
# MAIN ANALYSIS UI
# ============================

st.header("📊 Advanced Analytics & Insights - REAL DATA ANALYSIS")

# Module 4: Engagement vs Search Mismatch
st.subheader("Module 4: Engagement vs Search Performance Mismatch")
st.caption("Identify content with high engagement but poor search performance (Hidden Gems) and vice versa (Clickbait Risks)")

engagement_cards = run_module_safely(
    "Engagement vs Search Analysis", 
    engagement_mismatches, 
    filtered_df
)

if isinstance(engagement_cards, list):
    for card in engagement_cards:
        st.markdown(card)
        
# Engagement vs Search Scatter Plot
st.subheader("Engagement vs Search Scatter Analysis")
run_module_safely("Engagement Scatter Plot", scatter_engagement_vs_search, filtered_df)

st.divider()
# ============================
# PART 5/5: Complete Analysis & Export
# ============================

# Module 2: Category Performance
# ============================
# PART 5/5: Complete Analysis & Export - USING REAL DATA
# ============================

# Module 5: Category Performance - COMPLETE ANALYSIS
st.subheader("Module 5: Category Performance Analysis - COMPLETE DATA")
st.caption("Complete analysis of all your content categories using 100% of your data")

category_results = run_module_safely(
    "Category Performance", 
    analyze_category_performance, 
    filtered_df  # Using ALL filtered data, not samples
)

if isinstance(category_results, pd.DataFrame) and not category_results.empty:
    st.success(f"✅ Analyzed {len(category_results)} categories from your complete dataset")
    st.dataframe(category_results, use_container_width=True, hide_index=True)
    
    # Category visualizations using REAL data
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Category Traffic Distribution - Real Data")
        # Use available traffic metric
        traffic_col = _pick_col(filtered_df, ["screenPageViews", "totalUsers", "Clicks", "Impressions"])
        if traffic_col:
            run_module_safely(
                "Traffic Heatmap", 
                category_heatmap, 
                filtered_df, traffic_col, "Category Traffic"
            )
        else:
            st.info("No traffic metrics available for heatmap")
    
    with col2:
        # Top categories bar chart using REAL data
        if "total_gsc_clicks" in category_results.columns:
            st.subheader("Top Categories by GSC Clicks - Real Data")
            top_categories = category_results.nlargest(10, "total_gsc_clicks")
            if not top_categories.empty:
                # Use Plotly for better visualization
                if _HAS_PLOTLY:
                    fig = px.bar(
                        top_categories,
                        x="L2_Category",
                        y="total_gsc_clicks",
                        title="Top Categories by GSC Clicks",
                        color="L1_Category"
                    )
                    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
                else:
                    st.bar_chart(top_categories.set_index("L2_Category")["total_gsc_clicks"])
        elif "total_traffic" in category_results.columns:
            st.subheader("Top Categories by Traffic - Real Data")
            top_categories = category_results.nlargest(10, "total_traffic")
            if not top_categories.empty:
                if _HAS_PLOTLY:
                    fig = px.bar(
                        top_categories,
                        x="L2_Category",
                        y="total_traffic",
                        title="Top Categories by Traffic",
                        color="L1_Category"
                    )
                    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
                else:
                    st.bar_chart(top_categories.set_index("L2_Category")["total_traffic"])
else:
    st.info("Category performance analysis requires category columns and performance metrics in your data")

st.divider()

# Module 6: Trends & Forecasting - USING REAL AUGUST 2025 DATA
st.subheader("Module 6: Trends & Forecasting - Your August 2025 Data")
st.caption("Analyze historical trends and generate performance forecasts using your actual data")

if "date" in filtered_df.columns and not filtered_df.empty:
    # Prepare time series data from REAL August 2025 data
    ts_data = filtered_df.copy()
    ts_data["date"] = pd.to_datetime(ts_data["date"], errors="coerce")
    ts_data = ts_data.dropna(subset=["date"])
    
    if not ts_data.empty:
        # Show actual date range from user's data
        actual_start = ts_data["date"].min().strftime('%Y-%m-%d')
        actual_end = ts_data["date"].max().strftime('%Y-%m-%d')
        st.info(f"Your data date range: **{actual_start}** to **{actual_end}**")
        
        # Select primary metric for forecasting from available columns
        primary_metric = _pick_col(ts_data, ["totalUsers", "screenPageViews", "Clicks", "Impressions"])
        
        if primary_metric:
            # Daily aggregation from REAL data
            daily_series = ts_data.groupby("date")[primary_metric].sum().sort_index()
            
            st.subheader(f"{primary_metric} Trends - Your August 2025 Data")
            
            # Show actual time series from user's data
            run_module_safely(
                f"{primary_metric} Time Series", 
                time_series_trends, 
                ts_data, primary_metric, f"{primary_metric} Over Time"
            )
            
            # Forecasting section
            if len(daily_series) >= 7:  # Need at least 7 days for meaningful analysis
                st.subheader(f"14-Day Forecast - Based on Your August 2025 Data")
                
                forecast_data = forecast_series(daily_series, periods=14)
                
                if forecast_data is not None and _HAS_PLOTLY:
                    # Create forecast visualization using REAL data
                    historical_df = daily_series.reset_index().rename(columns={"index": "date", primary_metric: "value"})
                    
                    fig = go.Figure()
                    
                    # Historical data from user's August 2025
                    fig.add_trace(go.Scatter(
                        x=historical_df["date"],
                        y=historical_df["value"],
                        name="Your Historical Data",
                        line=dict(color="blue", width=3)
                    ))
                    
                    # Forecast based on user's data
                    fig.add_trace(go.Scatter(
                        x=forecast_data["date"],
                        y=forecast_data["forecast"],
                        name="Forecast",
                        line=dict(color="red", width=3, dash="dash")
                    ))
                    
                    # Confidence interval
                    fig.add_trace(go.Scatter(
                        x=pd.concat([forecast_data["date"], forecast_data["date"][::-1]]),
                        y=pd.concat([forecast_data["high"], forecast_data["low"][::-1]]),
                        fill="toself",
                        fillcolor="rgba(255,0,0,0.2)",
                        line=dict(color="rgba(255,255,255,0)"),
                        name="Confidence Interval",
                        showlegend=True
                    ))
                    
                    fig.update_layout(
                        title=f"{primary_metric} - 14-Day Forecast Based on Your August 2025 Data",
                        xaxis_title="Date",
                        yaxis_title=primary_metric,
                        hovermode="x unified"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
                    export_plot_html(fig, f"forecast_{primary_metric}")
                    
                    # Show forecast details
                    with st.expander("Forecast Details", expanded=False):
                        st.dataframe(forecast_data, use_container_width=True, hide_index=True)
                        st.caption("Forecast generated from your actual August 2025 data patterns")
                
                elif len(daily_series) < 7:
                    st.warning(f"Need at least 7 days of {primary_metric} data for forecasting. You have {len(daily_series)} days.")
            else:
                st.info(f"Need at least 7 days of {primary_metric} data for forecasting")
        else:
            st.info("No suitable metrics found for trend analysis in your data")
    else:
        st.info("No valid date data available for trend analysis")
else:
    st.info("Date column required for trend analysis and forecasting")

# Additional real metric trends
st.subheader("Additional Metric Trends - Your Real Data")
if "date" in filtered_df.columns and not filtered_df.empty:
    # Show available metrics for additional trends
    available_metrics = []
    for metric in ["Impressions", "CTR", "Position", "userEngagementDuration", "bounceRate"]:
        if metric in filtered_df.columns:
            available_metrics.append(metric)
    
    if available_metrics:
        # Show up to 2 additional metrics
        for metric in available_metrics[:2]:
            if metric != primary_metric:  # Don't repeat the primary metric
                run_module_safely(
                    f"Trend {metric}", 
                    time_series_trends, 
                    ts_data, metric, f"{metric} Over Time"
                )
    else:
        st.info("No additional metrics available for trend analysis")
else:
    st.info("No date data available for additional trends")

# ============================
# EXPORT & SUMMARY - REAL DATA
# ============================

st.divider()
st.subheader("📤 Export Results - Your Real Data")

# Create export options based on REAL data
col1, col2, col3 = st.columns(3)

with col1:
    if not filtered_df.empty:
        download_df_button(
            filtered_df, 
            f"growthoracle_analysis_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            "Download Analysis Data"
        )

with col2:
    if isinstance(category_results, pd.DataFrame) and not category_results.empty:
        download_df_button(
            category_results,
            f"category_performance_{pd.Timestamp.now().strftime('%Y%m%d')}.csv", 
            "Download Category Analysis"
        )

with col3:
    # Summary statistics from REAL data
    with st.expander("Real Data Summary", expanded=True):
        if not filtered_df.empty:
            st.metric("Total Rows Analyzed", f"{len(filtered_df):,}")
            if "msid" in filtered_df.columns:
                st.metric("Unique Articles", f"{filtered_df['msid'].nunique():,}")
            if "Clicks" in filtered_df.columns:
                total_clicks = filtered_df["Clicks"].sum()
                st.metric("Total GSC Clicks", f"{total_clicks:,}")

# Final recommendations based on REAL data
st.divider()
st.subheader("🎯 Key Recommendations - Based on Your Real Data")

if isinstance(engagement_cards, list) and len(engagement_cards) > 0:
    st.success("**Priority Actions Based on Your Data:**")
    
    # Count actual insights from real data
    hidden_gems = sum(1 for card in engagement_cards if "Hidden Gem" in card)
    clickbait_risks = sum(1 for card in engagement_cards if "Clickbait Risk" in card)
    low_ctr = sum(1 for card in engagement_cards if "Low CTR" in card)
    high_bounce = sum(1 for card in engagement_cards if "High Bounce" in card)
    
    if hidden_gems > 0:
        st.info(f"• **Optimize {hidden_gems} Hidden Gem(s):** Improve SEO for high-engagement content")
    if clickbait_risks > 0:
        st.warning(f"• **Fix {clickbait_risks} Clickbait Risk(s):** Enhance content quality and UX")
    if low_ctr > 0:
        st.warning(f"• **Address {low_ctr} Low CTR Issue(s):** Improve title and meta descriptions")
    if high_bounce > 0:
        st.error(f"• **Reduce {high_bounce} High Bounce Rate(s):** Enhance content relevance and UX")
    
    if isinstance(category_results, pd.DataFrame):
        stars = category_results[category_results["quadrant"] == "Stars"]
        workhorses = category_results[category_results["quadrant"] == "Workhorses"]
        hidden_gem_cats = category_results[category_results["quadrant"] == "Hidden Gems"]
        
        if not stars.empty:
            st.success(f"• **Leverage {len(stars)} Star Category(ies):** {', '.join(stars['L1_Category'].tolist())}")
        if not workhorses.empty:
            st.info(f"• **Optimize {len(workhorses)} Workhorse Category(ies):** {', '.join(workhorses['L1_Category'].tolist())}")
        if not hidden_gem_cats.empty:
            st.info(f"• **Promote {len(hidden_gem_cats)} Hidden Gem Category(ies):** {', '.join(hidden_gem_cats['L1_Category'].tolist())}")
else:
    st.info("Upload and process your data to get personalized recommendations based on your actual content performance")

# Footer
st.markdown("---")
st.caption("GrowthOracle AI v2.0 | Analyzing Your Real Data | August 2025 Insights")



