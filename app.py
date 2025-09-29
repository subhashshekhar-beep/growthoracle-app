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
    st.sidebar.warning("PyYAML not installed - using default config")

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except ImportError:
    _HAS_ST = False
    st.sidebar.info("Sentence transformers not available - similarity features limited")

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
    st.sidebar.warning("Plotly not available - charts disabled")

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    _HAS_STM = True
except ImportError:
    _HAS_STM = False
    st.sidebar.info("Statsmodels not available - forecasting disabled")

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
    warn = sum(1 for m in self.messages if m.category == "Warning") # <--- Corrected
    info = sum(1 for m in self.messages if m.category == "Info")    # <--- Corrected
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
            df[out_name] = dt.dt.date
            if dt.notna().any():
                maxd, mind = dt.max(), dt.min()
                maxd_utc, mind_utc = _ensure_utc(maxd), _ensure_utc(mind)
                now_utc = pd.Timestamp.now(tz="UTC")
                if pd.notna(maxd_utc) and (maxd_utc > now_utc + pd.Timedelta(days=1)):
                    vc.add("Warning", "FUTURE_DATE", f"{out_name} has future dates", sample=str(maxd_utc))
                if pd.notna(mind_utc) and (mind_utc < pd.Timestamp(2020, 1, 1, tz="UTC")):
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

# ---- Session defaults + sidebar ----
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

# ============================
# PART 3/5: Onboarding Steps, Mapping UI, Validation UI, Robust Processing
# ============================

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

# ---- Robust processing function ----
@st.cache_data(show_spinner=False)
def process_uploaded_files(prod_df_raw, ga4_df_raw, gsc_df_raw, prod_map, ga4_map, gsc_map,
                           vc_serialized: Optional[str] = None,
                           merge_strategy: Optional[Dict[str, str]] = None):
    """Process and merge uploaded files with comprehensive validation"""
    vc = ValidationCollector()
    
    # Load previous validation messages if provided
    if vc_serialized:
        try:
            for item in json.loads(vc_serialized):
                ctx = item.get("context")
                ctx = json.loads(ctx) if isinstance(ctx, str) else (ctx or {})
                vc.add(item["category"], item["code"], item["message"], **ctx)
        except Exception as e:
            vc.add("Warning", "VC_LOAD_FAIL", f"Failed to load previous validation: {e}")
    
    ms = merge_strategy or MERGE_STRATEGY

    # Create copies to avoid modifying originals
    prod_df = prod_df_raw.copy() if prod_df_raw is not None else None
    ga4_df = ga4_df_raw.copy() if ga4_df_raw is not None else None
    gsc_df = gsc_df_raw.copy() if gsc_df_raw is not None else None

    # Validate and rename Production columns
    if prod_df is None or prod_map.get("msid") not in prod_df.columns:
        vc.add("Critical", "MISSING_KEY", "Production missing MSID column", want=prod_map.get("msid"))
        return None, vc
        
    prod_df.rename(columns={prod_map["msid"]: "msid"}, inplace=True)
    if prod_map.get("title") in prod_df.columns:
        prod_df.rename(columns={prod_map["title"]: "Title"}, inplace=True)
    if prod_map.get("path") in prod_df.columns:
        prod_df.rename(columns={prod_map["path"]: "Path"}, inplace=True)
    if prod_map.get("publish") in prod_df.columns:
        prod_df.rename(columns={prod_map["publish"]: "Publish Time"}, inplace=True)

    # Validate and rename GA4 columns
    if ga4_df is None or ga4_map.get("msid") not in ga4_df.columns:
        vc.add("Critical", "MISSING_KEY", "GA4 missing MSID column", want=ga4_map.get("msid"))
        return None, vc
        
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

    # Validate and rename GSC columns
    if gsc_df is None:
        vc.add("Critical", "MISSING_DATA", "GSC data is missing")
        return None, vc
        
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

    # Early date standardization
    prod_df, ga4_df, gsc_df = standardize_dates_early(
        prod_df, ga4_df, gsc_df,
        {"prod": prod_map, "ga4": ga4_map, "gsc": gsc_map}, 
        vc
    )

    # Continue with processing...
    # [Rest of processing logic remains similar but with enhanced error handling]
    
    # Return placeholder for now - full implementation continues in next part
    return pd.DataFrame(), vc

# Run processing (simplified for example)
vc_serialized = rep_df.to_json(orient="records") if not rep_df.empty else "[]"

with st.spinner("Processing & merging with robust validation..."):
    master_df, vc_after = process_uploaded_files(
        prod_df_raw, ga4_df_raw, gsc_df_raw, 
        prod_map, ga4_map, gsc_map,
        vc_serialized=vc_serialized, 
        merge_strategy=MERGE_STRATEGY
    )

# [Rest of the processing and UI code continues...]

# ============================
# PART 4/5: Core Processing, Analysis Modules & Safe Execution
# ============================

# Continue the robust processing function from Part 3
@st.cache_data(show_spinner=False, max_entries=3)
def process_uploaded_files_complete(prod_df_raw, ga4_df_raw, gsc_df_raw, prod_map, ga4_map, gsc_map,
                                   vc_serialized: Optional[str] = None,
                                   merge_strategy: Optional[Dict[str, str]] = None):
    """Complete processing pipeline with enhanced error handling"""
    vc = ValidationCollector()
    
    # Load previous validation messages
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
            if df is not None:
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
        if gsc_df is not None and "msid" not in gsc_df.columns:
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
        if gsc_df is not None:
            if "Impressions" in gsc_df.columns:
                gsc_df["Impressions"] = coerce_numeric(gsc_df["Impressions"], "GSC.Impressions", vc, clamp=(0, float("inf")))
            
            if "CTR" in gsc_df.columns:
                if gsc_df["CTR"].dtype == object:
                    tmp = gsc_df["CTR"].astype(str).str.replace("%", "", regex=False)
                    ctr_val = pd.to_numeric(tmp, errors="coerce")
                    if (ctr_val > 1.0).any():
                        ctr_val = ctr_val / 100.0
                        vc.add("Info", "CTR_SCALE", "CTR parsed as percentage (÷100)")
                    gsc_df["CTR"] = ctr_val
                else:
                    gsc_df["CTR"] = pd.to_numeric(gsc_df["CTR"], errors="coerce")
                
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
        if prod_df is not None and "Path" in prod_df.columns:
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
            if prod_df is not None:
                prod_df["L1_Category"] = "Uncategorized"
                prod_df["L2_Category"] = "Uncategorized"
    except Exception as e:
        vc.add("Warning", "CATEGORY_PARSE_FAIL", f"Category parsing failed: {e}")
        if prod_df is not None:
            prod_df["L1_Category"] = "Uncategorized"
            prod_df["L2_Category"] = "Uncategorized"

    # Merge GSC with Production
    before_counts = {
        "prod": len(prod_df) if prod_df is not None else 0,
        "ga4": len(ga4_df) if ga4_df is not None else 0, 
        "gsc": len(gsc_df) if gsc_df is not None else 0
    }
    
    try:
        if gsc_df is not None and prod_df is not None:
            merged_1 = pd.merge(
                gsc_df,
                prod_df[["msid", "Title", "Path", "Publish Time", "L1_Category", "L2_Category"]].drop_duplicates(),
                on="msid", 
                how=ms.get("gsc_x_prod", "left"), 
                validate="many_to_one"
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
        if ga4_df is not None and "date" in ga4_df.columns:
            numeric_cols = ga4_df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                ga4_daily = ga4_df.groupby(["msid", "date"], as_index=False, dropna=False)[numeric_cols].sum(min_count=1)
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
        master_df = pd.merge(merged_1, ga4_daily, on=["msid", "date"], how=ms.get("ga4_align", "left"))
        vc.checkpoint("merge_ga4", after_master=len(master_df))
    except Exception as e:
        vc.add("Warning", "FINAL_MERGE_FAIL", f"Final merge with GA4 failed: {e}")
        master_df = merged_1.copy()

    # Final data cleaning
    try:
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

# Run the complete processing
with st.spinner("Processing & merging datasets..."):
    master_df, vc_after = process_uploaded_files_complete(
        prod_df_raw, ga4_df_raw, gsc_df_raw, 
        prod_map, ga4_map, gsc_map,
        vc_serialized=vc_serialized, 
        merge_strategy=MERGE_STRATEGY
    )

if master_df is None or master_df.empty:
    st.error("Processing failed. Check validation reports above.")
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

# Master data preview
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

# Sampling for large datasets
if master_df.shape[0] > CONFIG["performance"]["sample_row_limit"]:
    st.info(f"Large dataset detected. Sampling {CONFIG['performance']['sample_row_limit']:,} rows for interactive analysis.")
    master_df = master_df.sample(
        min(CONFIG["performance"]["sample_row_limit"], len(master_df)), 
        random_state=CONFIG["performance"]["seed"]
    )

# Preview data
st.subheader("Data Preview (First 10 rows)")
st.dataframe(master_df.head(10), use_container_width=True, hide_index=True)

# Processing logs
st.markdown("### Processing Logs")
if PROCESS_LOG:
    logs_df = pd.DataFrame(PROCESS_LOG)
    st.dataframe(logs_df.tail(20), use_container_width=True, hide_index=True)
    st.download_button(
        "Download Logs (CSV)", 
        data=logs_df.to_csv(index=False).encode("utf-8"),
        file_name=f"processing_logs_{pd.Timestamp.now().strftime('%Y%m%d')}.csv", 
        mime="text/csv"
    )
else:
    st.caption("No detailed processing logs captured.")

if step == "3) Validate & Process":
    st.success("Data processing complete! Move to **Step 4) Configure & Analyze** to generate insights.")
    st.stop()

# ============================
# Analysis Modules & Safe Execution
# ============================

# Date filtering
def filter_by_date(df, start_date, end_date):
    """Filter dataframe by date range"""
    if df is None or df.empty or "date" not in df.columns:
        return df
        
    df_copy = df.copy()
    try:
        df_copy["date"] = pd.to_datetime(df_copy["date"], errors="coerce").dt.date
        mask = (df_copy["date"] >= start_date) & (df_copy["date"] <= end_date)
        return df_copy[mask].copy()
    except Exception as e:
        st.warning(f"Date filtering failed: {e}")
        return df_copy

# Apply date filter
filtered_df = filter_by_date(master_df, *st.session_state.date_range)
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
        logger.error(f"[ModuleFail] {label}: {e}", exc_info=True)
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

# Engagement vs Search Mismatch Analysis (Enhanced)
def _build_clicks_proxy(df):
    """Build click proxy from available metrics"""
    if df is None:
        return None
        
    # Try Impressions * CTR first
    if "Impressions" in df.columns and "CTR" in df.columns:
        try:
            imp = pd.to_numeric(df["Impressions"], errors="coerce").fillna(0)
            ctr = pd.to_numeric(df["CTR"], errors="coerce").fillna(0)
            # Handle percentage CTR
            if ctr.max() > 1:
                ctr = ctr / 100
            return imp * ctr
        except:
            pass
            
    # Fallback to pageviews
    if "screenPageViews" in df.columns:
        return pd.to_numeric(df["screenPageViews"], errors="coerce").fillna(0)
        
    # Fallback to users
    if "totalUsers" in df.columns:
        return pd.to_numeric(df["totalUsers"], errors="coerce").fillna(0)
        
    return None

def engagement_mismatches(df):
    """Identify engagement vs search performance mismatches"""
    if df is None or df.empty:
        return ["No data available for analysis"]
        
    d = df.copy()
    
    # Resolve available columns
    dur_col = _pick_col(d, ["userEngagementDuration", "engagement_duration", "duration"])
    br_col = _pick_col(d, ["bounceRate", "bounce_rate", "bounce"])
    clicks_col = _pick_col(d, ["Clicks", "clicks", "gsc_clicks"])
    pos_col = _pick_col(d, ["Position", "position", "gsc_avg_position"])
    title_col = _pick_col(d, ["Title", "title", "headline"]) or "Title"
    
    # Check if we have sufficient data
    if not dur_col and not br_col:
        return ["Engagement analysis requires engagement metrics (duration or bounce rate). None found."]
    
    # Build click proxy if needed
    if not clicks_col:
        proxy = _build_clicks_proxy(d)
        if proxy is not None:
            d["__ClicksProxy"] = proxy
            clicks_col = "__ClicksProxy"
    
    # Coerce numeric types
    numeric_cols = [c for c in [dur_col, br_col, clicks_col, pos_col] if c]
    for col in numeric_cols:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    
    # Filter by position if available
    if pos_col:
        d = d[(d[pos_col].isna()) | (d[pos_col].between(1, 50))].copy()
    
    # Calculate engagement score
    engagement_parts = []
    if dur_col:
        engagement_parts.append(d[dur_col].rank(pct=True, na_option='keep'))
    if br_col:
        engagement_parts.append(1 - d[br_col].rank(pct=True, na_option='keep'))
    
    if not engagement_parts:
        return ["No usable engagement signals after data cleaning"]
    
    d["engagement_score"] = np.mean(engagement_parts, axis=0) if len(engagement_parts) > 1 else engagement_parts[0]
    
    # Calculate search score
    search_parts = []
    if clicks_col:
        search_parts.append(d[clicks_col].rank(pct=True, na_option='keep'))
    if pos_col:
        search_parts.append(1 - d[pos_col].rank(pct=True, na_option='keep'))
    
    if not search_parts:
        return ["No usable search signals after data cleaning"]
    
    d["search_score"] = np.mean(search_parts, axis=0) if len(search_parts) > 1 else search_parts[0]
    
    # Calculate mismatch and classify
    d["mismatch_score"] = d["engagement_score"] - d["search_score"]
    d["mismatch_type"] = np.where(
        (d["engagement_score"] > 0.7) & (d["search_score"] < 0.3), "Hidden Gem",
        np.where((d["search_score"] > 0.7) & (d["engagement_score"] < 0.3), "Clickbait Risk", None)
    )
    
    # Get top mismatches
    mismatches = (
        d.dropna(subset=["mismatch_type"])
         .sort_values(by="mismatch_score", key=lambda x: np.abs(x), ascending=False)
         .head(10)
    )
    
    if mismatches.empty:
        return ["No significant engagement-search mismatches detected in the data."]
    
    # Generate recommendation cards
    cards = []
    for _, row in mismatches.iterrows():
        emoji = "💎" if row["mismatch_type"] == "Hidden Gem" else "⚠️"
        title_text = str(row.get(title_col, "Untitled"))[:80] + "..." if len(str(row.get(title_col, "Untitled"))) > 80 else str(row.get(title_col, "Untitled"))
        
        if row["mismatch_type"] == "Hidden Gem":
            recommendations = [
                "**SEO Optimization:** Expand title/H1 & better match search intent",
                "**Internal Linking:** Add links from high-authority pages",
                "**Content Expansion:** Add related sections and depth"
            ]
            goal = f"*Goal: Leverage high engagement to improve search visibility{' from position ' + str(round(row[pos_col], 1)) if pos_col and not pd.isna(row.get(pos_col)) else ''}*"
        else:
            recommendations = [
                "**Content Depth:** Address thin content issues",
                "**UX Improvement:** Enhance page speed & mobile experience", 
                "**Title Alignment:** Ensure title promises match content"
            ]
            goal = f"*Goal: Improve user experience to match search performance*"
        
        # Create expandable details
        with st.expander(f"{emoji} {row['mismatch_type']}: {title_text}", expanded=False):
            st.write(f"**Engagement Score:** {row['engagement_score']:.3f} | **Search Score:** {row['search_score']:.3f}")
            if dur_col and not pd.isna(row.get(dur_col)):
                st.write(f"**Avg Duration:** {row[dur_col]:.1f}s")
            if br_col and not pd.isna(row.get(br_col)):
                st.write(f"**Bounce Rate:** {row[br_col]:.1%}")
            if pos_col and not pd.isna(row.get(pos_col)):
                st.write(f"**Avg Position:** {row[pos_col]:.1f}")
            
            confidence = min(0.95, 0.5 + abs(row['mismatch_score']) * 0.8)
            st.write(f"**Confidence:** {confidence:.1%}")
        
        # Build card content
        card_content = f"""
### {emoji} {row['mismatch_type']}
**MSID:** `{row.get('msid', 'N/A')}`  
**Title:** {title_text}  
**Engagement Score:** **{row['engagement_score']:.3f}** | **Search Score:** **{row['search_score']:.3f}**  

**Recommendations:**  
{"  \n".join(recommendations)}  

{goal}
"""
        cards.append(card_content)
    
    return cards

def scatter_engagement_vs_search(df):
    """Create engagement vs search scatter plot"""
    if df is None or df.empty:
        st.info("No data available for scatter plot")
        return
        
    # Resolve columns
    dur_col = _pick_col(df, ["userEngagementDuration", "engagement_duration"])
    br_col = _pick_col(df, ["bounceRate", "bounce_rate"]) 
    clicks_col = _pick_col(df, ["Clicks", "clicks"])
    pos_col = _pick_col(df, ["Position", "position"])
    
    if not (dur_col or br_col):
        st.info("Engagement vs Search analysis requires engagement metrics")
        return
        
    if not (clicks_col or pos_col):
        st.info("Engagement vs Search analysis requires search metrics") 
        return
        
    t = df.copy()
    
    # Prepare data for visualization
    numeric_cols = [c for c in [dur_col, br_col, clicks_col, pos_col] if c]
    for col in numeric_cols:
        t[col] = pd.to_numeric(t[col], errors="coerce")
    
    # Build click proxy if needed
    if not clicks_col:
        proxy = _build_clicks_proxy(t)
        if proxy is not None:
            t["__ClicksProxy"] = proxy
            clicks_col = "__ClicksProxy"
    
    # Calculate scores
    engagement_parts = []
    if dur_col:
        engagement_parts.append(t[dur_col].rank(pct=True, na_option='keep'))
    if br_col:
        engagement_parts.append(1 - t[br_col].rank(pct=True, na_option='keep'))
    
    if engagement_parts:
        t["engagement_score"] = np.mean(engagement_parts, axis=0) if len(engagement_parts) > 1 else engagement_parts[0]
    
    search_parts = []
    if clicks_col:
        search_parts.append(t[clicks_col].rank(pct=True, na_option='keep'))
    if pos_col:
        search_parts.append(1 - t[pos_col].rank(pct=True, na_option='keep'))
    
    if search_parts:
        t["search_score"] = np.mean(search_parts, axis=0) if len(search_parts) > 1 else search_parts[0]
    
    # Aggregate for plotting
    agg_cols = ["msid", "Title", "L2_Category"]
    agg_cols = [c for c in agg_cols if c in t.columns]
    
    if not agg_cols:
        st.info("Insufficient data for aggregation")
        return
        
    agg_data = t.groupby(agg_cols, as_index=False).agg({
        "engagement_score": "mean",
        "search_score": "mean",
        **({pos_col: "mean"} if pos_col else {}),
        **({clicks_col: "sum"} if clicks_col else {})
    })
    
    if _HAS_PLOTLY and not agg_data.empty:
        try:
            fig = px.scatter(
                agg_data, 
                x="engagement_score", 
                y="search_score",
                size=clicks_col if clicks_col else None,
                color="L2_Category" if "L2_Category" in agg_data.columns else None,
                hover_data=["msid", "Title"] + ([pos_col, clicks_col] if pos_col and clicks_col else []),
                title="Engagement vs Search Performance",
                labels={
                    "engagement_score": "Engagement Score (Percentile)",
                    "search_score": "Search Score (Percentile)" 
                }
            )
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")
            export_plot_html(fig, "engagement_vs_search")
        except Exception as e:
            st.error(f"Failed to create scatter plot: {e}")
    else:
        # Fallback to simple chart
        st.scatter_chart(
            agg_data, 
            x="engagement_score", 
            y="search_score",
            size=clicks_col if clicks_col else None
        )
# ============================
# PART 5/5: Category Analysis, Trends, Forecasting & UI
# ============================

# Category Performance Analysis
def category_heatmap(df, value_col, title):
    """Create category performance heatmap"""
    if not _HAS_PLOTLY or df is None or df.empty:
        st.info("Heatmap visualization requires Plotly and data")
        return
        
    t = df.copy()
    if "L1_Category" not in t.columns:
        t["L1_Category"] = "Uncategorized"
    if "L2_Category" not in t.columns:
        t["L2_Category"] = "Uncategorized"
    if value_col not in t.columns:
        st.info(f"Column '{value_col}' not found for heatmap")
        return
        
    try:
        # Aggregate data
        agg_data = t.groupby(["L1_Category", "L2_Category"]).agg(
            value=(value_col, "sum")
        ).reset_index()
        
        if agg_data.empty:
            st.info("No data available for heatmap after aggregation")
            return
            
        fig = px.density_heatmap(
            agg_data, 
            x="L1_Category", 
            y="L2_Category", 
            z="value", 
            color_continuous_scale="Viridis",
            title=title,
            labels={"value": value_col}
        )
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")
        export_plot_html(fig, f"heatmap_{value_col}")
        
    except Exception as e:
        st.error(f"Failed to create heatmap: {e}")

def analyze_category_performance(df):
    """Analyze category-level performance"""
    if df is None or df.empty:
        return pd.DataFrame()
        
    d = df.copy()
    
    # Ensure category columns exist
    if "L1_Category" not in d.columns:
        d["L1_Category"] = "Uncategorized"
    if "L2_Category" not in d.columns:
        d["L2_Category"] = "Uncategorized"
    if "msid" not in d.columns:
        d["msid"] = range(len(d))
    
    # Identify available metrics
    engagement_col = _pick_col(d, ["userEngagementDuration", "engagement_duration"])
    pageviews_col = _pick_col(d, ["screenPageViews", "pageviews"])
    users_col = _pick_col(d, ["totalUsers", "users"])
    clicks_col = _pick_col(d, ["Clicks", "clicks"])
    
    if not any([engagement_col, pageviews_col, users_col, clicks_col]):
        return pd.DataFrame()
    
    # Prepare numeric columns
    numeric_cols = [c for c in [engagement_col, pageviews_col, users_col, clicks_col] if c]
    for col in numeric_cols:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    
    # Build aggregation dictionary
    agg_dict = {"msid": "nunique"}
    if pageviews_col:
        agg_dict[pageviews_col] = "sum"
    elif users_col:
        agg_dict[users_col] = "sum"
    if engagement_col:
        agg_dict[engagement_col] = "mean"
    if clicks_col:
        agg_dict[clicks_col] = "sum"
    
    # Group by categories
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
            
        grouped = grouped.rename(columns=column_rename)
        
        # Calculate performance indices
        if "total_traffic" in grouped.columns:
            site_avg_traffic = grouped["total_traffic"].mean()
            if site_avg_traffic > 0:
                grouped["traffic_index"] = grouped["total_traffic"] / site_avg_traffic
            else:
                grouped["traffic_index"] = 0
                
        if "avg_engagement_duration" in grouped.columns:
            site_avg_eng = grouped["avg_engagement_duration"].mean()
            if site_avg_eng > 0:
                grouped["engagement_index"] = grouped["avg_engagement_duration"] / site_avg_eng
            else:
                grouped["engagement_index"] = 0
        
        # Classify quadrants
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
            
        return grouped
        
    except Exception as e:
        st.error(f"Category analysis failed: {e}")
        return pd.DataFrame()

# Trends & Forecasting
def forecast_series(daily_series, periods=14):
    """Generate time series forecasts"""
    if daily_series is None or len(daily_series) < 7:
        return None
        
    try:
        # Ensure daily frequency
        daily_series = daily_series.asfreq("D").fillna(method="ffill").fillna(0)
        
        if _HAS_STM and len(daily_series) >= 14:
            try:
                # Use Holt-Winters exponential smoothing
                model = ExponentialSmoothing(
                    daily_series, 
                    trend="add", 
                    seasonal="add", 
                    seasonal_periods=min(7, len(daily_series))
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
                logger.info(f"Holt-Winters forecast failed, using fallback: {e}")
        
        # Fallback: simple moving average
        moving_avg = daily_series.rolling(window=7, min_periods=1).mean()
        last_value = moving_avg.iloc[-1] if not moving_avg.empty else daily_series.mean()
        
        future_dates = pd.date_range(
            start=daily_series.index.max() + pd.Timedelta(days=1), 
            periods=periods, 
            freq="D"
        )
        
        forecast_values = [last_value] * periods
        
        return pd.DataFrame({
            "date": future_dates,
            "forecast": forecast_values,
            "low": [v * 0.8 for v in forecast_values],  # 20% lower bound
            "high": [v * 1.2 for v in forecast_values]  # 20% upper bound
        })
        
    except Exception as e:
        logger.error(f"Forecasting failed: {e}")
        return None

def time_series_trends(df, metric_col, title):
    """Create time series trend visualization"""
    if df is None or df.empty or "date" not in df.columns:
        st.info("Time series analysis requires date column and data")
        return
        
    t = df.copy()
    t["date"] = pd.to_datetime(t["date"], errors="coerce")
    t = t.dropna(subset=["date"])
    
    if t.empty or metric_col not in t.columns:
        st.info(f"No valid data for {metric_col} time series")
        return
        
    # Prepare metric
    t[metric_col] = pd.to_numeric(t[metric_col], errors="coerce").fillna(0)
    
    # Aggregate by date
    daily_data = t.groupby("date")[metric_col].sum().reset_index()
    
    if _HAS_PLOTLY:
        try:
            fig = px.line(
                daily_data, 
                x="date", 
                y=metric_col, 
                title=title,
                labels={metric_col: metric_col.replace("_", " ").title()}
            )
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")
            export_plot_html(fig, f"timeseries_{metric_col}")
        except Exception as e:
            st.error(f"Failed to create time series chart: {e}")
    else:
        # Fallback to Streamlit native chart
        st.line_chart(daily_data.set_index("date")[metric_col])

# ============================
# MAIN ANALYSIS UI
# ============================

st.header("📊 Advanced Analytics & Insights")

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

# Module 5: Category Performance
st.subheader("Module 5: Category Performance Analysis")
st.caption("Understand how different content categories perform across traffic and engagement metrics")

category_results = run_module_safely(
    "Category Performance", 
    analyze_category_performance, 
    filtered_df
)

if isinstance(category_results, pd.DataFrame) and not category_results.empty:
    st.dataframe(category_results, use_container_width=True, hide_index=True)
    
    # Category visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Traffic heatmap
        traffic_col = _pick_col(filtered_df, ["screenPageViews", "totalUsers", "Clicks"])
        if traffic_col:
            st.subheader("Category Traffic Distribution")
            run_module_safely(
                "Traffic Heatmap", 
                category_heatmap, 
                filtered_df, traffic_col, "Category Traffic Heatmap"
            )
    
    with col2:
        # Top categories bar chart
        if "total_gsc_clicks" in category_results.columns:
            st.subheader("Top Categories by GSC Clicks")
            top_categories = category_results.nlargest(10, "total_gsc_clicks")
            if not top_categories.empty:
                st.bar_chart(top_categories.set_index("L2_Category")["total_gsc_clicks"])
else:
    st.info("Category performance analysis requires sufficient category and metric data")

st.divider()

# Module 6: Trends & Forecasting
st.subheader("Module 6: Trends & Forecasting")
st.caption("Analyze historical trends and generate performance forecasts")

if "date" in filtered_df.columns and not filtered_df.empty:
    # Prepare time series data
    ts_data = filtered_df.copy()
    ts_data["date"] = pd.to_datetime(ts_data["date"], errors="coerce")
    ts_data = ts_data.dropna(subset=["date"])
    
    if not ts_data.empty:
        # Select primary metric for forecasting
        primary_metric = _pick_col(ts_data, ["totalUsers", "screenPageViews", "Clicks", "Impressions"])
        
        if primary_metric:
            # Daily aggregation
            daily_series = ts_data.groupby("date")[primary_metric].sum().sort_index()
            
            if len(daily_series) >= 7:  # Need at least 7 days for meaningful analysis
                # Generate forecast
                forecast_data = forecast_series(daily_series, periods=14)
                
                if forecast_data is not None:
                    # Create forecast visualization
                    if _HAS_PLOTLY:
                        fig = go.Figure()
                        
                        # Historical data
                        fig.add_trace(go.Scatter(
                            x=daily_series.index,
                            y=daily_series.values,
                            name="Historical",
                            line=dict(color="blue", width=2)
                        ))
                        
                        # Forecast
                        fig.add_trace(go.Scatter(
                            x=forecast_data["date"],
                            y=forecast_data["forecast"],
                            name="Forecast",
                            line=dict(color="red", width=2, dash="dash")
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
                            title=f"{primary_metric} - 14-Day Forecast",
                            xaxis_title="Date",
                            yaxis_title=primary_metric,
                            hovermode="x unified"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True, theme="streamlit")
                        export_plot_html(fig, f"forecast_{primary_metric}")
                    
                    # Show forecast table
                    with st.expander("Forecast Details", expanded=False):
                        st.dataframe(forecast_data, use_container_width=True, hide_index=True)
                
                # Additional time series trends
                st.subheader("Additional Metric Trends")
                secondary_metrics = [
                    _pick_col(ts_data, ["Impressions", "CTR", "Position"]),
                    _pick_col(ts_data, ["userEngagementDuration", "bounceRate"])
                ]
                secondary_metrics = [m for m in secondary_metrics if m and m != primary_metric]
                
                for metric in secondary_metrics[:2]:  # Show max 2 additional charts
                    if metric:
                        run_module_safely(
                            f"Trend {metric}", 
                            time_series_trends, 
                            ts_data, metric, f"{metric} Over Time"
                        )
            else:
                st.info(f"Need at least 7 days of {primary_metric} data for forecasting")
        else:
            st.info("No suitable metrics found for trend analysis")
    else:
        st.info("No valid date data available for trend analysis")
else:
    st.info("Date column required for trend analysis and forecasting")

# ============================
# EXPORT & SUMMARY
# ============================

st.divider()
st.subheader("📤 Export Results")

# Create export options
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
    # Summary statistics
    with st.expander("Analysis Summary"):
        if not filtered_df.empty:
            st.metric("Total Articles Analyzed", f"{filtered_df['msid'].nunique():,}")
            if "Clicks" in filtered_df.columns:
                st.metric("Total GSC Clicks", f"{filtered_df['Clicks'].sum():,}")
            if "screenPageViews" in filtered_df.columns:
                st.metric("Total Pageviews", f"{filtered_df['screenPageViews'].sum():,}")

# Final recommendations
st.divider()
st.subheader("🎯 Key Recommendations")

if isinstance(engagement_cards, list) and len(engagement_cards) > 0:
    st.success("**Priority Actions:**")
    
    hidden_gems = sum(1 for card in engagement_cards if "Hidden Gem" in card)
    clickbait_risks = sum(1 for card in engagement_cards if "Clickbait Risk" in card)
    
    if hidden_gems > 0:
        st.info(f"• **Optimize {hidden_gems} Hidden Gem(s):** Improve SEO for high-engagement content")
    if clickbait_risks > 0:
        st.warning(f"• **Fix {clickbait_risks} Clickbait Risk(s):** Enhance content quality and UX")
    
    if isinstance(category_results, pd.DataFrame):
        stars = category_results[category_results["quadrant"] == "Stars"]
        workhorses = category_results[category_results["quadrant"] == "Workhorses"]
        
        if not stars.empty:
            st.success(f"• **Leverage {len(stars)} Star Category(ies):** Scale successful content patterns")
        if not workhorses.empty:
            st.info(f"• **Optimize {len(workhorses)} Workhorse Category(ies):** Improve engagement on high-traffic content")
else:
    st.info("Upload and process your data to get personalized recommendations")

# Footer
st.markdown("---")
st.caption("GrowthOracle AI v2.0 | Advanced SEO & Content Intelligence Platform")

