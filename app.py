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
        # st.warning(f"No data to download for {label}") # Can be noisy
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
], horizontal=True, key="main_step")


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
    with col2:
        ga4_file = st.file_uploader("GA4 Data (CSV)", type=["csv"], key="ga4_csv")
    with col3:
        gsc_file = st.file_uploader("GSC Data (CSV)", type=["csv"], key="gsc_csv")

if not all([prod_file, ga4_file, gsc_file]):
    st.warning("Please upload all three CSV files to proceed.")
    st.stop()

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

def _guess_colmap(df, keywords):
    for key, alts in keywords.items():
        # Exact match first
        for alt in alts:
            if alt in df.columns:
                return alt
        # Then case-insensitive
        for alt in alts:
            for col in df.columns:
                if alt.lower() == col.lower():
                    return col
        # Then partial match
        for alt in alts:
            for col in df.columns:
                if alt.lower() in col.lower():
                    return col
    return None

def guess_colmap_enhanced(prod_df, ga4_df, gsc_df):
    prod_keywords = {"msid": ["msid"], "title": ["title"], "path": ["path", "url"], "publish": ["publish"]}
    ga4_keywords = {"msid": ["msid"], "date": ["date"], "pageviews": ["pageview"], "users": ["user"], "engagement": ["engagement"], "bounce": ["bounce"]}
    gsc_keywords = {"date": ["date"], "page": ["page"], "query": ["query"], "clicks": ["clicks"], "impr": ["impression"], "ctr": ["ctr"], "pos": ["position"]}
    
    prod_map = {k: _guess_colmap(prod_df, {k: v}) for k, v in prod_keywords.items()}
    ga4_map = {k: _guess_colmap(ga4_df, {k: v}) for k, v in ga4_keywords.items()}
    gsc_map = {k: _guess_colmap(gsc_df, {k: v}) for k, v in gsc_keywords.items()}
    return prod_map, ga4_map, gsc_map

def validate_columns_presence(prod_map, ga4_map, gsc_map, vc: ValidationCollector):
    req = {"Production": ["msid"], "GA4": ["msid", "date"], "GSC": ["date", "page", "query", "clicks", "impr", "pos"]}
    missing = []
    for source, keys in req.items():
        current_map = {"Production": prod_map, "GA4": ga4_map, "GSC": gsc_map}[source]
        for k in keys:
            if not current_map.get(k):
                missing.append(f"{source}: {k}")
    if missing:
        vc.add("Critical", "MISSING_COLMAP", "Missing required column mappings", items=missing)
    return not missing


if step == "2) Upload & Map Columns":
    prod_map_guess, ga4_map_guess, gsc_map_guess = guess_colmap_enhanced(prod_df_raw, ga4_df_raw, gsc_df_raw)
    st.subheader("Column Mapping")
    st.caption("We guessed likely columns. Adjust if needed.")

    with st.expander("Production Mapping", expanded=True):
        prod_map = {
            "msid": st.selectbox("MSID", prod_df_raw.columns, index=list(prod_df_raw.columns).index(prod_map_guess["msid"]) if prod_map_guess.get("msid") else 0),
            "title": st.selectbox("Title", prod_df_raw.columns, index=list(prod_df_raw.columns).index(prod_map_guess["title"]) if prod_map_guess.get("title") else 1),
            "path": st.selectbox("Path", prod_df_raw.columns, index=list(prod_df_raw.columns).index(prod_map_guess["path"]) if prod_map_guess.get("path") else 2),
            "publish": st.selectbox("Publish Time", prod_df_raw.columns, index=list(prod_df_raw.columns).index(prod_map_guess["publish"]) if prod_map_guess.get("publish") else 3),
        }
    with st.expander("GA4 Mapping", expanded=True):
        ga4_map = {
            "msid": st.selectbox("MSID (GA4)", ga4_df_raw.columns, index=list(ga4_df_raw.columns).index(ga4_map_guess["msid"]) if ga4_map_guess.get("msid") else 0),
            "date": st.selectbox("Date (GA4)", ga4_df_raw.columns, index=list(ga4_df_raw.columns).index(ga4_map_guess["date"]) if ga4_map_guess.get("date") else 1),
            "pageviews": st.selectbox("Pageviews", ga4_df_raw.columns, index=list(ga4_df_raw.columns).index(ga4_map_guess["pageviews"]) if ga4_map_guess.get("pageviews") else 2),
            "users": st.selectbox("Users", ga4_df_raw.columns, index=list(ga4_df_raw.columns).index(ga4_map_guess["users"]) if ga4_map_guess.get("users") else 3),
            "engagement": st.selectbox("Engagement", ga4_df_raw.columns, index=list(ga4_df_raw.columns).index(ga4_map_guess["engagement"]) if ga4_map_guess.get("engagement") else 4),
            "bounce": st.selectbox("Bounce Rate", ga4_df_raw.columns, index=list(ga4_df_raw.columns).index(ga4_map_guess["bounce"]) if ga4_map_guess.get("bounce") else 5),
        }
    with st.expander("GSC Mapping", expanded=True):
        gsc_map = {
            "date": st.selectbox("Date (GSC)", gsc_df_raw.columns, index=list(gsc_df_raw.columns).index(gsc_map_guess["date"]) if gsc_map_guess.get("date") else 0),
            "page": st.selectbox("Page URL", gsc_df_raw.columns, index=list(gsc_df_raw.columns).index(gsc_map_guess["page"]) if gsc_map_guess.get("page") else 1),
            "query": st.selectbox("Query", gsc_df_raw.columns, index=list(gsc_df_raw.columns).index(gsc_map_guess["query"]) if gsc_map_guess.get("query") else 2),
            "clicks": st.selectbox("Clicks", gsc_df_raw.columns, index=list(gsc_df_raw.columns).index(gsc_map_guess["clicks"]) if gsc_map_guess.get("clicks") else 3),
            "impr": st.selectbox("Impressions", gsc_df_raw.columns, index=list(gsc_df_raw.columns).index(gsc_map_guess["impr"]) if gsc_map_guess.get("impr") else 4),
            "ctr": st.selectbox("CTR", gsc_df_raw.columns, index=list(gsc_df_raw.columns).index(gsc_map_guess["ctr"]) if gsc_map_guess.get("ctr") else 5),
            "pos": st.selectbox("Position", gsc_df_raw.columns, index=list(gsc_df_raw.columns).index(gsc_map_guess["pos"]) if gsc_map_guess.get("pos") else 6),
        }

    vc_map = ValidationCollector()
    if validate_columns_presence(prod_map, ga4_map, gsc_map, vc_map):
        st.session_state.mapping = {"prod": prod_map, "ga4": ga4_map, "gsc": gsc_map}
        st.success("Column mapping saved. Proceed to **Step 3**.")
    else:
        st.error("Critical mapping issues detected. Please fix before proceeding.")
        st.dataframe(vc_map.to_dataframe())
    st.stop()


if "mapping" not in st.session_state:
    st.warning("Please complete **Step 2** (column mapping) first.")
    st.stop()

prod_map = st.session_state.mapping["prod"]
ga4_map = st.session_state.mapping["ga4"]
gsc_map = st.session_state.mapping["gsc"]

@st.cache_data(show_spinner="Processing and merging datasets...")
def process_uploaded_files_complete(_prod_df_raw, _ga4_df_raw, _gsc_df_raw, prod_map, ga4_map, gsc_map, merge_strategy):
    vc = ValidationCollector()
    prod_df = _prod_df_raw.copy()
    ga4_df = _ga4_df_raw.copy()
    gsc_df = _gsc_df_raw.copy()

    # Rename
    std_names = { "prod": {"msid": "msid", "title": "Title", "path": "Path", "publish": "PublishTime"}, "ga4": {"msid": "msid", "date": "date", "pageviews": "screenPageViews", "users": "totalUsers", "engagement": "userEngagementDuration", "bounce": "bounceRate"}, "gsc": {"date": "date", "page": "page_url", "query": "Query", "clicks": "Clicks", "impr": "Impressions", "ctr": "CTR", "pos": "Position"} }
    try:
        prod_df.rename(columns={prod_map[k]: v for k, v in std_names["prod"].items() if prod_map.get(k)}, inplace=True)
        ga4_df.rename(columns={ga4_map[k]: v for k, v in std_names["ga4"].items() if ga4_map.get(k)}, inplace=True)
        gsc_df.rename(columns={gsc_map[k]: v for k, v in std_names["gsc"].items() if gsc_map.get(k)}, inplace=True)
    except Exception as e:
        vc.add("Critical", "RENAME_FAIL", f"Failed column renaming: {e}")
        return None, vc

    # Standardize Dates & MSIDs
    for df, name in [(ga4_df, "GA4"), (gsc_df, "GSC")]:
        if "date" in df.columns: df["date"] = safe_dt_parse(df["date"], f"{name}.date", vc)
    if "msid" not in gsc_df.columns and "page_url" in gsc_df.columns:
        gsc_df["msid"] = gsc_df["page_url"].str.extract(r'(\d+)\.cms').iloc[:, 0]
    for df, name in [(prod_df, "Prod"), (ga4_df, "GA4"), (gsc_df, "GSC")]:
        if "msid" in df.columns:
            df["msid"] = coerce_numeric(df["msid"], f"{name}.msid", vc)
            df.dropna(subset=["msid"], inplace=True)
            if not df.empty: df["msid"] = df["msid"].astype("int64")

    # Coerce Numerics
    for col in ["screenPageViews", "totalUsers", "userEngagementDuration", "bounceRate"]:
        if col in ga4_df.columns: ga4_df[col] = coerce_numeric(ga4_df[col], f"GA4.{col}", vc)
    for col in ["Clicks", "Impressions", "CTR", "Position"]:
        if col in gsc_df.columns: gsc_df[col] = coerce_numeric(gsc_df[col], f"GSC.{col}", vc)

    # Merge
    if gsc_df is None or prod_df is None or gsc_df.empty or prod_df.empty:
        vc.add("Critical", "MERGE_FAIL", "Cannot merge due to missing GSC or Production data.")
        return None, vc
    
    prod_cols = [c for c in ["msid", "Title", "Path", "PublishTime"] if c in prod_df.columns]
    merged_1 = pd.merge(gsc_df, prod_df[prod_cols].drop_duplicates(subset=["msid"]), on="msid", how=merge_strategy["gsc_x_prod"])
    
    if ga4_df is not None and not ga4_df.empty and "date" in ga4_df.columns:
        numeric_cols = [c for c in ["screenPageViews", "totalUsers", "userEngagementDuration", "bounceRate"] if c in ga4_df.columns]
        ga4_daily = ga4_df.groupby(["msid", pd.Grouper(key="date", freq="D")]).agg({k: "sum" for k in numeric_cols}).reset_index()
        master_df = pd.merge(merged_1, ga4_daily, on=["msid", "date"], how=merge_strategy["ga4_align"])
    else:
        master_df = merged_1
        vc.add("Info", "NO_GA4_MERGE", "GA4 data not available for merge")

    # Final Cleaning
    if "Path" in master_df.columns:
        cats = master_df["Path"].str.strip('/').str.split('/', n=1, expand=True)
        master_df["L1_Category"] = cats[0].fillna("Uncategorized")
        master_df["L2_Category"] = cats[1].fillna("General")

    return master_df, vc

master_df, vc_after = process_uploaded_files_complete(prod_df_raw, ga4_df_raw, gsc_df_raw, prod_map, ga4_map, gsc_map, MERGE_STRATEGY)

if master_df is None or master_df.empty:
    st.error("Data processing failed critically.")
    st.dataframe(vc_after.to_dataframe())
    st.stop()

st.success(f"✅ Master dataset created: {master_df.shape[0]:,} rows × {master_df.shape[1]} columns")

if step != "4) Configure & Analyze":
    st.subheader("Data Preview")
    st.dataframe(master_df.head())
    st.info("Processing complete. Proceed to the 'Configure & Analyze' step to view insights.")
    st.stop()
    # ============================
# PART 4/5: Core Analysis Modules (MODIFIED)
# ============================

# Date filtering
start_date, end_date = pd.to_datetime(st.session_state.date_range[0]), pd.to_datetime(st.session_state.date_range[1])
master_df['date'] = pd.to_datetime(master_df['date'], utc=True)
filtered_df = master_df[(master_df['date'] >= start_date) & (master_df['date'] <= end_date)].copy()
st.info(f"Analyzing {len(filtered_df):,} rows from {st.session_state.date_range[0]} to {st.session_state.date_range[1]}")


def run_module_safely(label: str, fn, *args, **kwargs):
    try:
        with st.spinner(f"Running {label}..."):
            return fn(*args, **kwargs)
    except Exception as e:
        st.warning(f"Module '{label}' failed: {e}")
        logger.error(f"Module fail: {label}", exc_info=True)
        return None

# NEW: Function to generate a downloadable opportunities file
def identify_opportunities(df):
    """Identifies various opportunities and returns them as a single DataFrame."""
    opportunities = []
    df_agg = df.groupby('msid').agg({
        'Position': 'mean', 'CTR': 'mean', 'userEngagementDuration': 'mean',
        'bounceRate': 'mean', 'Title': 'first', 'Clicks': 'sum', 'Impressions': 'sum'
    }).reset_index()

    # Hidden Gems: High engagement, poor position
    if 'userEngagementDuration' in df_agg.columns:
        gems = df_agg[(df_agg['userEngagementDuration'] > df_agg['userEngagementDuration'].quantile(0.75)) & (df_agg['Position'] > 15)].copy()
        if not gems.empty:
            gems['opportunity_type'] = 'Hidden Gem (High Engagement, Poor Position)'
            opportunities.append(gems)

    # Low CTR at Good Position
    low_ctr = df_agg[(df_agg['Position'] <= 10) & (df_agg['CTR'] < 0.03) & (df_agg['Impressions'] > 100)].copy()
    if not low_ctr.empty:
        low_ctr['opportunity_type'] = 'Low CTR at Good Position'
        opportunities.append(low_ctr)

    # High Bounce Rate at Good Position
    if 'bounceRate' in df_agg.columns:
        high_bounce = df_agg[(df_agg['Position'] <= 15) & (df_agg['bounceRate'] > 0.7)].copy()
        if not high_bounce.empty:
            high_bounce['opportunity_type'] = 'High Bounce Rate at Good Position'
            opportunities.append(high_bounce)

    if not opportunities:
        return pd.DataFrame()
    return pd.concat(opportunities).drop_duplicates(subset=['msid'])

# MODIFIED: Reworked scatter plot to be cleaner and more insightful
def scatter_engagement_vs_search(df):
    """Create a cleaner scatter plot with highlighted performance quadrants."""
    eng_col = _pick_col(df, ["userEngagementDuration", "totalUsers"])
    pos_col = "Position"
    size_col = _pick_col(df, ["Clicks", "screenPageViews"])

    if not all([eng_col, pos_col, size_col]):
        st.info("Insufficient data for Engagement vs. Search scatter plot.")
        return

    plot_data = df.groupby('msid').agg({
        pos_col: 'mean', eng_col: 'mean', size_col: 'sum', 'L1_Category': 'first', 'Title': 'first'
    }).dropna().reset_index()

    if plot_data.empty: return

    median_eng = plot_data[eng_col].median()
    median_pos = plot_data[pos_col].median()

    def assign_quadrant(row):
        high_eng = row[eng_col] >= median_eng
        good_pos = row[pos_col] <= median_pos
        if high_eng and good_pos: return "Stars"
        if high_eng and not good_pos: return "Hidden Gems"
        if not high_eng and good_pos: return "Workhorses"
        return "Underperformers"

    plot_data['quadrant'] = plot_data.apply(assign_quadrant, axis=1)

    fig = px.scatter(
        plot_data, x=pos_col, y=eng_col, size=size_col, color="quadrant",
        hover_data=["Title"], title="Engagement vs. Search Performance Quadrants",
        labels={pos_col: "Avg. Search Position (Lower is Better)", eng_col: f"Avg. {eng_col} (Higher is Better)"},
        category_orders={"quadrant": ["Stars", "Hidden Gems", "Workhorses", "Underperformers"]},
        color_discrete_map={"Stars": "green", "Hidden Gems": "orange", "Workhorses": "blue", "Underperformers": "rgba(128,128,128,0.5)"},
        size_max=60
    )
    fig.add_vline(x=median_pos, line_dash="dash", line_color="grey", annotation_text="Median Position")
    fig.add_hline(y=median_eng, line_dash="dash", line_color="grey", annotation_text="Median Engagement")
    fig.update_layout(xaxis_autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)

def _pick_col(df, candidates):
    for c in candidates:
        if c in df.columns: return c
    return None

def forecast_series(daily_series, periods=14):
    if daily_series is None or len(daily_series) < 14 or not _HAS_STM: return None
    try:
        model = ExponentialSmoothing(daily_series, trend="add", seasonal="add", seasonal_periods=7).fit()
        forecast = model.forecast(periods)
        # Simple confidence interval
        std_err = np.std(model.resid) * np.sqrt(np.arange(1, periods + 1))
        return pd.DataFrame({"date": forecast.index, "forecast": forecast, "low": forecast - 1.96 * std_err, "high": forecast + 1.96 * std_err})
    except Exception as e:
        logger.warning(f"Forecasting failed: {e}")
        return None

# ============================
# MAIN ANALYSIS UI
# ============================
st.header("📊 Advanced Analytics & Insights")

st.subheader("Engagement vs. Search Performance Mismatch")
st.caption("Identify content that over- or under-performs relative to its visibility.")

# NEW: Generate and offer download for the opportunities file
opportunities_df = identify_opportunities(filtered_df)
if not opportunities_df.empty:
    st.success(f"Identified {len(opportunities_df)} articles with clear opportunities.")
    download_df_button(opportunities_df, "growthoracle_opportunities.csv", "Download Opportunities CSV")
else:
    st.info("No specific engagement mismatches found based on current criteria.")

st.subheader("Engagement vs. Search Scatter Analysis")
run_module_safely("Scatter Plot", scatter_engagement_vs_search, filtered_df)

st.divider()
# ============================
# PART 5/5: Complete Analysis & Export (MODIFIED)
# ============================

# MODIFIED: Enhanced category analysis function
def analyze_category_performance(df):
    """Analyze category performance with additional metrics."""
    if df is None or df.empty: return pd.DataFrame()

    agg_dict = {
        "msid": pd.NamedAgg(column="msid", aggfunc="nunique"),
        "Clicks": pd.NamedAgg(column="Clicks", aggfunc="sum"),
        "Impressions": pd.NamedAgg(column="Impressions", aggfunc="sum"),
    }
    # Add optional metrics if they exist
    if "totalUsers" in df.columns: agg_dict["totalUsers"] = pd.NamedAgg(column="totalUsers", aggfunc="sum")
    if "bounceRate" in df.columns: agg_dict["bounceRate"] = pd.NamedAgg(column="bounceRate", aggfunc="mean")
    if "Position" in df.columns: agg_dict["Position"] = pd.NamedAgg(column="Position", aggfunc="mean")

    grouped = df.groupby(["L1_Category", "L2_Category"]).agg(**agg_dict).reset_index()
    rename_map = {
        "msid": "unique_articles", "Clicks": "total_clicks", "Impressions": "total_impressions",
        "totalUsers": "total_users", "bounceRate": "avg_bounce_rate", "Position": "avg_position"
    }
    grouped = grouped.rename(columns=rename_map)
    return grouped.sort_values("total_clicks", ascending=False)

# MODIFIED: New, visually improved heatmap
def category_heatmap(df, value_col, title):
    """Create a visually improved heatmap with a log scale and text labels."""
    if df is None or df.empty or value_col not in df.columns:
        st.info(f"No data for '{value_col}' heatmap."); return

    heatmap_data = df.groupby(['L1_Category', 'L2_Category'])[value_col].sum().reset_index()
    if heatmap_data.empty: return

    pivot_table = heatmap_data.pivot(index='L2_Category', columns='L1_Category', values=value_col).fillna(0)
    if pivot_table.empty: return

    # Use a logarithmic scale for color to handle outliers, but display original text
    log_z = np.log10(pivot_table.where(pivot_table > 0, 1)) # Replace 0 with 1 for log

    fig = go.Figure(data=go.Heatmap(
        z=log_z, x=pivot_table.columns, y=pivot_table.index,
        text=pivot_table, texttemplate="%{text:,.0f}", textfont={"size":10},
        colorscale='Viridis', hoverongaps=False,
        colorbar_title=f"{value_col} (Log Scale)"
    ))
    fig.update_layout(title=title, xaxis_title="L1 Category", yaxis_title="L2 Category", yaxis_autorange='reversed')
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Category Performance Analysis")
st.caption("Analyze the performance of your content categories with enhanced metrics.")

category_results = run_module_safely("Category Analysis", analyze_category_performance, filtered_df)
if category_results is not None and not category_results.empty:
    st.dataframe(category_results.style.format({
        'total_clicks': '{:,.0f}', 'total_impressions': '{:,.0f}', 'total_users': '{:,.0f}',
        'avg_bounce_rate': '{:.2%}', 'avg_position': '{:.1f}'
    }, na_rep="-"))
    download_df_button(category_results, "category_performance.csv", "Download Category Analysis")
else:
    st.warning("Could not generate category performance data.")

st.subheader("Category Traffic Distribution")
traffic_col = _pick_col(filtered_df, ['Clicks', 'totalUsers', 'Impressions'])
if traffic_col:
    run_module_safely("Category Heatmap", category_heatmap, filtered_df, traffic_col, f"Category Heatmap by Total {traffic_col}")

st.divider()
st.subheader("Trends & Forecasting")

# REMOVED: Redundant click chart is gone.

primary_metric = _pick_col(filtered_df, ["totalUsers", "Clicks", "screenPageViews"])
if primary_metric:
    daily_series = filtered_df.groupby(pd.to_datetime(filtered_df['date'].dt.date))[primary_metric].sum()

    if len(daily_series) >= 14:
        st.subheader(f"14-Day Forecast for {primary_metric.replace('_', ' ').title()}")
        forecast_data = forecast_series(daily_series)
        if forecast_data is not None and _HAS_PLOTLY:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=daily_series.index, y=daily_series.values, name="Historical Data", line=dict(color="blue")))
            fig.add_trace(go.Scatter(x=forecast_data["date"], y=forecast_data["forecast"], name="Forecast", line=dict(color="red", dash="dash")))
            fig.add_trace(go.Scatter(x=pd.concat([forecast_data["date"], forecast_data["date"][::-1]]), y=pd.concat([forecast_data["high"], forecast_data["low"][::-1]]), fill="toself", fillcolor="rgba(255,0,0,0.2)", line=dict(color="rgba(255,255,255,0)"), name="Confidence Interval"))

            fig.update_layout(
                title=f"14-Day Forecast for {primary_metric}", hovermode="x unified",
                annotations=[dict(xref='paper', yref='paper', x=0.5, y=-0.25, showarrow=False, text="The shaded area represents the 95% confidence interval where actual results are expected to fall.")]
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("This forecast uses a Holt-Winters model to project future performance based on past trends and seasonality in your data.")
    else:
        st.info(f"Insufficient data (need at least 14 days) to generate a reliable forecast for {primary_metric}.")
else:
    st.info("No primary metric (totalUsers, Clicks) found for forecasting.")

st.divider()
st.subheader("📤 Export Full Dataset")
download_df_button(master_df, "growthoracle_full_dataset.csv", "Download Full Processed Data")
