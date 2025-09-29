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
    category: str
    code: str
    message: str
    context: Dict[str, Any] = field(default_factory=dict)

class ValidationCollector:
    def __init__(self):
        self.messages: List[ValidationMessage] = []
    def add(self, category: str, code: str, message: str, **ctx):
        self.messages.append(ValidationMessage(category, code, message, ctx))
    def quality_score(self) -> float:
        crit = sum(1 for m in self.messages if m.category == "Critical")
        warn = sum(1 for m in self.messages if m.category == "Warning")
        info = sum(1 for m in self.messages if m.category == "Info")
        return float(max(0, min(100, 100 - (25 * crit + 8 * warn + 1 * info))))
    def to_dataframe(self) -> pd.DataFrame:
        if not self.messages: return pd.DataFrame()
        return pd.DataFrame([m.__dict__ for m in self.messages])

def coerce_numeric(series, name: str, vc: ValidationCollector, clamp: Optional[Tuple[float, float]] = None) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    bad = s.isna().sum()
    if bad > 0: vc.add("Warning", "NUM_COERCE", f"Non-numeric values in {name}", bad_rows=int(bad))
    if clamp: s = s.clip(clamp[0], clamp[1])
    return s

def safe_dt_parse(col: pd.Series, name: str, vc: ValidationCollector) -> pd.Series:
    parsed = pd.to_datetime(col, errors="coerce", utc=True)
    bad = parsed.isna().sum()
    if bad > 0: vc.add("Warning", "DATE_PARSE", f"Unparseable datetime in {name}", bad_rows=int(bad))
    return parsed

def detect_date_cols(df: pd.DataFrame) -> List[str]:
    if df is None or df.empty: return []
    return [c for c in df.columns if any(k in c.lower() for k in ["date", "dt", "time", "timestamp", "publish"])]

MERGE_STRATEGY = {"gsc_x_prod": "left", "ga4_align": "left"}

def init_state_defaults():
    if "config" not in st.session_state: st.session_state.config = CONFIG
    if "thresholds" not in st.session_state: st.session_state.thresholds = CONFIG["thresholds"].copy()
    if "date_range" not in st.session_state:
        end = date.today()
        start = end - timedelta(days=CONFIG["defaults"]["date_lookback_days"])
        st.session_state.date_range = (start, end)

init_state_defaults()
# ============================
# PART 2/5: Templates, Strong Readers, UI State
# ============================

def _make_template_production():
    return pd.DataFrame({ "Msid": [101, 102], "Title": ["Budget 2025 highlights", "IPL 2025 schedule"], "Path": ["/business/budget-2025", "/sports/cricket/ipl-2025"], "Publish Time": ["2025-08-01 09:15:00", "2025-08-10 18:30:00"] })
def _make_template_ga4():
    return pd.DataFrame({ "customEvent:msid": [101, 102], "date": ["2025-08-01", "2025-08-10"], "screenPageViews": [5000, 15000], "totalUsers": [4000, 10000], "userEngagementDuration": [52.3, 41.0], "bounceRate": [0.42, 0.51] })
def _make_template_gsc():
    return pd.DataFrame({ "Date": ["2025-08-01", "2025-08-10"], "Page": ["https://example.com/a/101.cms", "https://example.com/b/102.cms"], "Query": ["budget 2025", "ipl 2025 schedule"], "Clicks": [200, 1200], "Impressions": [5000, 40000], "CTR": [0.04, 0.03], "Position": [8.2, 12.3] })

def download_df_button(df: pd.DataFrame, filename: str, label: str):
    if df is None or df.empty: return
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button( label=label, data=csv, file_name=filename, mime="text/csv", use_container_width=True )

def read_csv_safely(upload, name: str, vc: ValidationCollector) -> Optional[pd.DataFrame]:
    if upload is None:
        vc.add("Critical", "NO_FILE", f"{name} file not provided")
        return None
    try:
        return pd.read_csv(upload, encoding="utf-8")
    except:
        try:
            upload.seek(0)
            return pd.read_csv(upload, encoding="latin-1")
        except Exception as e:
            vc.add("Critical", "CSV_ENCODING", f"Failed to read {name}", error=str(e))
            return None

with st.sidebar:
    st.subheader("Configuration")
    preset = st.selectbox("Presets", ["Standard", "Conservative", "Aggressive"], index=0)
    if st.button("Apply Preset"):
        st.session_state.thresholds.update(st.session_state.config["presets"][preset])

    t = st.session_state.thresholds
    t["striking_distance_min"] = st.slider("Striking Distance — Min", 5, 50, t["striking_distance_min"])
    t["striking_distance_max"] = st.slider("Striking Distance — Max", 5, 50, t["striking_distance_max"])
    t["similarity_threshold"] = st.slider("Similarity Threshold", 0.30, 0.90, t["similarity_threshold"])

    st.subheader("Analysis Period")
    start_def, end_def = st.session_state.date_range
    start_date = st.date_input("Start Date", value=start_def)
    end_date = st.date_input("End Date", value=end_def)
    if start_date > end_date:
        start_date, end_date = end_date, start_date
    st.session_state.date_range = (start_date, end_date)

st.markdown("### Onboarding & Data Ingestion")
step = st.radio("Steps", ["1) Templates", "2) Upload & Map", "3) Process", "4) Analyze"], horizontal=True)

if step == "1) Templates":
    st.info("Download sample CSV templates to understand required structure.")
    c1, c2, c3 = st.columns(3)
    with c1: download_df_button(_make_template_production(), "template_production.csv", "Download Production Template")
    with c2: download_df_button(_make_template_ga4(), "template_ga4.csv", "Download GA4 Template")
    with c3: download_df_button(_make_template_gsc(), "template_gsc.csv", "Download GSC Template")
    st.stop()

st.subheader("Upload Your Data Files")
prod_file = st.file_uploader("Production Data (CSV)", type=["csv"])
ga4_file = st.file_uploader("GA4 Data (CSV)", type=["csv"])
gsc_file = st.file_uploader("GSC Data (CSV)", type=["csv"])

if not all([prod_file, ga4_file, gsc_file]):
    st.warning("Please upload all three CSV files to proceed.")
    st.stop()

vc_read = ValidationCollector()
prod_df_raw = read_csv_safely(prod_file, "Production", vc_read)
ga4_df_raw = read_csv_safely(ga4_file, "GA4", vc_read)
gsc_df_raw = read_csv_safely(gsc_file, "GSC", vc_read)

if any(df is None for df in [prod_df_raw, ga4_df_raw, gsc_df_raw]):
    st.error("One or more files could not be read. Please check the format and encoding.")
    st.dataframe(vc_read.to_dataframe())
    st.stop()
    # ============================
# PART 3/5: Mapping and Robust Processing
# ============================
def guess_colmap(df, keywords):
    for key, alts in keywords.items():
        for alt in alts:
            for col in df.columns:
                if alt in col.lower(): return col
    return df.columns[0]

prod_map = {"msid": st.selectbox("Production: MSID", prod_df_raw.columns, index=prod_df_raw.columns.get_loc(guess_colmap(prod_df_raw, {"msid": ["msid"]}))) }
ga4_map = { "msid": st.selectbox("GA4: MSID", ga4_df_raw.columns, index=ga4_df_raw.columns.get_loc(guess_colmap(ga4_df_raw, {"msid": ["msid"]}))), "date": st.selectbox("GA4: Date", ga4_df_raw.columns, index=ga4_df_raw.columns.get_loc(guess_colmap(ga4_df_raw, {"date": ["date"]}))), "users": st.selectbox("GA4: Total Users", ga4_df_raw.columns, index=ga4_df_raw.columns.get_loc(guess_colmap(ga4_df_raw, {"users": ["users"]}))), "engagement": st.selectbox("GA4: Engagement", ga4_df_raw.columns, index=ga4_df_raw.columns.get_loc(guess_colmap(ga4_df_raw, {"engagement": ["engagement"]}))), "bounce": st.selectbox("GA4: Bounce Rate", ga4_df_raw.columns, index=ga4_df_raw.columns.get_loc(guess_colmap(ga4_df_raw, {"bounce": ["bounce"]})))}
gsc_map = { "date": st.selectbox("GSC: Date", gsc_df_raw.columns, index=gsc_df_raw.columns.get_loc(guess_colmap(gsc_df_raw, {"date": ["date"]}))), "page": st.selectbox("GSC: Page URL", gsc_df_raw.columns, index=gsc_df_raw.columns.get_loc(guess_colmap(gsc_df_raw, {"page": ["page"]}))), "clicks": st.selectbox("GSC: Clicks", gsc_df_raw.columns, index=gsc_df_raw.columns.get_loc(guess_colmap(gsc_df_raw, {"clicks": ["clicks"]}))), "impressions": st.selectbox("GSC: Impressions", gsc_df_raw.columns, index=gsc_df_raw.columns.get_loc(guess_colmap(gsc_df_raw, {"impressions": ["impr"]}))), "position": st.selectbox("GSC: Position", gsc_df_raw.columns, index=gsc_df_raw.columns.get_loc(guess_colmap(gsc_df_raw, {"position": ["pos"]})))}

@st.cache_data
def process_data(prod_df_raw, ga4_df_raw, gsc_df_raw, prod_map, ga4_map, gsc_map):
    vc = ValidationCollector()
    # RENAME
    prod_df = prod_df_raw[[prod_map["msid"]]].rename(columns={prod_map["msid"]: "msid"})
    ga4_df = ga4_df_raw[[ga4_map["msid"], ga4_map["date"], ga4_map["users"], ga4_map["engagement"], ga4_map["bounce"]]].rename(columns={ga4_map["msid"]: "msid", ga4_map["date"]: "date", ga4_map["users"]: "totalUsers", ga4_map["engagement"]: "userEngagementDuration", ga4_map["bounce"]: "bounceRate"})
    gsc_df = gsc_df_raw[[gsc_map["date"], gsc_map["page"], gsc_map["clicks"], gsc_map["impressions"], gsc_map["position"]]].rename(columns={gsc_map["date"]: "date", gsc_map["page"]: "page_url", gsc_map["clicks"]: "Clicks", gsc_map["impressions"]: "Impressions", gsc_map["position"]: "Position"})
    # PROCESS DATES & MSID
    gsc_df["msid"] = gsc_df["page_url"].str.extract(r'(\d+)\.cms').iloc[:, 0]
    for df in [prod_df, ga4_df, gsc_df]:
        df["msid"] = coerce_numeric(df["msid"], "msid", vc)
        df.dropna(subset=["msid"], inplace=True)
        df["msid"] = df["msid"].astype(int)
    for df in [ga4_df, gsc_df]:
        df["date"] = safe_dt_parse(df["date"], "date", vc).dt.date
    # PROCESS NUMERICS
    for col in ["totalUsers", "userEngagementDuration", "bounceRate"]: ga4_df[col] = coerce_numeric(ga4_df[col], col, vc)
    for col in ["Clicks", "Impressions", "Position"]: gsc_df[col] = coerce_numeric(gsc_df[col], col, vc)
    # MERGE
    master = pd.merge(gsc_df, prod_df.drop_duplicates(subset=["msid"]), on="msid", how="left")
    master = pd.merge(master, ga4_df, on=["msid", "date"], how="left")
    master["Path"] = master["page_url"].str.extract(r'example\.com(.*)/\d+\.cms').iloc[:,0]
    master[["L1_Category", "L2_Category"]] = master["Path"].str.strip('/').str.split('/', n=1, expand=True).fillna("General")
    return master, vc.to_dataframe()

master_df, validation_df = process_data(prod_df_raw, ga4_df_raw, gsc_df_raw, prod_map, ga4_map, gsc_map)

if not validation_df.empty:
    st.subheader("Processing Validation")
    st.dataframe(validation_df)

st.success(f"✅ Data processed: {len(master_df):,} rows created.")
if step != "4) Analyze": st.stop()
    # ============================
# PART 4/5: Core Analysis Modules (MODIFIED)
# ============================

# Date filtering
start_date, end_date = st.session_state.date_range
filtered_df = master_df[(master_df['date'] >= start_date) & (master_df['date'] <= end_date)].copy()
st.info(f"Showing analysis for {start_date} to {end_date} ({len(filtered_df):,} rows)")

# MODIFIED: New function to generate a downloadable opportunities file
def identify_opportunities(df, thresholds):
    """Identifies various opportunities and returns them as a single DataFrame."""
    opportunities = []
    
    # Hidden Gems: High engagement, poor position
    gems = df[(df['userEngagementDuration'] > df['userEngagementDuration'].quantile(0.75)) & (df['Position'] > 15)].copy()
    if not gems.empty:
        gems['opportunity_type'] = 'Hidden Gem (High Engagement, Poor Position)'
        opportunities.append(gems)
        
    # Low CTR at Good Position
    low_ctr = df[(df['Position'] <= 10) & (df['Clicks'] / df['Impressions'].replace(0,1) < 0.03)].copy()
    if not low_ctr.empty:
        low_ctr['opportunity_type'] = 'Low CTR at Good Position'
        opportunities.append(low_ctr)
        
    # High Bounce Rate at Good Position
    high_bounce = df[(df['Position'] <= 15) & (df['bounceRate'] > 0.7)].copy()
    if not high_bounce.empty:
        high_bounce['opportunity_type'] = 'High Bounce Rate at Good Position'
        opportunities.append(high_bounce)
        
    if not opportunities:
        return pd.DataFrame()
        
    return pd.concat(opportunities).drop_duplicates(subset=['msid', 'date'])

# MODIFIED: Reworked scatter plot to be cleaner and more insightful
def scatter_engagement_vs_search(df):
    """Create a cleaner scatter plot with highlighted performance quadrants."""
    if df is None or df.empty or "userEngagementDuration" not in df.columns or "Position" not in df.columns:
        st.info("Insufficient data for Engagement vs. Search scatter plot.")
        return

    d = df.dropna(subset=["userEngagementDuration", "Position", "Clicks"]).copy()
    if d.empty:
        st.info("No complete data rows for scatter plot.")
        return

    # Define quadrant boundaries (using median for robustness)
    median_engagement = d['userEngagementDuration'].median()
    median_position = d['Position'].median()

    # Assign quadrant labels
    def assign_quadrant(row):
        high_eng = row['userEngagementDuration'] >= median_engagement
        good_pos = row['Position'] <= median_position
        if high_eng and good_pos: return "Stars"
        if high_eng and not good_pos: return "Hidden Gems"
        if not high_eng and good_pos: return "Workhorses"
        return "Underperformers"
    
    d['quadrant'] = d.apply(assign_quadrant, axis=1)

    fig = px.scatter(
        d,
        x="Position",
        y="userEngagementDuration",
        size="Clicks",
        color="quadrant",
        hover_data=["msid", "L1_Category"],
        title="Engagement vs. Search Performance Quadrants",
        labels={"Position": "Average Search Position (Lower is Better)", "userEngagementDuration": "Average User Engagement (Higher is Better)"},
        category_orders={"quadrant": ["Stars", "Hidden Gems", "Workhorses", "Underperformers"]},
        color_discrete_map={
            "Stars": "green", "Hidden Gems": "orange",
            "Workhorses": "blue", "Underperformers": "rgba(128, 128, 128, 0.5)"
        },
        size_max=50
    )
    fig.add_vline(x=median_position, line_dash="dash", line_color="grey", annotation_text="Median Position")
    fig.add_hline(y=median_engagement, line_dash="dash", line_color="grey", annotation_text="Median Engagement")
    fig.update_layout(xaxis_autorange="reversed") # Lower position is better, so reverse axis
    st.plotly_chart(fig, use_container_width=True)

# MODIFIED: Enhanced category analysis function
def analyze_category_performance(df):
    """Analyze category performance with additional metrics."""
    if df is None or df.empty: return pd.DataFrame()
    
    # NOTE: 'Query' is not included as it's a one-to-many relationship. A category has thousands of queries.
    # NOTE: 'averageSessionDuration' is not used; the app standardizes on 'userEngagementDuration'.
    agg_dict = {
        "msid": pd.NamedAgg(column="msid", aggfunc="nunique"),
        "Clicks": pd.NamedAgg(column="Clicks", aggfunc="sum"),
        "Impressions": pd.NamedAgg(column="Impressions", aggfunc="sum"),
        "totalUsers": pd.NamedAgg(column="totalUsers", aggfunc="sum"),
        "bounceRate": pd.NamedAgg(column="bounceRate", aggfunc="mean"),
        "Position": pd.NamedAgg(column="Position", aggfunc="mean"),
        "userEngagementDuration": pd.NamedAgg(column="userEngagementDuration", aggfunc="mean")
    }
    
    grouped = df.groupby(["L1_Category", "L2_Category"]).agg(**agg_dict).reset_index()
    grouped = grouped.rename(columns={
        "msid": "unique_articles", "Clicks": "total_clicks", "Impressions": "total_impressions",
        "bounceRate": "avg_bounce_rate", "Position": "avg_position", "userEngagementDuration": "avg_engagement_s"
    })
    
    return grouped.sort_values("total_clicks", ascending=False)

# MODIFIED: New, visually improved heatmap
def category_heatmap(df, value_col, title):
    """Create a visually improved heatmap with a log scale and text labels."""
    if df is None or df.empty or value_col not in df.columns:
        st.info(f"No data for '{value_col}' heatmap."); return
        
    heatmap_data = df.groupby(['L1_Category', 'L2_Category'])[value_col].sum().reset_index()
    
    # Pivot the data for the heatmap
    pivot_table = heatmap_data.pivot(index='L2_Category', columns='L1_Category', values=value_col).fillna(0)
    
    # Use a logarithmic scale for color to handle outliers, but display original text
    log_z = np.log10(pivot_table.replace(0, 1)) # Replace 0 with 1 for log transform
    
    fig = go.Figure(data=go.Heatmap(
        z=log_z,
        x=pivot_table.columns,
        y=pivot_table.index,
        text=pivot_table,
        texttemplate="%{text:,.0f}",
        textfont={"size":10},
        colorscale='Viridis',
        hoverongaps=False))
        
    fig.update_layout(title=title, xaxis_title="L1 Category", yaxis_title="L2 Category")
    st.plotly_chart(fig, use_container_width=True)

def forecast_series(daily_series, periods=14):
    if daily_series is None or len(daily_series) < 14 or not _HAS_STM: return None
    try:
        model = ExponentialSmoothing(daily_series, trend="add", seasonal="add", seasonal_periods=7).fit()
        forecast = model.forecast(periods)
        std_err = np.std(model.resid) * np.sqrt(np.arange(1, periods + 1))
        return pd.DataFrame({"date": forecast.index, "forecast": forecast, "low": forecast - 1.96 * std_err, "high": forecast + 1.96 * std_err})
    except: return None

# ============================
# MAIN ANALYSIS UI
# ============================
st.header("📊 Advanced Analytics & Insights")

st.subheader("Engagement vs. Search Performance Mismatch")
st.caption("Identify content that over- or under-performs relative to its visibility.")

# NEW: Generate and offer download for the opportunities file
opportunities_df = identify_opportunities(filtered_df, st.session_state.thresholds)
if not opportunities_df.empty:
    st.success(f"Identified {len(opportunities_df)} potential opportunities for review.")
    download_df_button(opportunities_df, "growthoracle_opportunities.csv", "Download Opportunities CSV")
else:
    st.info("No specific engagement mismatches found based on current criteria.")

st.subheader("Engagement vs. Search Scatter Analysis")
scatter_engagement_vs_search(filtered_df)
# ============================
# PART 5/5: Category Analysis, Forecasting & Export (MODIFIED)
# ============================

st.divider()
st.subheader("Category Performance Analysis")
st.caption("Analyze the performance of your content categories with enhanced metrics.")

category_results = analyze_category_performance(filtered_df)

if not category_results.empty:
    # Display the new, richer table
    st.dataframe(category_results.style.format({
        'total_clicks': '{:,.0f}', 'total_impressions': '{:,.0f}', 'totalUsers': '{:,.0f}',
        'avg_bounce_rate': '{:.2%}', 'avg_position': '{:.1f}', 'avg_engagement_s': '{:.1f}s'
    }))
    download_df_button(category_results, "category_performance.csv", "Download Category Analysis")
else:
    st.warning("Could not generate category performance data.")

st.subheader("Category Traffic Distribution")
# Call the new, better heatmap
category_heatmap(filtered_df, 'Clicks', "Category Heatmap by Total Clicks")


st.divider()
st.subheader("Trends & Forecasting")
st.caption("Analyze historical trends and generate performance forecasts.")

# REMOVED: The redundant Clicks over time chart is gone.

primary_metric = "totalUsers" if "totalUsers" in filtered_df.columns else "Clicks"
daily_series = filtered_df.groupby(pd.to_datetime(filtered_df['date']))[primary_metric].sum()

if len(daily_series) >= 14:
    st.subheader(f"14-Day Forecast for {primary_metric.replace('_', ' ').title()}")
    forecast_data = forecast_series(daily_series)
    if forecast_data is not None and _HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily_series.index, y=daily_series.values, name="Historical Data", line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=forecast_data["date"], y=forecast_data["forecast"], name="Forecast", line=dict(color="red", dash="dash")))
        fig.add_trace(go.Scatter(x=pd.concat([forecast_data["date"], forecast_data["date"][::-1]]), y=pd.concat([forecast_data["high"], forecast_data["low"][::-1]]), fill="toself", fillcolor="rgba(255,0,0,0.2)", line=dict(color="rgba(255,255,255,0)"), name="Confidence Interval"))
        
        # MODIFIED: Added annotations for clarity
        fig.update_layout(
            title=f"14-Day Forecast for {primary_metric}",
            hovermode="x unified",
            annotations=[
                dict(xref='paper', yref='paper', x=0.5, y=-0.2, showarrow=False,
                     text="The shaded area represents the 95% confidence interval, where actual results are expected to fall.")
            ]
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("This forecast uses the Holt-Winters model to project future performance based on past trends and seasonality in your data.")
else:
    st.info(f"Insufficient data (need at least 14 days) to generate a reliable forecast for {primary_metric}.")

st.divider()
st.subheader("📤 Export Full Dataset")
download_df_button(master_df, "growthoracle_full_dataset.csv", "Download Full Processed Data")
