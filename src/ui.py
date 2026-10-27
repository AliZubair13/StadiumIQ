"""Streamlit presentation helpers: styling, formatting, cached data access.

Business logic lives in the other src modules; this file only loads artifacts
(with caching) and renders shared UI elements.
"""
from __future__ import annotations

import json
from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import database
from src.config import DB_PATH, REPORTS_DIR, SYNTHETIC_BADGE, TEAM_NAME
from src.recommendations import checkpoint_table, typical_share_curve

NAVY = "#0B1F3A"
TEAL = "#0B9C96"
# Validated categorical order (CVD + contrast checked): actual, model, baseline, extra.
SERIES = {"actual": "#2E5FAE", "xgboost": TEAL, "baseline": "#C9790A", "comparable": "#7A5BC4"}
GRID = "#E6EAF0"
MUTED = "#5B6B82"
STATUS_STYLE = {
    "At risk": ("⚠️", "#B42318"),
    "Near capacity": ("🔶", "#B54708"),
    "Sold out": ("✅", "#067647"),
    "On pace": ("✔️", "#067647"),
}

CSS = f"""
<style>
.block-container {{padding-top: 2.2rem; max-width: 1250px;}}
h1, h2, h3 {{color: {NAVY}; letter-spacing: -0.01em;}}
.siq-header {{display:flex; flex-wrap:wrap; gap:.6rem; align-items:center; margin-bottom:.4rem;}}
.siq-badge {{background:{TEAL}; color:white; border-radius:999px; padding:.18rem .75rem;
            font-size:.78rem; font-weight:600;}}
.siq-asof {{background:#EEF2F7; color:{NAVY}; border-radius:999px; padding:.18rem .75rem; font-size:.78rem;}}
.siq-kpi {{background:white; border:1px solid {GRID}; border-left:4px solid {TEAL}; border-radius:10px;
          padding:.8rem 1rem; height:100%;}}
.siq-kpi .label {{color:{MUTED}; font-size:.78rem; text-transform:uppercase; letter-spacing:.04em;}}
.siq-kpi .value {{color:{NAVY}; font-size:1.65rem; font-weight:700; line-height:1.25;}}
.siq-kpi .sub {{color:{MUTED}; font-size:.8rem;}}
.siq-rec {{border:1px solid {GRID}; border-radius:10px; padding:.9rem 1.1rem; margin-bottom:.8rem; background:white;}}
.siq-rec .tag {{font-size:.72rem; font-weight:700; border-radius:6px; padding:.1rem .5rem; margin-right:.4rem;}}
.siq-note {{color:{MUTED}; font-size:.85rem;}}
[data-testid="stSidebar"] {{background:{NAVY};}}
[data-testid="stSidebar"] * {{color:#F4F7FB !important;}}
</style>
"""


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def fmt_int(x) -> str:
    return "—" if pd.isna(x) else f"{x:,.0f}"


def fmt_money(x, decimals: int = 0) -> str:
    if pd.isna(x):
        return "—"
    if abs(x) >= 1_000_000:
        return f"${x / 1_000_000:,.2f}M"
    return f"${x:,.{decimals}f}"


def fmt_pct(x, decimals: int = 1) -> str:
    return "—" if pd.isna(x) else f"{x:.{decimals}%}"


def fmt_date(d) -> str:
    return "—" if pd.isna(d) else pd.Timestamp(d).strftime("%b %d, %Y")


# ---------------------------------------------------------------------------
# Layout pieces
# ---------------------------------------------------------------------------

def page_header(title: str, subtitle: str, as_of: date) -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.title(title)
    st.markdown(
        f'<div class="siq-header"><span class="siq-badge">🧪 {SYNTHETIC_BADGE}</span>'
        f'<span class="siq-asof">📅 Data as of {fmt_date(as_of)}</span>'
        f'<span class="siq-asof">🏟️ {TEAM_NAME} (fictional)</span></div>',
        unsafe_allow_html=True,
    )
    st.caption(subtitle)


def kpi_row(items: list[tuple[str, str, str]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value, sub) in zip(cols, items):
        col.markdown(
            f'<div class="siq-kpi"><div class="label">{label}</div>'
            f'<div class="value">{value}</div><div class="sub">{sub}</div></div>',
            unsafe_allow_html=True,
        )


def style_fig(fig: go.Figure, height: int = 380, y_title: str | None = None, x_title: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height,
        template="plotly_white",
        font=dict(family="Inter, Segoe UI, sans-serif", color=NAVY, size=13),
        margin=dict(l=10, r=10, t=78, b=10),
        title=dict(x=0, xanchor="left", y=0.985, yanchor="top", font=dict(size=15)),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, title=None),
        bargap=0.15,
        hovermode="x unified",
        hoverlabel=dict(bgcolor="white", font_color=NAVY),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, title=x_title)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, title=y_title, rangemode="tozero")
    return fig


def status_label(status: str) -> str:
    icon, _ = STATUS_STYLE.get(status, ("", MUTED))
    return f"{icon} {status}"


def download_button(df: pd.DataFrame, label: str, file_name: str, key: str) -> None:
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=file_name, mime="text/csv", key=key)


# ---------------------------------------------------------------------------
# Cached data access
# ---------------------------------------------------------------------------

# The pages read only the database and precomputed reports; model files are
# needed for retraining/rescoring, not for viewing.
REQUIRED = [
    DB_PATH,
    REPORTS_DIR / "forecast_metrics.json",
    REPORTS_DIR / "forecast_operational.csv",
    REPORTS_DIR / "propensity_metrics.json",
]


def artifacts_ready() -> bool:
    return all(p.exists() for p in REQUIRED)


@st.cache_resource(show_spinner=False)
def _bootstrap_once() -> bool:
    """Build artifacts in-process (once per server) for hosts that start from a clean clone."""
    from src import bootstrap

    bootstrap.main()
    return True


def require_artifacts() -> None:
    if artifacts_ready():
        return
    st.markdown(CSS, unsafe_allow_html=True)
    try:
        with st.spinner("First launch: generating synthetic data and training models (about 1-2 minutes)..."):
            _bootstrap_once()
    except Exception as exc:  # surface a clear fix instead of a stack trace
        st.error(f"Could not build the data and model artifacts: {exc}")
        st.code("python -m src.bootstrap", language="bash")
        st.stop()
    st.rerun()


def _version() -> float:
    """Cache key: database modification time, so a re-bootstrap refreshes caches."""
    return DB_PATH.stat().st_mtime


@st.cache_data(show_spinner=False)
def _as_of(version: float) -> date:
    with database.connect() as conn:
        return database.get_as_of(conn)


def as_of() -> date:
    return _as_of(_version())


@st.cache_data(show_spinner=False)
def _game_summary(version: float) -> pd.DataFrame:
    with database.connect() as conn:
        return database.game_summary(conn, database.get_as_of(conn))


def game_summary() -> pd.DataFrame:
    return _game_summary(_version())


@st.cache_data(show_spinner=False)
def _cumulative(game_ids: tuple[str, ...], version: float) -> pd.DataFrame:
    with database.connect() as conn:
        return database.cumulative_sales(conn, list(game_ids))


def cumulative(game_ids: list[str]) -> pd.DataFrame:
    return _cumulative(tuple(game_ids), _version())


@st.cache_data(show_spinner=False)
def _season_daily(season: int, version: float) -> pd.DataFrame:
    with database.connect() as conn:
        return database.season_daily_sales(conn, season)


def season_daily(season: int) -> pd.DataFrame:
    return _season_daily(season, _version())


@st.cache_data(show_spinner=False)
def _pace(version: float) -> pd.DataFrame:
    with database.connect() as conn:
        return database.completed_pace_curves(conn, database.get_as_of(conn))


def pace_curves() -> pd.DataFrame:
    return _pace(_version())


@st.cache_data(show_spinner=False)
def _comparables(game_id: str, lead: int, version: float) -> pd.DataFrame:
    with database.connect() as conn:
        return database.comparable_games_at_lead(conn, game_id, lead, database.get_as_of(conn))


def comparables(game_id: str, lead: int) -> pd.DataFrame:
    return _comparables(game_id, lead, _version())


@st.cache_data(show_spinner=False)
def _experiment(version: float) -> pd.DataFrame:
    with database.connect() as conn:
        return database.experiment_data(conn)


def experiment_frame() -> pd.DataFrame:
    return _experiment(_version())


@st.cache_data(show_spinner=False)
def _json(name: str, version: float) -> dict:
    return json.loads((REPORTS_DIR / name).read_text())


def report_json(name: str) -> dict:
    return _json(name, _version())


@st.cache_data(show_spinner=False)
def _csv(name: str, version: float, parse_dates: tuple[str, ...] = ()) -> pd.DataFrame:
    path = REPORTS_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=list(parse_dates))


def report_csv(name: str, parse_dates: tuple[str, ...] = ()) -> pd.DataFrame:
    return _csv(name, _version(), parse_dates)



def operational_forecast() -> pd.DataFrame:
    return report_csv("forecast_operational.csv", ("sales_date", "origin"))


def live_propensity() -> pd.DataFrame:
    return report_csv("propensity_scores_live.csv", ("snapshot_date", "game_date"))


def upcoming_with_status() -> pd.DataFrame:
    summary = game_summary()
    upcoming = summary[summary["status"] == "Upcoming"]
    curve = typical_share_curve(pace_curves())
    return checkpoint_table(upcoming, operational_forecast(), curve, as_of())
