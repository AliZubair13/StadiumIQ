"""StadiumIQ — Sports Ticket Demand & Fan Insights (Streamlit entry point).

Run `python -m src.bootstrap` once, then `streamlit run app.py`.
"""
import streamlit as st

from src.config import SYNTHETIC_BADGE, TEAM_NAME
from src import ui

st.set_page_config(page_title="StadiumIQ", page_icon="🏟️", layout="wide")

pages = [
    st.Page("pages/home.py", title="Home", icon="🏠", default=True),
    st.Page("pages/sales_overview.py", title="Sales Overview", icon="📊"),
    st.Page("pages/forecast.py", title="7-Day Forecast", icon="📈"),
    st.Page("pages/propensity.py", title="Fan Propensity", icon="🎯"),
    st.Page("pages/recommendations.py", title="Recommendations", icon="🧭"),
    st.Page("pages/experiment.py", title="A/B Experiment", icon="🧪"),
    st.Page("pages/data_methods.py", title="Data & Methods", icon="📚"),
]
nav = st.navigation(pages)

with st.sidebar:
    st.markdown("### 🏟️ StadiumIQ")
    st.caption(f"Ticket demand & fan insights for the fictional **{TEAM_NAME}**")
    st.markdown(f"**🧪 {SYNTHETIC_BADGE}**")
    if ui.artifacts_ready():
        st.markdown(f"**Data as of:** {ui.fmt_date(ui.as_of())}")
    st.caption("All data is synthetic. No affiliation with any real team or league.")

nav.run()
