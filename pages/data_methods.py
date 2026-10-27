import pandas as pd
import streamlit as st

from src import database, ui
from src.config import (
    BASE_CAPACITY,
    COMPLETED_SEASONS,
    CURRENT_SEASON,
    N_CUSTOMERS,
    SALES_WINDOW_DAYS,
    SEED,
    TEST_SEASON,
    TRAIN_SEASONS,
    VALIDATION_SEASON,
)

ui.require_artifacts()
as_of = ui.as_of()
ui.page_header("Data & Methods", "Synthetic data schema, simulation assumptions, leakage safeguards and limitations.", as_of)


@st.cache_data(show_spinner=False)
def table_counts(version: float) -> pd.DataFrame:
    with database.connect() as conn:
        rows = []
        for t in ["games", "customers", "transactions", "engagement", "experiment_assignments",
                  "experiment_outcomes", "daily_sales"]:
            n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            rows.append({"table": t, "rows": n})
        return pd.DataFrame(rows)


st.markdown(
    f"""
### Simulation at a glance
- **Fixed seed:** {SEED} · **Simulation as-of date:** {ui.fmt_date(as_of)} (the app never reads the real clock).
- **Seasons:** {COMPLETED_SEASONS[0]}–{COMPLETED_SEASONS[-1]} completed, {CURRENT_SEASON} in progress; 10 home
  games per season at a fictional {BASE_CAPACITY:,}-seat venue.
- **Fans:** {N_CUSTOMERS:,} registered fan accounts; every ticket in the data is bought by one of them.
- **Sales window:** {SALES_WINDOW_DAYS} days before kickoff through game day, with a complete daily calendar.
- **Observed vs missing:** days after the as-of date are stored as missing (NULL), never as zero sales.
  Current-season purchases after the as-of date are simulated only to keep sell-outs coherent and are
  then discarded — they never reach the database, features or training.
"""
)
st.dataframe(table_counts(ui.DB_PATH.stat().st_mtime), hide_index=True)

st.markdown(
    """
### How the simulator creates relationships
- Each fan has an **unobserved loyalty** level (higher on average for the "Loyalist" segment), plus a
  yearly interest shock. Loyalty drives both engagement frequency and purchase probability.
- Purchase probability per fan and game rises with opponent popularity, rivalry and weekend games;
  falls with distance (especially for weeknight games) and with price relative to a reference price;
  rises if the fan bought the previous home game (habit); and includes an unobserved game-level shock.
- Buyers choose a purchase day from segment-specific lead-time profiles (early on-sale buyers vs.
  last-minute buyers) perturbed by random daily noise and two promotional spikes per game.
- Purchases are processed in date order until the venue is full, so realized sales are capped by inventory.
- Fans browse more (especially ticket pages) in the three weeks before a purchase.
- **These assumptions are why the models find the patterns they find.** They are plausible, not measured.

### Leakage prevention
| Risk | Safeguard |
|---|---|
| Today's sales inside today's features | Lags/rolling means shifted by ≥1 day, computed within each game |
| Future actuals during multi-day forecasts | Recursive forecasting: future actuals discarded at the origin; predictions feed later lags and inventory |
| Training on the evaluation period | Expanding-window refit at each rolling origin uses only labels dated on/before the origin |
| Tuning on the test set | Model choice on 2023; 2024 scored once afterwards |
| Future purchases/engagement in fan features | All propensity features filtered to dates ≤ snapshot (kickoff − 14 days) |
| Partially observed outcomes | Only games with kickoff ≤ as-of date are used to train/evaluate propensity |
| Identifier memorization | Customer and game IDs are never model inputs |
"""
)
st.markdown(
    f"**Splits:** train {TRAIN_SEASONS[0]}–{TRAIN_SEASONS[-1]}, validation {VALIDATION_SEASON}, test {TEST_SEASON}. "
    f"Operational models are refit on everything observed up to {ui.fmt_date(as_of)} with the validation-selected settings."
)

st.markdown(
    """
### Limitations
- Synthetic data does not establish real-world model performance.
- Relationships reflect the generator's assumptions.
- Sales are constrained by inventory; forecasts are of realized sales, not unconstrained demand.
- Purchase propensity is not treatment uplift.
- Using price as a forecasting input does not establish causal price sensitivity.
- No revenue improvement has been demonstrated.
"""
)
