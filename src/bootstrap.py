"""One-command pipeline: generate data -> SQLite -> train/evaluate -> artifacts.

Usage:  python -m src.bootstrap
"""
from __future__ import annotations

import json
import time

from src import database, train_forecast, train_propensity
from src.config import AS_OF_DATE, DB_PATH, REPORTS_DIR, SimulationConfig
from src.experiments import analyze_experiment
from src.generate_data import generate_all


def main() -> None:
    t0 = time.time()
    cfg = SimulationConfig()
    print(f"[bootstrap] generating synthetic data (seed={cfg.seed}, as-of={cfg.as_of}) ...")
    tables = generate_all(cfg)
    database.create_database(tables, DB_PATH, cfg.as_of, cfg.seed)
    print(f"[bootstrap] wrote {DB_PATH} " + ", ".join(f"{k}={len(v):,}" for k, v in tables.items()))

    REPORTS_DIR.mkdir(exist_ok=True)
    with database.connect(DB_PATH) as conn:
        forecast = train_forecast.run(conn, AS_OF_DATE)
        propensity = train_propensity.run(conn, AS_OF_DATE)
        exp = analyze_experiment(database.experiment_data(conn))
    exp_out = {"test": exp["test"], "srm": exp["srm"], "interpretation": exp["interpretation"],
               "groups": exp["groups"].to_dict(orient="records")}
    (REPORTS_DIR / "experiment_summary.json").write_text(json.dumps(exp_out, indent=2, default=float))
    print(f"[bootstrap] experiment: diff={exp['test']['diff_pp']:+.2f}pp p={exp['test']['p_value']:.3f}")
    print(f"[bootstrap] forecast selected={forecast['selected_approach']} | propensity selected={propensity['selected']}")
    print(f"[bootstrap] done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
