"""Executes every Streamlit page headlessly (requires `python -m src.bootstrap` first)."""
from __future__ import annotations

import pytest

from src import ui
from src.config import ROOT

PAGES = ["home", "sales_overview", "forecast", "propensity", "recommendations", "experiment", "data_methods"]

pytestmark = pytest.mark.skipif(not ui.artifacts_ready(), reason="run `python -m src.bootstrap` first")


def _app(page: str):
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(str(ROOT / "pages" / f"{page}.py"), default_timeout=120).run()


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exceptions(page):
    at = _app(page)
    assert not at.exception, [e.message for e in at.exception]
    assert at.title, "page should render a title"


def test_sales_overview_every_season_and_a_game():
    at = _app("sales_overview")
    for i in range(len(at.selectbox[0].options)):
        at.selectbox[0].select_index(i).run()
        assert not at.exception, [e.message for e in at.exception]
    at = _app("sales_overview")
    assert len(at.selectbox[1].options) > 1
    for i in (1, len(at.selectbox[1].options) - 1):
        at.selectbox[1].select_index(i).run()
        assert not at.exception, [e.message for e in at.exception]


def test_forecast_page_every_upcoming_game():
    at = _app("forecast")
    for i in range(len(at.selectbox[0].options)):
        at.selectbox[0].select_index(i).run()
        assert not at.exception, [e.message for e in at.exception]
