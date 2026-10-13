import os

import pytest

from complaints.core import ROOT


@pytest.mark.skipif(os.getenv("RUN_LLM_TESTS") != "1", reason="opt-in actual three-model UI comparison")
def test_app_compare():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    app.button[1].click().run()
    assert not app.exception
    results = app.session_state["results"]
    assert [r["model"] for r in results] == ["classic", "base", "lora"]
    assert len({r["text"] for r in results}) == 1
    assert all(r["seconds"] > 0 for r in results)
