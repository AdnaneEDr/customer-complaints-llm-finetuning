import io

import pandas as pd
import pytest

from complaints.batch import export_csv, load_csv
from complaints.core import ROOT


def test_csv_validation():
    with pytest.raises(ValueError):
        load_csv(io.StringIO("wrong\na\n"))
    with pytest.raises(ValueError):
        load_csv(io.StringIO('text\n" "\n'))


@pytest.mark.skipif(not (ROOT / "artifacts/classic.joblib").exists(), reason="requires real baseline")
def test_csv_inference_export():
    from complaints.models import Engine

    texts = load_csv(ROOT / "demo.csv")
    engine = Engine("classic")
    results = [{"text": text, **engine.predict(text)} for text in texts]
    exported = pd.read_csv(io.BytesIO(export_csv(results)))
    assert len(exported) == 2 and exported["output"].notna().all()
    engine.close()


@pytest.mark.skipif(not (ROOT / "artifacts/classic.joblib").exists(), reason="requires real baseline")
def test_app_analyze():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception
    assert len(app.json) >= 1
    assert app.session_state["results"][0]["output"]["category"] in [
        "Mortgage",
        "Debt collection",
        "Credit card",
    ]
