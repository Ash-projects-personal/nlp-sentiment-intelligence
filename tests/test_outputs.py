import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_results():
    m = json.loads((ROOT / "outputs" / "metrics.json").read_text())
    assert m["dataset"]["train_rows_used"] == 4000
    assert m["dataset"]["test_rows_used"] == 1000
    for v in m["results"].values():
        assert 0.5 < v["accuracy"] <= 1.0


def test_no_sales_claim():
    m = json.loads((ROOT / "outputs" / "metrics.json").read_text())
    assert "correlation" not in json.dumps(m["results"]).lower()
