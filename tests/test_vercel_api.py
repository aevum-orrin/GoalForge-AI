"""The Vercel serverless functions (api/*.py, stdlib only) must import and produce valid output."""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = ROOT / "api"


def _load(name):
    sys.path.insert(0, str(API))          # so `from _engine import ...` resolves, as on Vercel
    try:
        spec = importlib.util.spec_from_file_location(name, API / f"{name}.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m
    finally:
        sys.path.remove(str(API))


def test_engine_predict_valid():
    eng = _load("_engine")
    model = json.loads((API / "model.json").read_text())
    r = eng.predict_dict(model, "Argentina", "France",
                         model["squads"]["Argentina"][:11], model["squads"]["France"][:11], True)
    assert abs(r["prob_home"] + r["prob_draw"] + r["prob_away"] - 1.0) < 1e-6
    assert len(r["most_likely_score"]) == 2
    probs = [s["prob"] for s in r["home_scorers"]]
    assert probs == sorted(probs, reverse=True)
    assert r["home_scorers"][0]["player"] in model["squads"]["Argentina"]


def test_handlers_import():
    for name in ("health", "teams", "squad", "predict"):
        assert hasattr(_load(name), "handler")
