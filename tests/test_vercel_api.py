"""The self-contained Vercel function (api/index.py) must match the reference inference."""
import importlib.util
import json
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]


def _load_vercel_module():
    spec = importlib.util.spec_from_file_location("vercel_index", ROOT / "api" / "index.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_vercel_function_matches_reference():
    m = _load_vercel_module()
    client = TestClient(m.app)

    assert client.get("/api/health").status_code == 200
    teams = client.get("/api/teams").json()["teams"]
    assert len(teams) >= 2

    body = {"home_team": "Argentina", "away_team": "France", "neutral": True, "n_sims": 20000}
    r = client.post("/api/predict", json=body)
    assert r.status_code == 200, r.text
    p = r.json()

    # Same model + seed as goalforge.inference -> identical prediction (guards against drift).
    from goalforge import inference
    model = json.loads((ROOT / "api" / "model.json").read_text())
    ref = inference.predict(model, "Argentina", "France", neutral=True, n_sims=20000, seed=0)
    assert abs(p["prob_home"] - ref.prob_home) < 1e-9
    assert p["home_scorers"][0]["player"] == ref.home_scorers[0][0]
