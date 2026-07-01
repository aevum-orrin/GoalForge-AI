from fastapi.testclient import TestClient

from goalforge.api import app as api_app
from goalforge.data.synthetic import generate_dataset
from goalforge.models.player import PlayerRatings
from goalforge.models.scoreline import DixonColesModel


def _make_model():
    d = generate_dataset(n_teams=6, seed=0)
    dc = DixonColesModel(l2=0.1).fit(d.matches)
    r = PlayerRatings().fit(d.appearances, d.goals)
    squads = {t: list(d.appearances[d.appearances.team == t]
                      .groupby("player").minutes.sum().sort_values(ascending=False).index)
              for t in d.teams}
    model = {
        "meta": {"note": "test"},
        "score": {"mu": dc.mu_, "home_adv": dc.home_adv_, "rho": dc.rho_,
                  "attack": dc.attack_, "defence": dc.defence_},
        "players": {"scoring": r.scoring_, "assist": r.assist_,
                    "global_score": r.global_score_, "global_assist": r.global_assist_},
        "squads": squads,
    }
    return model, d.teams


def test_api_endpoints(monkeypatch):
    model, teams = _make_model()
    monkeypatch.setattr(api_app, "_model", model)
    client = TestClient(api_app.app)

    r = client.get("/api/teams")
    assert r.status_code == 200
    assert set(r.json()["teams"]) == set(teams)

    r = client.get(f"/api/squad?team={teams[0]}")
    assert r.status_code == 200
    assert len(r.json()["default_xi"]) == 11

    body = {"home_team": teams[0], "away_team": teams[1], "neutral": True, "n_sims": 2000}
    r = client.post("/api/predict", json=body)
    assert r.status_code == 200, r.text
    js = r.json()
    assert abs(js["prob_home"] + js["prob_draw"] + js["prob_away"] - 1.0) < 1e-6
    assert len(js["home_scorers"]) >= 1
    assert len(js["most_likely_score"]) == 2

    assert client.post("/api/predict", json={**body, "home_team": "Nope"}).status_code == 404
    assert client.post("/api/predict", json={**body, "away_team": teams[0]}).status_code == 400
