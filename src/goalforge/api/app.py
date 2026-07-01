"""GoalForge FastAPI backend.

Loads a portable JSON model (``scripts/export_model.py``) and serves predictions with a
NumPy-only inference path (no SciPy / Pandas / pickle) — so the same app runs locally and as a
small serverless function (Vercel). The frontend in ``web/`` is served at ``/``; the JSON API
lives under ``/api``.

    source slurm/env_setup.sh
    python scripts/export_model.py                                   # writes model.json
    python -m uvicorn goalforge.api.app:app --host 127.0.0.1 --port 8793
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ..inference import load_model
from ..inference import predict as infer_predict
from .schemas import PredictRequest, PredictResponse, ScoreCell, ScorerProb

_REPO = Path(__file__).resolve().parents[3]
MODEL_JSON = os.environ.get("GOALFORGE_MODEL_JSON") or str(_REPO / "api" / "model.json")
WEB_DIR = _REPO / "public"

app = FastAPI(title="GoalForge API", version="0.2")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_model: dict | None = None


def get_model() -> dict:
    global _model
    if _model is None:
        if not Path(MODEL_JSON).exists():
            raise HTTPException(503, f"model not found: {MODEL_JSON} — run scripts/export_model.py")
        _model = load_model(MODEL_JSON)
    return _model


@app.get("/api/health")
def health():
    return {"status": "ok" if Path(MODEL_JSON).exists() else "no_model", "model": MODEL_JSON}


@app.get("/api/teams")
def teams():
    m = get_model()
    return {"teams": sorted(m["squads"].keys()), "meta": m.get("meta", {})}


@app.get("/api/squad")
def squad(team: str):
    m = get_model()
    if team not in m["squads"]:
        raise HTTPException(404, f"unknown team: {team}")
    return {"team": team, "players": m["squads"][team], "default_xi": m["squads"][team][:11]}


@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    m = get_model()
    for t in (req.home_team, req.away_team):
        if t not in m["squads"]:
            raise HTTPException(404, f"unknown team: {t}")
    if req.home_team == req.away_team:
        raise HTTPException(400, "home_team and away_team must differ")

    home_xi = req.home_xi or m["squads"][req.home_team][:11]
    away_xi = req.away_xi or m["squads"][req.away_team][:11]
    if not home_xi or not away_xi:
        raise HTTPException(400, "each team needs at least one player")

    n_sims = int(np.clip(req.n_sims, 1_000, 200_000))
    pred = infer_predict(m, req.home_team, req.away_team, home_xi, away_xi,
                         neutral=req.neutral, n_sims=n_sims)

    def pr(lst):
        return [ScorerProb(player=n, prob=float(p)) for n, p in lst[:8]]

    return PredictResponse(
        home_team=pred.home_team, away_team=pred.away_team,
        prob_home=pred.prob_home, prob_draw=pred.prob_draw, prob_away=pred.prob_away,
        exp_home_goals=pred.exp_home_goals, exp_away_goals=pred.exp_away_goals,
        most_likely_score=list(pred.most_likely_score),
        top_scores=[ScoreCell(home=h, away=aw, prob=float(p)) for (h, aw), p in pred.top_scores[:6]],
        home_scorers=pr(pred.home_scorers), away_scorers=pr(pred.away_scorers),
        home_assisters=pr(pred.home_assisters), away_assisters=pr(pred.away_assisters),
    )


# Serve the static frontend at "/" (declared after the API routes so /api/* wins).
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
