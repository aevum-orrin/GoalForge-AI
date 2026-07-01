"""Self-contained GoalForge prediction API for Vercel (NumPy + FastAPI only).

No imports from the `goalforge` package and no static-file serving, so the serverless bundle is
tiny with no cross-directory dependencies. The prediction logic mirrors
`goalforge.simulation.montecarlo` exactly (verified in tests). The model (`api/model.json`) is
loaded lazily and searched on several paths, so a missing file yields a clear API error rather
than an import crash. The frontend is served separately from `public/`.
"""
import json
import math
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="GoalForge API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_MODEL = None


def _load_model() -> dict:
    here = Path(__file__).resolve().parent
    candidates = [here / "model.json", here.parent / "model.json",
                  Path.cwd() / "api" / "model.json", Path("/var/task/api/model.json")]
    for p in candidates:
        if p.exists():
            return json.loads(p.read_text())
    raise HTTPException(503, "model.json not found (run scripts/export_model.py)")


def model() -> dict:
    global _MODEL
    if _MODEL is None:
        _MODEL = _load_model()
    return _MODEL


class PredictRequest(BaseModel):
    home_team: str
    away_team: str
    home_xi: Optional[List[str]] = None
    away_xi: Optional[List[str]] = None
    neutral: bool = True
    n_sims: int = 50000


def _expected_goals(M, home, away, neutral):
    s = M["score"]
    att, dfc = s["attack"], s["defence"]
    ha = 0.0 if neutral else s["home_adv"]
    lh = math.exp(s["mu"] + ha + att.get(home, 0.0) + dfc.get(away, 0.0))
    la = math.exp(s["mu"] + att.get(away, 0.0) + dfc.get(home, 0.0))
    return lh, la


def _rate(M, name, kind):
    p = M["players"]
    table = p["scoring"] if kind == "scoring" else p["assist"]
    default = p["global_score"] if kind == "scoring" else p["global_assist"]
    return float(table.get(name, default))


def _allocate(counts, weights, rng):
    out = np.zeros((len(counts), len(weights)), dtype=np.int64)
    s = weights.sum()
    w = weights / s if s > 0 else np.ones(len(weights)) / len(weights)
    for g in np.unique(counts):
        if g <= 0:
            continue
        m = counts == g
        out[m] = rng.multinomial(int(g), w, size=int(m.sum()))
    return out


def _predict(M, home, away, home_xi, away_xi, neutral, n_sims,
             assisted_rate=0.78, pen_fraction=0.10, max_goals=10):
    rng = np.random.default_rng(0)
    lh, la = _expected_goals(M, home, away, neutral)
    gh = rng.poisson(lh, n_sims)
    ga = rng.poisson(la, n_sims)

    K = max_goals
    grid = np.zeros((K + 1, K + 1))
    np.add.at(grid, (np.clip(gh, 0, K), np.clip(ga, 0, K)), 1.0)
    grid /= n_sims
    order = np.argsort(grid, axis=None)[::-1][:6]
    top = [{"home": int(i), "away": int(j), "prob": float(grid[i, j])}
           for i, j in (np.unravel_index(o, grid.shape) for o in order)]

    def team_alloc(counts, xi):
        sw = np.array([max(_rate(M, n, "scoring"), 0.0) for n in xi], dtype=float)
        aw = np.array([max(_rate(M, n, "assist"), 0.0) for n in xi], dtype=float)
        pen = int(np.argmax(sw)) if len(xi) else None
        gp = rng.binomial(counts, pen_fraction) if (pen is not None and pen_fraction > 0) \
            else np.zeros_like(counts)
        scorers = _allocate(counts - gp, sw, rng)
        if pen is not None:
            scorers[:, pen] += gp
        assisters = _allocate(rng.binomial(counts, assisted_rate), aw, rng)
        sp = sorted(zip(xi, (scorers >= 1).mean(0)), key=lambda t: -t[1])
        ap = sorted(zip(xi, (assisters >= 1).mean(0)), key=lambda t: -t[1])
        return ([{"player": n, "prob": float(p)} for n, p in sp[:8]],
                [{"player": n, "prob": float(p)} for n, p in ap[:8]])

    hs, ha = team_alloc(gh, home_xi)
    as_, aa = team_alloc(ga, away_xi)
    return {
        "home_team": home, "away_team": away,
        "prob_home": float(np.mean(gh > ga)), "prob_draw": float(np.mean(gh == ga)),
        "prob_away": float(np.mean(gh < ga)),
        "exp_home_goals": float(gh.mean()), "exp_away_goals": float(ga.mean()),
        "most_likely_score": [top[0]["home"], top[0]["away"]], "top_scores": top,
        "home_scorers": hs, "away_scorers": as_,
        "home_assisters": ha, "away_assisters": aa,
    }


@app.get("/api/health")
def health():
    try:
        return {"status": "ok", "teams": len(model()["squads"])}
    except Exception as e:  # never fail import; report clearly
        return {"status": "error", "detail": str(getattr(e, "detail", e))}


@app.get("/api/teams")
def teams():
    M = model()
    return {"teams": sorted(M["squads"].keys()), "meta": M.get("meta", {})}


@app.get("/api/teams/{team}/squad")
def squad(team: str):
    M = model()
    if team not in M["squads"]:
        raise HTTPException(404, f"unknown team: {team}")
    return {"team": team, "players": M["squads"][team], "default_xi": M["squads"][team][:11]}


@app.post("/api/predict")
def predict(req: PredictRequest):
    M = model()
    for t in (req.home_team, req.away_team):
        if t not in M["squads"]:
            raise HTTPException(404, f"unknown team: {t}")
    if req.home_team == req.away_team:
        raise HTTPException(400, "home_team and away_team must differ")
    home_xi = req.home_xi or M["squads"][req.home_team][:11]
    away_xi = req.away_xi or M["squads"][req.away_team][:11]
    if not home_xi or not away_xi:
        raise HTTPException(400, "each team needs at least one player")
    n_sims = int(np.clip(req.n_sims, 1_000, 200_000))
    return _predict(M, req.home_team, req.away_team, home_xi, away_xi, req.neutral, n_sims)
