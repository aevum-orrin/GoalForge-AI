"""Export a trained GoalForgeAgent checkpoint to a portable JSON model (for serving / Vercel).

    python scripts/export_model.py [--checkpoint models/agent_intl.pkl] [--out model.json]

The JSON holds the Dixon-Coles parameters, per-player scoring/assist rates, and squads — small
and pickle-free, so the FastAPI backend and the Vercel serverless function load the same model.
"""
import argparse
import json
import os

from goalforge.prediction.agent import GoalForgeAgent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.path.join(
        os.environ.get("GOALFORGE_MODELS_DIR", "models"), "agent_intl.pkl"))
    ap.add_argument("--out", default="api/model.json")
    args = ap.parse_args()

    a = GoalForgeAgent.load(args.checkpoint)
    dc, r = a.scoreline, a.ratings
    model = {
        "meta": a.meta,
        "score": {
            "mu": float(dc.mu_), "home_adv": float(dc.home_adv_), "rho": float(dc.rho_),
            "attack": {k: float(v) for k, v in dc.attack_.items()},
            "defence": {k: float(v) for k, v in dc.defence_.items()},
        },
        "players": {
            "scoring": {k: float(v) for k, v in r.scoring_.items()},
            "assist": {k: float(v) for k, v in r.assist_.items()},
            "global_score": float(r.global_score_), "global_assist": float(r.global_assist_),
        },
        "squads": a.squads,
    }
    with open(args.out, "w") as f:
        json.dump(model, f)
    size = os.path.getsize(args.out) / 1024
    print(f"wrote {args.out}: {len(model['squads'])} teams, "
          f"{len(model['players']['scoring'])} players, {size:.0f} KB")


if __name__ == "__main__":
    main()
