"""Build api/model.json for the real 2026 FIFA World Cup (48 teams).

Combines three real sources into one portable, NumPy-free JSON the stdlib engine can serve:

  * Team strength (scoreline): the validated Dixon-Coles checkpoint (Dixon-Coles on martj42
    international results through the checkpoint cutoff; honest held-out backtest in its meta).
    All 48 qualified teams are covered.
  * Scorer rates: each player's REAL international goals-per-cap (Wikipedia 2026 squad tables),
    empirical-Bayes-shrunk toward a position prior so lightly-capped players aren't 0/noisy.
  * Assist rates: position-based ESTIMATE with a mild experience weight — there is no public
    international assist dataset, so this layer is a prior, not fit data. Labeled as such.

Squads are the official 26-man rosters, ordered so the first 11 are a plausible XI
(most-capped player per position in a 4-3-3) — a heuristic the user can override in the UI.

    python scripts/scrape_wc2026.py            # first: cache the squads
    python scripts/build_wc2026_model.py       # -> api/model.json
"""
import argparse
import datetime as dt
import json
import os

import pandas as pd

from goalforge.prediction.agent import GoalForgeAgent

HOSTS = ["United States", "Canada", "Mexico"]
# Rough goals-per-international-appearance priors by position (shrinkage targets).
SCORE_PRIOR = {"GK": 0.005, "DF": 0.04, "MF": 0.12, "FW": 0.35}
ASSIST_PRIOR = {"GK": 0.01, "DF": 0.05, "MF": 0.15, "FW": 0.11}
K = 8.0                     # shrinkage strength, in "caps"; low-cap players lean on the prior
FORMATION = [("GK", 1), ("DF", 4), ("MF", 3), ("FW", 3)]   # 4-3-3 for the default XI


def norm_pos(p: str) -> str:
    p = str(p).upper()
    for k in ("GK", "DF", "MF", "FW"):
        if k in p:
            return k
    return "MF"


def order_xi(rows: pd.DataFrame) -> list[str]:
    """Order a squad so squad[:11] is a plausible 4-3-3 (most-capped per position)."""
    rows = rows.assign(_pos=rows.pos.map(norm_pos))
    rows = rows.sort_values(["caps", "goals"], ascending=False)
    used, starters = set(), []
    for pos, n in FORMATION:
        for name in rows[rows._pos == pos].player.tolist()[:n]:
            starters.append(name)
            used.add(name)
    rest = [n for n in rows.player.tolist() if n not in used]
    while len(starters) < 11 and rest:            # backfill if a line is short (e.g. 3 GKs only)
        starters.append(rest.pop(0))
    return starters + rest


def _load_form():
    """{(team, player): club-form row} from build_player_form.py, or {} if not built."""
    fp = os.path.join(os.environ.get("GOALFORGE_DATA_DIR", "data"), "player_form.parquet")
    if not os.path.exists(fp):
        return {}
    df = pd.read_parquet(fp)
    return {(r.team, r.player): r for _, r in df.iterrows() if pd.notna(r.club_nineties)}


def build(checkpoint: str, squads_pq: str) -> dict:
    agent = GoalForgeAgent.load(checkpoint)
    dc, dc_meta = agent.scoreline, agent.meta
    sq = pd.read_parquet(squads_pq)
    form = _load_form()

    scoring, assist, info, squads, groups = {}, {}, {}, {}, {}
    for team, rows in sq.groupby("team"):
        squads[team] = order_xi(rows)
        groups[team] = rows.group.iloc[0]
        for _, r in rows.iterrows():
            pos, caps, goals, name = norm_pos(r.pos), int(r.caps), int(r.goals), r.player
            f = form.get((team, name))
            if f is not None:            # blend recent club xG (scorer) & xA (assister, the Step-2 win)
                n90, cxg, cxa = float(f["club_nineties"]), float(f["club_xG"]), float(f["club_xA"])
                rate = (goals + cxg + K * SCORE_PRIOR[pos]) / (caps + n90 + K)
                arate = (cxa + K * ASSIST_PRIOR[pos]) / (n90 + K)
            else:                        # no top-5-league data -> international + position prior only
                rate = (goals + K * SCORE_PRIOR[pos]) / (caps + K)
                arate = ASSIST_PRIOR[pos] * (0.7 + 0.3 * caps / (caps + 30))
            scoring[name] = round(rate, 5)
            assist[name] = round(arate, 5)
            info[name] = {"team": team, "pos": pos, "caps": caps, "goals": goals, "club": r.club,
                          "club_xg": round(float(f["club_xG"]), 1) if f is not None else None,
                          "club_xa": round(float(f["club_xA"]), 1) if f is not None else None}

    g_score = round(sum(scoring.values()) / len(scoring), 5)
    g_assist = round(sum(assist.values()) / len(assist), 5)
    meta = {
        "competition": "2026 FIFA World Cup", "n_teams": int(sq.team.nunique()),
        "hosts": HOSTS, "groups": groups,
        "team_source": "martj42 international results (Dixon-Coles)",
        "player_source": "Wikipedia 2026 squads (caps & intl goals) + Understat club xG/xA "
                         "(5 leagues, 2018-2022) for players with top-5-league data",
        "club_coverage": f"{len(form)}/{int(len(sq))} squad players have club xG/xA",
        "cutoff": dc_meta.get("cutoff", ""),
        "generated": dt.date.today().isoformat(),
        "method": {
            "scoreline": "Dixon-Coles team strengths + Poisson score grid",
            "scorer": "intl goals/caps blended with recent club xG (Understat), shrunk to position",
            "assist": "recent club xA/90 (Understat chance creation) where available, else position prior",
            "default_xi": "most-capped player per position in a 4-3-3 (editable)",
            "venue": "neutral by default; hosts (USA/Canada/Mexico) get home advantage",
        },
        "backtest": {k: dc_meta[k] for k in ("test_rps", "test_logloss", "n_matches")
                     if k in dc_meta},
        "note": "Research demo. Team layer is backtested; scorer/assist layers are history/"
                "prior-based and not validated on 2026 outcomes.",
    }
    return {
        "meta": meta,
        "score": {"mu": float(dc.mu_), "home_adv": float(dc.home_adv_), "rho": float(dc.rho_),
                  "attack": {k: float(v) for k, v in dc.attack_.items()},
                  "defence": {k: float(v) for k, v in dc.defence_.items()}},
        "players": {"scoring": scoring, "assist": assist,
                    "global_score": g_score, "global_assist": g_assist},
        "player_info": info,
        "squads": squads,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.path.join(
        os.environ.get("GOALFORGE_MODELS_DIR", "models"), "agent_intl.pkl"))
    ap.add_argument("--squads", default=os.path.join(
        os.environ.get("GOALFORGE_DATA_DIR", "data"), "wc2026_squads.parquet"))
    ap.add_argument("--out", default="api/model.json")
    args = ap.parse_args()

    model = build(args.checkpoint, args.squads)
    with open(args.out, "w") as f:
        json.dump(model, f, ensure_ascii=False)
    kb = os.path.getsize(args.out) / 1024
    print(f"wrote {args.out}: {model['meta']['n_teams']} teams, "
          f"{len(model['players']['scoring'])} players, {kb:.0f} KB")


if __name__ == "__main__":
    main()
