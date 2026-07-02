"""Player-layer ML bake-off — do learned models on features beat the simple rate baselines?

The honest question from Steps 1-2: a per-player rate (shrunk to position, or xA-based) ranks
scorers/assisters. Can a model that ingests a richer FEATURE VECTOR (recent club xG / xA / npxG
/ key passes / shots per 90 from Understat, plus leakage-free international rate, plus position)
do better? We compare, per player in each match's starting XI, at predicting P(scores>=1) and
P(assists>=1), leave-one-tournament-out:

  * baselines   — position prior, international rate, club-xA rate (the Step-2 winner),
  * gbm         — gradient boosting (LightGBM) on the feature vector,
  * mlp         — a small neural net on the same features (torch; runs on CPU or GPU),

scored by recall@{1,3,5} (rank within each team's XI) + log-loss/ECE. Honest either way.

    python scripts/player_ml.py                 # baselines + GBM (+ MLP if torch present)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_player_form import _match  # noqa: E402
from evaluate_players import KS, TOURN, _order, _recall_at_k  # noqa: E402

from goalforge.data.statsbomb import load_competition  # noqa: E402
from goalforge.models.player import PlayerRatings  # noqa: E402

CLUB = ["goals", "xG", "npxG", "assists", "xA", "key_passes", "shots"]   # summed -> per 90
FEATS = [f"club_{c}" for c in CLUB] + ["has_club", "intl_rate", "caps",
                                       "pos_GK", "pos_DF", "pos_MF", "pos_FW"]


def _club_form(cache, players):
    us = pd.read_parquet(os.path.join(cache, "understat_players.parquet"))
    m = _match(list(players), us.player_name.tolist())
    us = us.set_index("player_name")
    form = {}
    for tp, un in m.items():
        r = us.loc[un] if not isinstance(us.loc[un], pd.DataFrame) else us.loc[un].iloc[0]
        n90 = max(float(r["nineties"]), 1.0)
        form[tp] = {c: float(r[c]) / n90 for c in CLUB}
    return form


def _rows(labels_data, form, pr, kind, lineups):
    col = "scorer" if kind == "scoring" else "assister"
    rows = []
    for mid in labels_data.matches.match_id:
        r = labels_data.matches.loc[labels_data.matches.match_id == mid, ["home_team", "away_team"]].values[0]
        for team in r:
            xi = lineups[(lineups.match_id == mid) & (lineups.team == team)]
            if len(xi) < 7:
                continue
            hit = set(labels_data.goals[(labels_data.goals.match_id == mid)
                                        & (labels_data.goals.team == team)][col].dropna())
            for _, p in xi.iterrows():
                cf = form.get(p.player, {})
                row = {f"club_{c}": cf.get(c, 0.0) for c in CLUB}
                row.update({"has_club": int(p.player in form), "intl_rate": pr.rate(p.player, kind),
                            "caps": 0.0,
                            "pos_GK": int(p.pos == "GK"), "pos_DF": int(p.pos == "DF"),
                            "pos_MF": int(p.pos == "MF"), "pos_FW": int(p.pos == "FW"),
                            "match_id": mid, "team": team, "player": p.player, "y": int(p.player in hit)})
                rows.append(row)
    return pd.DataFrame(rows)


def _ece(p, y, bins=10):
    p, y = np.asarray(p), np.asarray(y)
    e = 0.0
    for b in range(bins):
        m = (p > b / bins) & (p <= (b + 1) / bins)
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def _recall(df, prob_col):
    """recall@k by ranking each team's XI in each match by prob_col."""
    hits, n = {k: 0 for k in KS}, 0
    for (_, _), g in df.groupby(["match_id", "team"]):
        ranked = _order(dict(zip(g.player, g[prob_col])))
        actual = list(g[g.y == 1].player)
        h, m = _recall_at_k(ranked, actual)
        for k in KS:
            hits[k] += h[k]
        n += m
    return hits, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.environ.get("GOALFORGE_DATA_DIR", "data"))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    data = {t[2]: load_competition(t[0], t[1], verbose=False) for t in TOURN}
    lineups = pd.read_parquet(os.path.join(args.cache, "intl_lineups_players.parquet"))
    all_players = set(lineups.player)
    form = _club_form(args.cache, all_players)
    print(f"club-form matched: {len(form)}/{len(all_players)} tournament starters")

    try:
        import lightgbm as lgb
        have_gbm = True
    except ImportError:
        have_gbm = False
        print("(lightgbm not installed — skipping GBM)")
    mlp_fn = _make_mlp(args.device)

    for kind in ("scoring", "assist"):
        methods = ["position", "intl_rate", "club_xa"] + (["gbm"] if have_gbm else []) + \
                  (["mlp"] if mlp_fn else [])
        agg = {m: {k: 0 for k in KS} for m in methods}
        ntot = 0
        for label in data:
            tr_app = pd.concat([data[o].appearances for o in data if o != label])
            tr_goal = pd.concat([data[o].goals for o in data if o != label])
            pr = PlayerRatings().fit(tr_app, tr_goal)
            lu_tr = lineups[lineups.match_id.isin(set(pd.concat([data[o].matches for o in data if o != label]).match_id))]
            lu_te = lineups[lineups.match_id.isin(set(data[label].matches.match_id))]
            train = pd.concat([_rows(data[o], form, pr, kind, lu_tr) for o in data if o != label], ignore_index=True)
            test = _rows(data[label], form, pr, kind, lu_te)
            if test.empty:
                continue
            # simple-rate baselines
            test["position"] = test.apply(lambda r: {"FW": .35, "MF": .12, "DF": .04, "GK": .005}
                                          .get(_pos(r), .08), axis=1)
            test["club_xa"] = test["club_xA"] if kind == "assist" else test["club_xG"]
            preds = {"position": "position", "intl_rate": "intl_rate", "club_xa": "club_xa"}
            if have_gbm:
                clf = lgb.LGBMClassifier(n_estimators=200, num_leaves=15, learning_rate=0.05,
                                         min_child_samples=30, subsample=0.8, verbose=-1)
                clf.fit(train[FEATS], train.y)
                test["gbm"] = clf.predict_proba(test[FEATS])[:, 1]
                preds["gbm"] = "gbm"
            if mlp_fn:
                test["mlp"] = mlp_fn(train[FEATS].values, train.y.values, test[FEATS].values)
                preds["mlp"] = "mlp"
            for m, col in preds.items():
                h, n = _recall(test, col)
                for k in KS:
                    agg[m][k] += h[k]
                if m == "position":
                    ntot += n

        title = "SCORER" if kind == "scoring" else "ASSISTER"
        print(f"\n=== {title} recall@k, LOTO (n={ntot}) ===")
        print(f"  {'method':<11}" + "".join(f"   r@{k}" for k in KS))
        for m in methods:
            print(f"  {m:<11}" + "".join(f"  {agg[m][k] / max(ntot, 1):5.1%}" for k in KS))


def _pos(r):
    for p in ("GK", "DF", "MF", "FW"):
        if r.get(f"pos_{p}"):
            return p
    return "MF"


def _make_mlp(device):
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return None

    def run(Xtr, ytr, Xte):
        torch.manual_seed(0)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        xt = torch.tensor(Xtr, dtype=torch.float32, device=device)
        yt = torch.tensor(ytr, dtype=torch.float32, device=device)
        net = nn.Sequential(nn.Linear(Xtr.shape[1], 32), nn.ReLU(), nn.Dropout(0.3),
                            nn.Linear(32, 1)).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-3)
        lossf = nn.BCEWithLogitsLoss()
        for _ in range(300):
            opt.zero_grad()
            loss = lossf(net(xt).squeeze(-1), yt)
            loss.backward()
            opt.step()
        with torch.no_grad():
            return torch.sigmoid(net(torch.tensor((Xte), dtype=torch.float32, device=device)).squeeze(-1)).cpu().numpy()
    return run


if __name__ == "__main__":
    main()
