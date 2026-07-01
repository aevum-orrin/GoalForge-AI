"""Lineup-aware neural Dixon-Coles — does knowing the actual XI beat team-level Dixon-Coles?

An honest test of the Phase-3 hypothesis. The model is a generalized Dixon-Coles fit by
gradient descent (PyTorch, GPU): a team's attack/defence is its team-level rating PLUS the
mean of its starting XI's per-player attack/defence deltas (with position base effects), so
team strength adapts to *who plays*. Player deltas are strongly L2-shrunk, and any player not
seen in the training fold keeps a zero delta and falls back to team + position — the honest
behaviour for a held-out tournament full of new faces.

Evaluation is **leave-one-tournament-out**: train on 5 tournaments, predict the 6th, repeat.
This is the generalization regime that matters for a World Cup (a fresh tournament, new form,
squad turnover). We report RPS / log-loss / ECE against the same baselines the project already
uses: team-level Dixon-Coles, Elo, and the base rate.

    python scripts/train_neural.py                       # CPU/GPU auto; LOTO over 6 tournaments
    sbatch  slurm/train_neural.sbatch                     # on a Great Lakes GPU

Modeling honesty: hyper-parameters are FIXED a priori (below), not tuned on the test folds.
"""
import argparse
import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from goalforge.evaluation.baselines import EloBaseline, base_rates
from goalforge.evaluation.metrics import log_loss, outcome_index, rps
from goalforge.models.scoreline import DixonColesModel

POS = {"GK": 0, "DF": 1, "MF": 2, "FW": 3}
MAXP = 11            # starters per side
KGRID = 10           # score grid 0..K for outcome probabilities
# Fixed regularization (a priori, not tuned on test):
L2_PLAYER, L2_TEAM, L2_POS = 1.0, 0.05, 0.02
EPOCHS, LR = 400, 0.05


def ece(probs, ys, bins=10):
    """Confidence-ECE over the 3 outcomes."""
    probs, ys = np.asarray(probs), np.asarray(ys)
    conf, pred = probs.max(1), probs.argmax(1)
    correct = (pred == ys).astype(float)
    e = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


class LineupDC(nn.Module):
    def __init__(self, n_teams, n_players):
        super().__init__()
        self.t_att = nn.Embedding(n_teams, 1)
        self.t_def = nn.Embedding(n_teams, 1)
        self.p_att = nn.Embedding(n_players + 1, 1, padding_idx=0)   # 0 = pad / unseen
        self.p_def = nn.Embedding(n_players + 1, 1, padding_idx=0)
        self.pos_att = nn.Embedding(4, 1)
        self.pos_def = nn.Embedding(4, 1)
        self.mu = nn.Parameter(torch.zeros(1))
        self.rho_raw = nn.Parameter(torch.zeros(1))
        for e in (self.t_att, self.t_def, self.pos_att, self.pos_def):
            nn.init.zeros_(e.weight)

    def rho(self):
        return 0.15 * torch.tanh(self.rho_raw)          # keep Dixon-Coles tau positive

    def _side(self, team, pidx, ppos, mask):
        pa = (self.p_att(pidx).squeeze(-1) + self.pos_att(ppos).squeeze(-1)) * mask
        pd_ = (self.p_def(pidx).squeeze(-1) + self.pos_def(ppos).squeeze(-1)) * mask
        n = mask.sum(1).clamp(min=1)
        att = self.t_att(team).squeeze(-1) + pa.sum(1) / n
        dfc = self.t_def(team).squeeze(-1) + pd_.sum(1) / n
        return att, dfc

    def forward(self, b):
        ah, dh = self._side(b["home"], b["hp"], b["hpos"], b["hmask"])
        aa, da = self._side(b["away"], b["ap"], b["apos"], b["amask"])
        log_lh = (self.mu + ah - da).clamp(-2.0, 2.5)   # tournament matches ~ neutral (no home adv)
        log_la = (self.mu + aa - dh).clamp(-2.0, 2.5)
        return log_lh, log_la

    def l2(self):
        return (L2_PLAYER * (self.p_att.weight.pow(2).sum() + self.p_def.weight.pow(2).sum())
                + L2_TEAM * (self.t_att.weight.pow(2).sum() + self.t_def.weight.pow(2).sum())
                + L2_POS * (self.pos_att.weight.pow(2).sum() + self.pos_def.weight.pow(2).sum()))


def _batch(rows, lineups, tvoc, pvoc, device):
    """Build padded tensors for a set of matches."""
    n = len(rows)
    home = torch.zeros(n, dtype=torch.long)
    away = torch.zeros(n, dtype=torch.long)
    hp = torch.zeros(n, MAXP, dtype=torch.long)
    ap = torch.zeros(n, MAXP, dtype=torch.long)
    hpos = torch.zeros(n, MAXP, dtype=torch.long)
    apos = torch.zeros(n, MAXP, dtype=torch.long)
    hmask = torch.zeros(n, MAXP)
    amask = torch.zeros(n, MAXP)
    yh = torch.zeros(n)
    ya = torch.zeros(n)
    for i, (_, m) in enumerate(rows.iterrows()):
        home[i], away[i] = tvoc[m.home_team], tvoc[m.away_team]
        yh[i], ya[i] = m.home_goals, m.away_goals
        for side, pidx, ppos, mask in (("home", hp, hpos, hmask), ("away", ap, apos, amask)):
            lu = lineups.get((m.match_id, 1 if side == "home" else 0), [])
            for k, (pl, pos) in enumerate(lu[:MAXP]):
                pidx[i, k] = pvoc.get(pl, 0)
                ppos[i, k] = POS[pos]
                mask[i, k] = 1.0
    d = {"home": home, "away": away, "hp": hp, "ap": ap, "hpos": hpos, "apos": apos,
         "hmask": hmask, "amask": amask, "yh": yh, "ya": ya}
    return {k: v.to(device) for k, v in d.items()}


def _outcome_probs(lh, la, rho, K=KGRID):
    i = np.arange(K + 1)
    fact = np.array([math.factorial(k) for k in i], dtype=float)
    ph = np.exp(-lh) * lh ** i / fact
    pa = np.exp(-la) * la ** i / fact
    P = np.outer(ph, pa)
    P[0, 0] *= 1 - lh * la * rho
    P[0, 1] *= 1 + lh * rho
    P[1, 0] *= 1 + la * rho
    P[1, 1] *= 1 - rho
    P = np.clip(P, 1e-12, None)
    P /= P.sum()
    return [float(np.tril(P, -1).sum()), float(np.trace(P)), float(np.triu(P, 1).sum())]


def _fit_neural(train, lineups, tvoc, pvoc, device, seed=0):
    torch.manual_seed(seed)
    model = LineupDC(len(tvoc), len(pvoc)).to(device)
    b = _batch(train, lineups, tvoc, pvoc, device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(EPOCHS):
        opt.zero_grad()
        log_lh, log_la = model(b)
        lh, la = log_lh.exp(), log_la.exp()
        nll = (lh - b["yh"] * log_lh).mean() + (la - b["ya"] * log_la).mean()
        (nll + model.l2() / len(train)).backward()
        opt.step()
    return model


def _neural_preds(model, test, lineups, tvoc, pvoc, device):
    b = _batch(test, lineups, tvoc, pvoc, device)
    with torch.no_grad():
        log_lh, log_la = model(b)
    rho = float(model.rho())
    lh, la = log_lh.exp().cpu().numpy(), log_la.exp().cpu().numpy()
    return [_outcome_probs(lh[i], la[i], rho) for i in range(len(test))]


def _score(preds, test):
    ys = [outcome_index(m.home_goals, m.away_goals) for _, m in test.iterrows()]
    r = float(np.mean([rps(p, y) for p, y in zip(preds, ys)]))
    ll = float(np.mean([log_loss(p, y) for p, y in zip(preds, ys)]))
    return r, ll, ece(preds, ys), ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.environ.get("GOALFORGE_DATA_DIR", "data"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    print(f"device={args.device} | torch={torch.__version__} | cuda={torch.cuda.is_available()}")

    matches = pd.read_parquet(os.path.join(args.cache, "intl_lineups_matches.parquet"))
    players = pd.read_parquet(os.path.join(args.cache, "intl_lineups_players.parquet"))
    matches["neutral"] = True
    matches["date"] = pd.to_datetime(matches["date"])
    lineups = {}
    for (mid, home), g in players.groupby(["match_id", "is_home"]):
        lineups[(mid, home)] = list(zip(g.player, g.pos))
    tvoc = {t: i for i, t in enumerate(sorted(set(matches.home_team) | set(matches.away_team)))}
    pvoc = {p: i + 1 for i, p in enumerate(sorted(players.player.unique()))}  # 0 reserved = unseen
    comps = sorted(matches.comp.unique())

    agg = {m: {"rps": [], "ll": [], "ece": [], "n": []} for m in ("Neural", "DixonColes", "Elo", "Base")}
    print(f"\nleave-one-tournament-out over {comps}\n")
    print(f"{'held-out':<11}{'n':>4}  {'Neural':>18}{'DixonColes':>18}{'Elo':>12}{'Base':>8}")
    for comp in comps:
        tr, te = matches[matches.comp != comp], matches[matches.comp == comp]
        model = _fit_neural(tr, lineups, tvoc, pvoc, args.device)
        preds_n = _neural_preds(model, te, lineups, tvoc, pvoc, args.device)

        dc = DixonColesModel(l2=0.1).fit(tr, half_life_days=3650, ref_date=te.date.min())
        elo = EloBaseline().fit(tr)
        base = base_rates(tr)
        preds_dc, preds_elo, preds_base = [], [], []
        for _, m in te.iterrows():
            if m.home_team in dc.attack_ and m.away_team in dc.attack_:
                p = dc.predict_proba(m.home_team, m.away_team, neutral=True)
                preds_dc.append([p["home_win"], p["draw"], p["away_win"]])
            else:
                preds_dc.append(base)
            pe = elo.predict_proba(m.home_team, m.away_team, neutral=True)
            preds_elo.append([pe["home_win"], pe["draw"], pe["away_win"]])
            preds_base.append(base)

        row = {"Neural": preds_n, "DixonColes": preds_dc, "Elo": preds_elo, "Base": preds_base}
        cells = {}
        for name, preds in row.items():
            r, ll, e, _ = _score(preds, te)
            agg[name]["rps"].append(r * len(te))
            agg[name]["ll"].append(ll * len(te))
            agg[name]["ece"].append(e * len(te))
            agg[name]["n"].append(len(te))
            cells[name] = (r, ll, e)
        print(f"{comp:<11}{len(te):>4}  "
              f"{cells['Neural'][0]:.4f}/{cells['Neural'][1]:.3f}   "
              f"{cells['DixonColes'][0]:.4f}/{cells['DixonColes'][1]:.3f}   "
              f"{cells['Elo'][0]:.4f}    {cells['Base'][0]:.4f}")

    print(f"\n{'=== weighted overall (RPS / logloss / ECE) ===':<48}")
    for name in ("Neural", "DixonColes", "Elo", "Base"):
        N = sum(agg[name]["n"])
        r = sum(agg[name]["rps"]) / N
        ll = sum(agg[name]["ll"]) / N
        e = sum(agg[name]["ece"]) / N
        print(f"  {name:<12} RPS {r:.4f}   logloss {ll:.3f}   ECE {e:.3f}")
    print("\n(lower is better; RPS is the headline. Honest read in the printed table.)")


if __name__ == "__main__":
    main()
