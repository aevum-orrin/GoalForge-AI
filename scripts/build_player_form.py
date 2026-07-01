"""Merge free data sources into one per-2026-player "form" table (club xG/xA + international).

Combines, for every player in the 48 real 2026 World Cup squads:
  * club form   — recent goals / xG / assists / xA / key passes / minutes from Understat
    (five big leagues, 2018-2022; via the GitHub mirror, see fetch_understat.py),
  * international — caps and international goals (Wikipedia 2026 squads).

Player names are reconciled across sources by accent-stripped normalization plus a safe
"first-initial + surname" fuzzy pass (unique matches only). Players outside the five big
leagues simply have no club row and fall back to the international + position prior downstream.

    python scripts/fetch_understat.py && python scripts/scrape_wc2026.py   # inputs
    python scripts/build_player_form.py                                    # -> player_form.parquet
"""
import argparse
import os
import re
import unicodedata

import pandas as pd

CLUB = ["time", "goals", "xG", "npxG", "assists", "xA", "key_passes", "nineties"]


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z ]", " ", s.lower()).strip()


def _fi_sur(n: str) -> str:
    parts = n.split()
    return f"{parts[0][0]} {parts[-1]}" if len(parts) >= 2 else n


def _match(squad_names, understat_names):
    """Reconcile names: exact-normalized, then unique first-initial+surname, then unique
    token-subset (squad tokens all contained in an Understat name — catches full surnames like
    'Kylian Mbappe-Lottin')."""
    us_norm, fi_map, fi_amb, us_tokens = {}, {}, set(), {}
    for n in understat_names:
        nn = _norm(n)
        us_norm.setdefault(nn, n)
        k = _fi_sur(nn)
        if k in fi_map and fi_map[k] != n:
            fi_amb.add(k)
        fi_map[k] = n
        us_tokens[n] = set(nn.split())
    out = {}
    for s in squad_names:
        k = _norm(s)
        if k in us_norm:
            out[s] = us_norm[k]
            continue
        fk = _fi_sur(k)
        if fk in fi_map and fk not in fi_amb:
            out[s] = fi_map[fk]
            continue
        st = set(k.split())
        if len(st) >= 2:
            cand = [n for n, t in us_tokens.items() if st <= t]
            if len(cand) == 1:
                out[s] = cand[0]
    return out


def build(cache: str) -> pd.DataFrame:
    sq = pd.read_parquet(os.path.join(cache, "wc2026_squads.parquet"))
    us = pd.read_parquet(os.path.join(cache, "understat_players.parquet"))
    m = _match(sq.player.tolist(), us.player_name.tolist())
    sq["us_name"] = sq.player.map(m)
    usc = us[["player_name", *CLUB]].rename(columns={c: f"club_{c}" for c in CLUB})
    merged = sq.merge(usc, left_on="us_name", right_on="player_name", how="left")
    keep = ["team", "player", "pos", "caps", "goals", "club_nineties", "club_goals",
            "club_xG", "club_npxG", "club_assists", "club_xA", "club_key_passes"]
    out = merged[keep].rename(columns={"goals": "intl_goals"})
    dst = os.path.join(cache, "player_form.parquet")
    out.to_parquet(dst, index=False)
    return out, dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.environ.get("GOALFORGE_DATA_DIR", "data"))
    args = ap.parse_args()
    out, dst = build(args.cache)
    cov = out.club_nineties.notna().mean()
    print(f"{len(out)} squad players | club form matched: {out.club_nineties.notna().sum()} ({cov:.0%}) -> {dst}")
    stars = ["Lionel Messi", "Kylian Mbappé", "Erling Haaland", "Harry Kane", "Jude Bellingham",
             "Vinícius Júnior", "Kevin De Bruyne", "Mohamed Salah"]
    chk = out[out.player.isin(stars)][["player", "team", "pos", "intl_goals", "club_xG", "club_xA", "club_key_passes"]]
    print("\nstar check (should have club xG/xA):")
    print(chk.round(1).to_string(index=False) if len(chk) else "  (none of the sample stars in squads)")


if __name__ == "__main__":
    main()
