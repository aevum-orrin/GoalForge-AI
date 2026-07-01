"""Assemble a lineup-aware international-match dataset from StatsBomb open data.

For a lineup-aware scoreline model we need, per match: the two STARTING XIs (with positions)
and the final score. This pulls the modern men's international tournaments StatsBomb covers
for free and caches two frames to the (regenerable) scratch cache:

  intl_lineups_matches.parquet : match_id, comp, date, home_team, away_team, home_goals, away_goals
  intl_lineups_players.parquet : match_id, team, player, pos (GK/DF/MF/FW), is_home

    python scripts/build_intl_lineups.py            # -> $GOALFORGE_DATA_DIR/intl_lineups_*.parquet

Non-commercial use with StatsBomb attribution.
"""
import argparse
import os

import pandas as pd

# (competition_id, season_id, label) — modern tournaments with player overlap into 2026.
TOURNAMENTS = [
    (43, 106, "WC2022"), (43, 3, "WC2018"),
    (55, 282, "Euro2024"), (55, 43, "Euro2020"),
    (223, 282, "Copa2024"), (1267, 107, "AFCON2023"),
]


def _pos_bucket(name: str) -> str:
    n = str(name)
    if "Goalkeeper" in n:
        return "GK"
    if "Back" in n:                      # Center/Full/Wing Back
        return "DF"
    if "Midfield" in n:
        return "MF"
    if any(k in n for k in ("Forward", "Wing", "Striker")):
        return "FW"
    return "MF"


def _starters(lineups: dict) -> list:
    """One row per starting player: (team, player, pos). Starter = a position spell from 00:00."""
    out = []
    for team, df in lineups.items():
        for _, r in df.iterrows():
            start = next((p for p in (r.get("positions") or []) if str(p.get("from")) == "00:00"), None)
            if start:
                out.append((team, r["player_name"], _pos_bucket(start.get("position"))))
    return out


def build(cache: str, verbose: bool = True) -> tuple:
    from statsbombpy import sb
    m_rows, p_rows = [], []
    for cid, sid, label in TOURNAMENTS:
        fx = sb.matches(competition_id=cid, season_id=sid).sort_values("match_date")
        if verbose:
            print(f"{label}: {len(fx)} matches")
        for _, f in fx.iterrows():
            mid = int(f["match_id"])
            try:
                lu = sb.lineups(match_id=mid)
            except Exception as e:  # pragma: no cover
                if verbose:
                    print(f"  skip {mid}: {e}")
                continue
            st = _starters(lu)
            home, away = f["home_team"], f["away_team"]
            if sum(t == home for t, _, _ in st) < 7 or sum(t == away for t, _, _ in st) < 7:
                continue                 # incomplete lineup data -> drop the match
            m_rows.append((mid, label, f["match_date"], home, away,
                           int(f["home_score"]), int(f["away_score"])))
            for team, player, pos in st:
                p_rows.append((mid, team, player, pos, int(team == home)))

    matches = pd.DataFrame(m_rows, columns=["match_id", "comp", "date", "home_team",
                                            "away_team", "home_goals", "away_goals"])
    players = pd.DataFrame(p_rows, columns=["match_id", "team", "player", "pos", "is_home"])
    os.makedirs(cache, exist_ok=True)
    matches.to_parquet(os.path.join(cache, "intl_lineups_matches.parquet"), index=False)
    players.to_parquet(os.path.join(cache, "intl_lineups_players.parquet"), index=False)
    return matches, players


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.environ.get("GOALFORGE_DATA_DIR", "data"))
    args = ap.parse_args()
    matches, players = build(args.cache)
    print(f"\n{len(matches)} matches, {players.player.nunique()} distinct starters, "
          f"{len(players)} starter-rows across {matches.comp.nunique()} tournaments")
    print(matches.groupby("comp").size().to_string())


if __name__ == "__main__":
    main()
