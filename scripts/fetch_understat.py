"""Fetch recent club xG / xA per-player stats from a free Understat GitHub mirror.

FBref and Understat block this cluster's datacenter IP (403 / stripped pages), but the mirror
`douglasbc/scraping-understat-dataset` publishes Understat's per-player season tables as CSVs on
GitHub (raw is reachable). This gives goals / xG / assists / xA / key passes / minutes for the
five big leagues (14/15-21/22) — recent enough to enrich most 2026 World Cup players' club form,
and the xA needed for the (validated) chance-creation assist model.

We pool the last few seasons per player into one row and cache it.

    python scripts/fetch_understat.py            # -> $GOALFORGE_DATA_DIR/understat_players.parquet

Data: Understat (free, non-commercial). Attribution to Understat.
"""
import argparse
import io
import json
import os
import urllib.request

import pandas as pd

REPO = "douglasbc/scraping-understat-dataset"
LEAGUES = ["epl", "la_liga", "serie_a", "bundesliga", "ligue_1"]
SEASONS = ["18-19", "19-20", "20-21", "21-22"]        # recent 4 seasons in the mirror
SUM = ["time", "goals", "xG", "npg", "npxG", "assists", "xA", "shots", "key_passes"]


def _get(url: str) -> bytes:
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "ctlang"}),
                                  timeout=30).read()


def fetch(cache: str, verbose: bool = True) -> pd.DataFrame:
    frames = []
    for lg in LEAGUES:
        listing = json.loads(_get(f"https://api.github.com/repos/{REPO}/contents/datasets/{lg}"))
        urls = {it["name"]: it["download_url"] for it in listing}
        for s in SEASONS:
            name = f"players_{lg}_{s}.csv"
            if name not in urls:
                continue
            df = pd.read_csv(io.BytesIO(_get(urls[name])))
            df["league"], df["season"] = lg, s
            frames.append(df)
            if verbose:
                print(f"  {name}: {len(df)} players")
    allp = pd.concat(frames, ignore_index=True)

    # pool seasons per player (sum volumes; keep the latest team & the modal position)
    agg = allp.groupby("player_name", as_index=False).agg(
        {**{c: "sum" for c in SUM}, "season": "max", "team_title": "last"})
    pos = allp.groupby("player_name").position.agg(lambda s: s.mode().iloc[0]).rename("position")
    out = agg.merge(pos, on="player_name")
    out["nineties"] = out["time"] / 90.0
    os.makedirs(cache, exist_ok=True)
    dst = os.path.join(cache, "understat_players.parquet")
    out.to_parquet(dst, index=False)
    return out, dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.environ.get("GOALFORGE_DATA_DIR", "data"))
    args = ap.parse_args()
    out, dst = fetch(args.cache)
    print(f"\n{len(out)} distinct players pooled over {SEASONS} x {len(LEAGUES)} leagues -> {dst}")
    top = out.sort_values("xA", ascending=False).head(6)
    print(top[["player_name", "team_title", "nineties", "goals", "xG", "assists", "xA", "key_passes"]]
          .round(1).to_string(index=False))


if __name__ == "__main__":
    main()
