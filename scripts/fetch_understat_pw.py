"""Playwright Understat scraper — runs ON Great Lakes (no laptop needed).

Understat assigns its per-player table to a JavaScript global `playersData` at runtime, so it
is absent from the served HTML (static scrapers — urllib, curl_cffi — get a stripped page). A
real headless browser executes that JS; we then read the global with page.evaluate(). Headless
Chromium runs fine on the GL login node and is cached on scratch (not $HOME).

One-time setup (on GL):
    source slurm/env_setup.sh
    pip install playwright
    PLAYWRIGHT_BROWSERS_PATH=$(dirname "$GOALFORGE_DATA_DIR")/playwright playwright install chromium

Run:
    python scripts/fetch_understat_pw.py        # -> $GOALFORGE_DATA_DIR/understat_players.parquet

Output matches fetch_understat.py, so build_player_form.py / build_wc2026_model.py are unchanged.
Data: Understat (free, non-commercial); attribution to Understat.
"""
import argparse
import os

# Point Playwright at the browser cached on scratch (must be set before importing playwright).
_CACHE = os.environ.get("GOALFORGE_DATA_DIR", "data")
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", os.path.join(os.path.dirname(_CACHE), "playwright"))

import pandas as pd  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

LEAGUES = ["EPL", "La_liga", "Serie_A", "Bundesliga", "Ligue_1"]
SEASONS = list(range(2018, 2026))                  # 2018/19-2025/26 (Understat year = season start)
NUM = ["time", "goals", "xG", "npg", "npxG", "assists", "xA", "shots", "key_passes",
       "xGChain", "xGBuildup"]                     # chain/buildup = playmaking involvement


def _block(route):                                 # drop images/media/fonts to save renderer memory
    route.abort() if route.request.resource_type in ("image", "media", "font") else route.continue_()


def fetch(cache: str, verbose: bool = True) -> tuple:
    rows = []
    with sync_playwright() as pw:
        # HPC fix: /dev/shm is tiny on login nodes -> Chromium renderer crashes without these flags.
        browser = pw.chromium.launch(headless=True,
                                     args=["--disable-dev-shm-usage", "--no-sandbox", "--disable-gpu"])
        for lg in LEAGUES:
            for yr in SEASONS:
                data = None
                for attempt in range(3):           # a fresh context each try avoids memory buildup
                    ctx = browser.new_context()
                    ctx.route("**/*", _block)
                    page = ctx.new_page()
                    try:
                        page.goto(f"https://understat.com/league/{lg}/{yr}", timeout=45000,
                                  wait_until="domcontentloaded")
                        page.wait_for_timeout(1200)
                        data = page.evaluate("() => (typeof playersData!=='undefined')?playersData:null")
                    except Exception as e:
                        if verbose:
                            print(f"  retry {lg}/{yr} ({attempt + 1}/3): {type(e).__name__}")
                    finally:
                        ctx.close()
                    if data:
                        break
                if not data:
                    print(f"  WARN {lg}/{yr}: no data after retries")
                    continue
                for d in data:
                    r = {"player_name": d["player_name"], "position": d["position"],
                         "team_title": d["team_title"]}
                    r.update({k: float(d[k]) for k in NUM})
                    rows.append(r)
                if verbose:
                    print(f"  {lg}/{yr}: {len(data)} players")
        browser.close()

    if not rows:
        return pd.DataFrame(), None
    allp = pd.DataFrame(rows)
    agg = allp.groupby("player_name", as_index=False).agg(
        {**{c: "sum" for c in NUM}, "team_title": "last"})
    pos = allp.groupby("player_name").position.agg(lambda s: s.mode().iloc[0]).rename("position")
    out = agg.merge(pos, on="player_name")
    out["nineties"] = out["time"] / 90.0
    os.makedirs(cache, exist_ok=True)
    dst = os.path.join(cache, "understat_players.parquet")
    out.to_parquet(dst, index=False)
    return out, dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=_CACHE)
    args = ap.parse_args()
    out, dst = fetch(args.cache)
    if dst:
        print(f"\n{len(out)} players over {SEASONS} x {len(LEAGUES)} leagues -> {dst}")
        print(out.sort_values("xA", ascending=False).head(5)
              [["player_name", "team_title", "nineties", "goals", "xG", "assists", "xA"]].round(1).to_string(index=False))
    else:
        print("\nNo data — playersData never loaded (site change?).")


if __name__ == "__main__":
    main()
