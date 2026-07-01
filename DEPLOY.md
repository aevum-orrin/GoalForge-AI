# Deploying GoalForge on Vercel

The current (statistical) model is tiny (`api/model.json` ≈ 230 KB — 48 teams, ~1250 players)
and inference is analytic, so the whole app runs on Vercel: a static frontend + tiny Python
serverless functions. This mirrors the working reference project `aevum-orrin/applet-material`.

## Structure
- `public/` — static frontend (`outputDirectory` in `vercel.json`), served at `/`.
- `api/health.py`, `api/teams.py`, `api/squad.py`, `api/predict.py` — **one serverless function
  per endpoint**, hit directly at `/api/<name>`. **Python standard library only** (no
  numpy/fastapi → nothing to install, cannot fail to import). They share `api/_engine.py`
  (analytic Poisson inference: scoreline + per-player Poisson-thinning anytime probabilities)
  and `api/model.json`.
- `vercel.json` — `outputDirectory: public`; a negative-lookahead rewrite sends every non-`/api/`
  path to the frontend; `includeFiles: "api/**"` bundles the whole `api/` dir (the shared
  `_engine.py` **and** `model.json`) into each function's lambda. Each `api/*.py` is a separate
  lambda, so every one needs `_engine.py` shipped with it; each handler also does
  `sys.path.insert(0, dirname(__file__))` so `import _engine` resolves inside the lambda.
- `requirements.txt` — empty (the functions need no third-party deps).
- `.vercelignore` — **excludes `pyproject.toml`** (and `src/`). This matters: newer Vercel builders
  scan `pyproject.toml`, and ours lists `fastapi` (optional-deps) while the real FastAPI app lives in
  `src/goalforge/api/app.py`. If Vercel sees that, it switches to "FastAPI backend" mode, hunts for a
  single ASGI entrypoint, finds only the `handler` functions, and fails the build with
  *"No FastAPI entrypoint found"*. Ignoring `pyproject.toml`/`src/` keeps it on the classic
  static-site + per-file Python serverless-functions path (what applet-material uses).

## Connect (you handle the GitHub↔Vercel link)
1. Push to GitHub (done).
2. vercel.com → New Project → import `GoalForge-AI`. Framework preset **Other**; root = repo root.
3. Deploy → public URL. Every `git push` redeploys. Check `<url>/api/health` first — it returns
   `{"status":"ok","teams":48}` when the model is bundled.

## Updating the model / squads (real 2026 World Cup)
```bash
python scripts/train.py               # (re)fit the Dixon-Coles team layer -> models/agent_intl.pkl
python scripts/scrape_wc2026.py       # scrape the 48 real 2026 squads (Wikipedia) -> scratch cache
python scripts/fetch_understat.py     # club xG/xA per player (5 big leagues; free Understat mirror)
python scripts/build_player_form.py   # name-match club form onto the 2026 squads -> player_form
python scripts/build_wc2026_model.py  # combine -> api/model.json (xA assists + club-enriched scoring)
git add api/model.json && git commit -m "update 2026 model" && git push
```
`build_wc2026_model.py` reuses the validated team checkpoint for the scoreline, and for the ~1/3
of squad players with top-5-league data blends recent club **xG** (scorer) and **xA** (assister,
the validated chance-creation signal); the rest fall back to international goals/caps + a position
prior. See docs/evaluation.md §8-11 for the honest evaluation behind these choices.

## Endgame (after Phase 3)
Once neural models arrive, the heavy model moves to a **Hugging Face Space** (torch + GPU);
Vercel keeps serving the frontend, calling the HF backend. See docs/workflow.md.
