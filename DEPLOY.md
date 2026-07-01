# Deploying GoalForge on Vercel

The current (statistical) model is tiny (`api/model.json` ≈ 100 KB) and inference is analytic,
so the whole app runs on Vercel: a static frontend + tiny Python serverless functions. This
mirrors the working structure of the reference project `aevum-orrin/applet-material`.

## Structure
- `public/` — static frontend (`outputDirectory` in `vercel.json`), served at `/`.
- `api/health.py`, `api/teams.py`, `api/squad.py`, `api/predict.py` — **one serverless function
  per endpoint**, hit directly at `/api/<name>`. **Python standard library only** (no
  numpy/fastapi → nothing to install, cannot fail to import). They share `api/_engine.py`
  (analytic Poisson inference: scoreline + per-player Poisson-thinning anytime probabilities)
  and `api/model.json`.
- `vercel.json` — `outputDirectory: public`; a negative-lookahead rewrite sends every non-`/api/`
  path to the frontend; `includeFiles` bundles `api/model.json` into each function.
- `requirements.txt` — empty (the functions need no third-party deps).

## Connect (you handle the GitHub↔Vercel link)
1. Push to GitHub (done).
2. vercel.com → New Project → import `GoalForge-AI`. Framework preset **Other**; root = repo root.
3. Deploy → public URL. Every `git push` redeploys. Check `<url>/api/health` first — it returns
   `{"status":"ok","teams":32}` when the model is bundled.

## Updating the model
```bash
python scripts/train.py            # retrain on Great Lakes
python scripts/export_model.py     # writes api/model.json
git add api/model.json && git commit -m "update model" && git push
```

## Endgame (after Phase 3)
Once neural models arrive, the heavy model moves to a **Hugging Face Space** (torch + GPU);
Vercel keeps serving the frontend, calling the HF backend. See docs/workflow.md.
