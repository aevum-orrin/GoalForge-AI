# Deploying GoalForge on Vercel

The current (statistical) model is tiny (`api/model.json` ≈ 100 KB) and inference is pure NumPy,
so the whole app runs on Vercel: a static frontend + one small serverless function.

## Structure (standard Vercel layout)
- `public/` — static frontend, served by Vercel at `/`.
- `api/index.py` — **self-contained** serverless function (NumPy + FastAPI only; no imports from
  `src/`, no static serving). Handles `/api/*`; reads `api/model.json` beside it.
- `api/model.json` — the exported model (regenerate with `python scripts/export_model.py`).
- `vercel.json` — rewrites `/api/*` to the function; `public/` is auto-served.
- `requirements.txt` — minimal (`numpy`, `fastapi`, `uvicorn`). `.vercelignore` excludes
  everything else (src/, data/, tests/, …) so the function bundle stays tiny.

## Connect (you handle the GitHub↔Vercel link)
1. Push to GitHub (done).
2. vercel.com → New Project → import `GoalForge-AI`. Framework preset **Other**; root = repo root;
   no build/output overrides. Vercel installs `requirements.txt` and builds `api/index.py`.
3. Deploy → public URL. Every `git push` redeploys.
4. If a request fails, open the deployment's **Logs**; the full traceback tells us the cause
   (send it over and I'll fix). Locally the same app runs via
   `python -m uvicorn goalforge.api.app:app --port 8793`.

## Updating the model
```bash
python scripts/train.py            # retrain on Great Lakes
python scripts/export_model.py     # writes api/model.json
git add api/model.json && git commit -m "update model" && git push
```

## Endgame (after Phase 3)
Once neural models arrive, the heavy model moves to a **Hugging Face Space** (torch + GPU);
Vercel keeps serving the frontend, calling the HF backend. See docs/workflow.md.
