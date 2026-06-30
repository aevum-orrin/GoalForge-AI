# scripts/

Thin CLI entry points that wrap the `goalforge` package. Planned (see
[../docs/workflow.md](../docs/workflow.md)); not implemented yet:

- `download_data.py` — fetch & cache provider data into `data/`
- `build_features.py` — assemble the player / coach / team feature tables
- `train.py` — fit the scoreline and player goal/assist models
- `predict_match.py` — two lineups -> scoreline + scorer / assist probabilities
- `backtest.py` — walk-forward evaluation

Prefer running package modules directly (e.g. `python -m goalforge.models.train`);
these scripts are convenience shims.
