"""Player-layer evaluation: are the actual World-Cup scorers among the model's top-k picks?

    python scripts/evaluate_scorers.py

Uses real StatsBomb WC2022 lineups and goals: for each team in each match, predict per-player
anytime-scorer probabilities from the model and check whether each actual goal's scorer is in
the team's top-k predicted scorers (recall@k). With 11 players on the pitch, random would give
k/11; a useful player model should beat that.
"""
import numpy as np

from goalforge import inference
from goalforge.data.statsbomb import WORLD_CUP_2022, load_competition


def main():
    d = load_competition(*WORLD_CUP_2022, verbose=False)
    model = inference.load_model("model.json")
    ks = [1, 3, 5]
    hit = {k: 0 for k in ks}
    total = 0

    matches = d.matches.set_index("match_id")
    for mid, row in matches.iterrows():
        for team, opp in [(row.home_team, row.away_team), (row.away_team, row.home_team)]:
            xi = list(d.appearances[(d.appearances.match_id == mid)
                                    & (d.appearances.team == team)].player.unique())
            actual = list(d.goals[(d.goals.match_id == mid)
                                  & (d.goals.team == team)].scorer.dropna())
            if not xi or not actual or team not in model["squads"]:
                continue
            pred = inference.predict(model, team, opp, home_xi=xi,
                                     away_xi=model["squads"].get(opp, xi)[:11],
                                     neutral=True, n_sims=5000)
            ranked = [p for p, _ in pred.home_scorers]
            for s in actual:
                total += 1
                for k in ks:
                    if s in ranked[:k]:
                        hit[k] += 1

    print(f"WC2022 scorer recall@k (real lineups & goals, n={total} goals):")
    for k in ks:
        print(f"  recall@{k} = {hit[k] / total:.1%}   (random baseline ≈ {k}/11 = {k / 11:.1%})")


if __name__ == "__main__":
    main()
