"""Compare every model over several test seasons instead of just one.

A single test season of 380 matches is too small to tell models apart: a
result accuracy measured on it is only good to about +/- 5 percentage
points (95% interval). So this repeats the train / validation / test set-up
with each of the later seasons in turn as the test season:

    test season S:  settings chosen on the 2 seasons before S, then S is
                    predicted with them, using only matches played earlier

The last fold (test 2025-26, validation 2023-24 and 2024-25) is exactly the
split in train.csv / validation.csv / test.csv. The first fold needs a
season of history before its validation seasons, so the test seasons run
from 2019-20 to 2025-26.

Each model chooses its settings the same way as in its own script: the
average models by validation goals RMSE before rounding, the Poisson model
by validation log-likelihood of the goals.

Run download_results.py first, then:  python evaluate.py
It takes several minutes.
"""

import math

import average_model as avg
import predict_scores as poisson

VALIDATION_SEASONS = 2
FIRST_TEST = 3  # index of the first test season: one history + 2 validation


def folds(matches):
    """(validation indices, test indices, test season) per test season."""
    seasons = sorted({m["Season"] for m in matches})
    rows = {s: [i for i, m in enumerate(matches) if m["Season"] == s]
            for s in seasons}
    for k in range(FIRST_TEST, len(seasons)):
        validation = [i for s in seasons[k - VALIDATION_SEASONS:k]
                      for i in rows[s]]
        yield validation, rows[seasons[k]], seasons[k]


def average_model_predictions(matches, fold_list):
    """Test-season predictions per average method, one fold at a time."""
    first_scored = matches[fold_list[0][0][0]]["Date"]
    results = {}
    for name in avg.METHODS:
        print(f"  {name}")
        by_setting = avg.predict_all(matches, name, first_scored)
        rows, chosen = [], []
        for validation, test, _ in fold_list:
            setting = min(by_setting, key=lambda s: avg.validation_error(
                [by_setting[s][i] for i in validation]))
            rows.append([by_setting[setting][i] for i in test])
            chosen.append(avg.setting_label(setting))
        results[name] = (rows, chosen)
    return results


def poisson_predictions(fold_list):
    """Test-season predictions of the Poisson model, one fold at a time.

    Returns (match, expected home goals, expected away goals) rows; the
    predicted score and result come from poisson.predict.
    """
    matches = poisson.load("premier_league_results")
    poisson.mark_promoted(matches)
    design = poisson.Design(matches)
    scored = list(range(fold_list[0][0][0], len(matches)))
    by_setting = {}
    for half_life in poisson.HALF_LIVES:
        for penalty in poisson.PENALTIES:
            print(f"  Poisson: half-life {half_life}, penalty {penalty}")
            expected = poisson.walk_forward(design, scored, half_life,
                                            penalty)
            by_setting[half_life, penalty] = dict(zip(scored, expected))
    rows, chosen = [], []
    for validation, test, _ in fold_list:
        setting = max(by_setting, key=lambda s: poisson.mean_log_likelihood(
            [matches[i] for i in validation],
            [by_setting[s][i] for i in validation]))
        rows.append([(matches[i], *by_setting[setting][i]) for i in test])
        chosen.append(f"{setting[0]}/{setting[1]}")
    return rows, chosen


def poisson_scores(rows):
    """Poisson rows -> its predicted scorelines, as the average models give."""
    scored = []
    for m, home_mean, away_mean in rows:
        _, _, home_goals, away_goals, _ = poisson.predict(home_mean,
                                                          away_mean)
        scored.append((m, home_goals, away_goals))
    return scored


def interval(p, n):
    """Half-width of a 95% interval for a proportion p measured on n."""
    return 1.96 * math.sqrt(p * (1 - p) / n)


def main():
    matches = avg.load("premier_league_results")
    fold_list = list(folds(matches))
    test_seasons = [season for _, _, season in fold_list]

    print("Predicting with the average models:")
    models = {name: (rows, chosen) for name, (rows, chosen)
              in average_model_predictions(matches, fold_list).items()}
    print("Predicting with the Poisson model:")
    p_rows, p_chosen = poisson_predictions(fold_list)

    rounded = {name: [avg.rounded(r) for r in rows]
               for name, (rows, _) in models.items()}
    rounded["Poisson"] = [poisson_scores(r) for r in p_rows]
    unrounded = {name: rows for name, (rows, _) in models.items()}
    unrounded["Poisson"] = p_rows
    rounded["Always 1-0"] = [[(m, 1, 0) for m, _, _ in r] for r in p_rows]
    chosen = {name: c for name, (_, c) in models.items()}
    chosen["Poisson"] = p_chosen

    n = sum(len(r) for r in p_rows)
    print("\nSettings chosen on validation for each test season "
          "(half-life, or half-life/penalty):")
    print(f"{'':20}" + "".join(f"{s:>18}" for s in test_seasons))
    for name, values in chosen.items():
        print(f"{name:20}" + "".join(f"{v:>18}" for v in values))

    print(f"\nAll {len(test_seasons)} test seasons together ({n} matches), "
          "predicted scores:")
    pooled = {name: avg.metrics([row for r in rows for row in r])
              for name, rows in rounded.items()}
    avg.print_table(pooled)

    print("\nResult correct, with 95% interval:")
    for name, values in pooled.items():
        p = values["result_accuracy"]
        print(f"  {name:20} {p:.1%} +/- {interval(p, n):.1%}")

    print("\nBefore rounding (goals MAE / RMSE per team, and results correct "
          "if the side with\nhigher expected goals is picked, so never a "
          "draw):")
    for name, rows in unrounded.items():
        values = avg.metrics([row for r in rows for row in r])
        print(f"  {name:20} {values['goals_mae']:.3f} / "
              f"{values['goals_rmse']:.3f}, {values['result_accuracy']:.1%}")

    print("\nResult correct per test season:")
    print(f"{'':20}" + "".join(f"{s:>9}" for s in test_seasons))
    for name, rows in rounded.items():
        print(f"{name:20}" + "".join(
            f"{avg.metrics(r)['result_accuracy']:>9.1%}" for r in rows))


if __name__ == "__main__":
    main()
