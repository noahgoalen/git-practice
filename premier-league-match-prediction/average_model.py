"""Predict scores from each team's average goals in its previous matches.

Each team has two ratings, taken from its previous matches:
    offensive rating: average goals it scored
    defensive rating: average goals it conceded (higher is worse)

Three ways of predicting a team's goals in a match are compared:
    "Scored only":       its offensive rating
    "Offence + defence": the average of its offensive rating and the
                         opponent's defensive rating
    "Home/away split":   the same average, but with ratings from matches at
                         the same venue: the home team's ratings come from
                         its home matches and the away team's from its away
                         matches, which builds in home advantage
The first two use all previous matches, home and away alike. The prediction
is rounded to the nearest whole number, and the predicted result follows
from the two rounded scores, so equal scores mean a draw.

"Previous matches" can be every earlier match in the data or only the most
recent few (counted at that venue for the home/away split). The window is
chosen on the validation seasons (by how often the result is right),
separately for each method, then the test season is predicted once with it.
Only matches before the one being predicted are ever used.

Run split_data.py (and predict_scores.py, for the comparison) first, then:
    python average_model.py
Output:  test_predictions_average.csv (next to this script) plus metrics.
"""

import csv
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
WINDOWS = [5, 10, 19, 38, None]  # last N matches; None means all of them


def load(name):
    with open(os.path.join(HERE, f"{name}.csv"), newline="",
              encoding="utf-8") as f:
        return list(csv.DictReader(f))


def round_half_up(x):
    return math.floor(x + 0.5)


def result(home_goals, away_goals):
    if home_goals > away_goals:
        return "H"
    return "D" if home_goals == away_goals else "A"


def ratings(matches, window, by_venue=False):
    """Each team's (offensive, defensive) rating before every match.

    Returns one ((home off, home def), (away off, away def)) per match, using
    only earlier matches. With by_venue, the home team is rated on its home
    matches only and the away team on its away matches only. Matches on the
    same day are rated before any of them are added to the history.

    A team with no earlier matches (at that venue, with by_venue) gets the
    league average instead: average goals per team overall, or with by_venue
    the average home or away goals as fits the rating.
    """
    scored = {}    # (team, venue) -> goals scored in each earlier match
    conceded = {}  # (team, venue) -> goals conceded in each earlier match
    goals = {"Home": [], "Away": []}  # every earlier home and away score
    by_date = {}
    for m in matches:
        by_date.setdefault(m["Date"], []).append(m)

    def average(values, fallback):
        if window is not None:
            values = values[-window:]
        return sum(values) / len(values) if values else fallback

    def league_average(*venues):
        values = [g for v in venues for g in goals[v]]
        return sum(values) / len(values) if values else 1.0

    result_ratings = []
    for day in sorted(by_date):
        for m in by_date[day]:
            rated = []
            for side, other in (("Home", "Away"), ("Away", "Home")):
                venue = side if by_venue else "All"
                key = (m[f"{side}Team"], venue)
                scored_fallback = league_average(
                    *((side,) if by_venue else ("Home", "Away")))
                conceded_fallback = league_average(
                    *((other,) if by_venue else ("Home", "Away")))
                rated.append((average(scored.get(key, []), scored_fallback),
                              average(conceded.get(key, []),
                                      conceded_fallback)))
            result_ratings.append(tuple(rated))
        for m in by_date[day]:
            home, away = int(m["HomeGoals"]), int(m["AwayGoals"])
            for team, venue, goals_for, goals_against in (
                    (m["HomeTeam"], "Home", home, away),
                    (m["AwayTeam"], "Away", away, home)):
                for key in ((team, "All"), (team, venue)):
                    scored.setdefault(key, []).append(goals_for)
                    conceded.setdefault(key, []).append(goals_against)
            goals["Home"].append(home)
            goals["Away"].append(away)
    return result_ratings


def scored_only(home, away):
    return home[0], away[0]


def offence_defence(home, away):
    return (home[0] + away[1]) / 2, (away[0] + home[1]) / 2


# name -> (how ratings become goals, whether ratings are split by venue)
METHODS = {
    "Scored only": (scored_only, False),
    "Offence + defence": (offence_defence, False),
    "Home/away split": (offence_defence, True),
}
SAVED_METHOD = "Home/away split"  # written to test_predictions_average.csv


def predict(matches, window, method):
    """(match, predicted home goals, predicted away goals) before rounding."""
    formula, by_venue = method
    return [(m, *formula(home, away)) for m, (home, away)
            in zip(matches, ratings(matches, window, by_venue))]


def metrics(rows):
    """rows: (match, predicted home goals, predicted away goals) as numbers."""
    n = len(rows)
    abs_err = sq_err = total_err = diff_err = exact = correct = 0.0
    draws = 0
    for m, ph, pa in rows:
        hg, ag = int(m["HomeGoals"]), int(m["AwayGoals"])
        abs_err += abs(ph - hg) + abs(pa - ag)
        sq_err += (ph - hg) ** 2 + (pa - ag) ** 2
        total_err += abs(ph + pa - hg - ag)
        diff_err += abs(ph - pa - (hg - ag))
        exact += (ph, pa) == (hg, ag)
        correct += result(ph, pa) == m["Result"]
        draws += ph == pa
    return {
        "goals_mae": abs_err / (2 * n),
        "goals_rmse": math.sqrt(sq_err / (2 * n)),
        "total_goals_mae": total_err / n,
        "goal_diff_mae": diff_err / n,
        "exact_score": exact / n,
        "result_accuracy": correct / n,
        "draws": draws,
    }


def rounded(predictions):
    return [(m, round_half_up(h), round_half_up(a)) for m, h, a in predictions]


def print_table(columns):
    labels = [
        ("goals_mae", "Goals MAE per team", "{:.3f}"),
        ("goals_rmse", "Goals RMSE per team", "{:.3f}"),
        ("total_goals_mae", "Total goals MAE per match", "{:.3f}"),
        ("goal_diff_mae", "Goal difference MAE per match", "{:.3f}"),
        ("exact_score", "Exact score correct", "{:.1%}"),
        ("result_accuracy", "Result (H/D/A) correct", "{:.1%}"),
        ("draws", "Draws predicted", "{}"),
    ]
    names = list(columns)
    print(f"{'':32}" + "".join(f"{n:>19}" for n in names))
    for key, label, fmt in labels:
        print(f"{label:32}" + "".join(f"{fmt.format(columns[n][key]):>19}"
                                      for n in names))


def main():
    train, validation, test = (load(n) for n in
                               ("train", "validation", "test"))
    matches = train + validation + test
    start_val, start_test = len(train), len(train) + len(validation)

    columns = {}
    unrounded = {}
    for name, method in METHODS.items():
        print(f"{name}: choosing the window on validation (result correct):")
        best = None
        for window in WINDOWS:
            rows = rounded(predict(matches, window, method))
            accuracy = metrics(rows[start_val:start_test])["result_accuracy"]
            label = "all" if window is None else f"last {window}"
            print(f"  {label:>8} matches: {accuracy:.1%}")
            if best is None or accuracy > best[0]:
                best = (accuracy, window, label)
        _, window, label = best
        print(f"Chosen: {label} matches\n")
        raw = predict(matches, window, method)[start_test:]
        columns[name] = metrics(rounded(raw))
        unrounded[name] = metrics(raw)
        if name == SAVED_METHOD:
            chosen_window = window

    # The Poisson model's predicted scores, if predict_scores.py has run.
    poisson_path = os.path.join(HERE, "test_predictions.csv")
    if os.path.exists(poisson_path):
        poisson = load("test_predictions")
        columns["Poisson"] = metrics(
            [(m, int(m["PredHomeGoals"]), int(m["PredAwayGoals"]))
             for m in poisson])

    columns["Always 1-0"] = metrics([(m, 1, 0) for m in test])

    print(f"Test season {test[0]['Season']} ({len(test)} matches, "
          f"{sum(m['Result'] == 'D' for m in test)} actual draws), "
          "predicted scores:")
    print_table(columns)
    print("\nBefore rounding, goals MAE / RMSE per team:")
    for name, values in unrounded.items():
        print(f"  {name}: {values['goals_mae']:.3f} / "
              f"{values['goals_rmse']:.3f}")

    # Write one method's predictions with the ratings behind them.
    out = os.path.join(HERE, "test_predictions_average.csv")
    formula, by_venue = METHODS[SAVED_METHOD]
    test_ratings = ratings(matches, chosen_window, by_venue)[start_test:]
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Season", "Date", "HomeTeam", "AwayTeam",
                         "HomeGoals", "AwayGoals", "Result",
                         "HomeOffence", "HomeDefence",
                         "AwayOffence", "AwayDefence",
                         "ExpHomeGoals", "ExpAwayGoals",
                         "PredHomeGoals", "PredAwayGoals", "PredResult"])
        for m, (home, away) in zip(test, test_ratings):
            h, a = formula(home, away)
            ph, pa = round_half_up(h), round_half_up(a)
            writer.writerow([m["Season"], m["Date"], m["HomeTeam"],
                             m["AwayTeam"], m["HomeGoals"], m["AwayGoals"],
                             m["Result"], *(f"{r:.3f}" for r in home + away),
                             f"{h:.3f}", f"{a:.3f}", ph, pa, result(ph, pa)])
    print(f"Wrote {SAVED_METHOD} predictions to {out}")


if __name__ == "__main__":
    main()
