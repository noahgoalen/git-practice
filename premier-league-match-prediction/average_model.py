"""Predict scores from each team's average goals in its previous matches.

For each match, a team's predicted goals is the average number of goals it
scored in its previous matches (home and away alike), rounded to the nearest
whole number. The predicted result follows from the two rounded scores, so
equal scores mean a draw.

"Previous matches" can be every earlier match in the data or only the most
recent few. The window is chosen on the validation seasons (by how often the
result is right), then the test season is predicted once with it. Only matches
before the one being predicted are ever used.

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


def predict(matches, window):
    """Average-goals prediction for every match, using only earlier matches.

    Matches on the same day are predicted before any of them are added to the
    history. A team with no earlier matches gets the average goals per team
    of all earlier matches.
    """
    history = {}  # team -> goals scored in each earlier match, oldest first
    all_goals = []
    predictions = []
    by_date = {}
    for m in matches:
        by_date.setdefault(m["Date"], []).append(m)

    for day in sorted(by_date):
        for m in by_date[day]:
            fallback = sum(all_goals) / len(all_goals) if all_goals else 1.0
            averages = []
            for team in (m["HomeTeam"], m["AwayTeam"]):
                goals = history.get(team, [])
                if window is not None:
                    goals = goals[-window:]
                averages.append(sum(goals) / len(goals) if goals else fallback)
            predictions.append((m, averages[0], averages[1]))
        for m in by_date[day]:
            for team, goals in ((m["HomeTeam"], m["HomeGoals"]),
                                (m["AwayTeam"], m["AwayGoals"])):
                history.setdefault(team, []).append(int(goals))
                all_goals.append(int(goals))
    return predictions


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
    print(f"{'':32}" + "".join(f"{n:>14}" for n in names))
    for key, label, fmt in labels:
        print(f"{label:32}" + "".join(f"{fmt.format(columns[n][key]):>14}"
                                      for n in names))


def main():
    train, validation, test = (load(n) for n in
                               ("train", "validation", "test"))
    matches = train + validation + test
    start_val, start_test = len(train), len(train) + len(validation)

    print("Choosing the window on validation (result correct):")
    best = None
    for window in WINDOWS:
        rows = rounded(predict(matches, window))[start_val:start_test]
        accuracy = metrics(rows)["result_accuracy"]
        label = "all" if window is None else f"last {window}"
        print(f"  {label:>8} matches: {accuracy:.1%}")
        if best is None or accuracy > best[0]:
            best = (accuracy, window, label)
    _, window, label = best
    print(f"Chosen: {label} matches\n")

    raw = predict(matches, window)[start_test:]
    columns = {"Average": metrics(rounded(raw))}

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
    unrounded = metrics(raw)
    print(f"\nBefore rounding, the averages have goals MAE "
          f"{unrounded['goals_mae']:.3f} and RMSE "
          f"{unrounded['goals_rmse']:.3f} per team.")

    out = os.path.join(HERE, "test_predictions_average.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Season", "Date", "HomeTeam", "AwayTeam",
                         "HomeGoals", "AwayGoals", "Result",
                         "AvgHomeGoals", "AvgAwayGoals",
                         "PredHomeGoals", "PredAwayGoals", "PredResult"])
        for m, h, a in raw:
            ph, pa = round_half_up(h), round_half_up(a)
            writer.writerow([m["Season"], m["Date"], m["HomeTeam"],
                             m["AwayTeam"], m["HomeGoals"], m["AwayGoals"],
                             m["Result"], f"{h:.3f}", f"{a:.3f}", ph, pa,
                             result(ph, pa)])
    print(f"Wrote predictions to {out}")


if __name__ == "__main__":
    main()
