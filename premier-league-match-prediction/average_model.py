"""Predict scores from each team's average goals in its previous matches.

Each team has two ratings, taken from its previous matches:
    offensive rating: average goals it scored
    defensive rating: average goals it conceded (higher is worse)

Five ways of predicting a team's goals in a match are compared:
    "Scored only":       its offensive rating
    "Offence + defence": the average of its offensive rating and the
                         opponent's defensive rating
    "Home/away split":   the same average, but with ratings from matches at
                         the same venue: the home team's ratings come from
                         its home matches and the away team's from its away
                         matches, which builds in home advantage
    "Opponent-adjusted": the home/away split, with each team's ratings
                         corrected for the strength of the opponents it
                         faced (see opponent_adjusted)
    "Shots on target":   the home/away split, but with ratings from shots on
                         target taken and faced rather than goals. The
                         expected shots on target are turned into goals with
                         the league's goals-per-shot-on-target rate at that
                         venue. Shots on target are a steadier measure of how
                         well a team plays than goals, which are noisier
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


OTHER_VENUE = {"Home": "Away", "Away": "Home", "All": "All"}
ADJUST_ROUNDS = 20  # enough for the opponent adjustment to settle


def opponent_adjusted(history, window, scored_base, conceded_base):
    """Offensive and defensive ratings adjusted for the opponents faced.

    Scoring 3 against a leaky defence is worth less than scoring 3 against a
    tight one. So each goal tally is corrected by how far the opponent's
    defensive rating is from the league average (and each goals-conceded
    tally by the opponent's offensive rating), and every team is re-rated
    with the corrected numbers. The opponents' ratings depend on their own
    opponents, so this is repeated until the ratings settle.

    history: (team, venue) -> [(goals for, goals against, opponent key)]
    scored_base / conceded_base: venue -> league average goals scored /
    conceded at that venue.
    """
    recent = {key: matches[-window:] if window is not None else matches
              for key, matches in history.items()}
    offence = {k: sum(gf for gf, _, _ in ms) / len(ms)
               for k, ms in recent.items()}
    defence = {k: sum(ga for _, ga, _ in ms) / len(ms)
               for k, ms in recent.items()}
    for _ in range(ADJUST_ROUNDS):
        offence, defence = (
            {(team, venue): sum(gf - (defence[opp] - scored_base[venue])
                                for gf, _, opp in ms) / len(ms)
             for (team, venue), ms in recent.items()},
            {(team, venue): sum(ga - (offence[opp] - conceded_base[venue])
                                for _, ga, opp in ms) / len(ms)
             for (team, venue), ms in recent.items()})
    return offence, defence


def ratings(matches, window, by_venue=False, adjust_from=None, stat="Goals"):
    """Each team's (offensive, defensive) rating before every match.

    stat is what is averaged: "Goals" (goals scored and conceded) or
    "ShotsOnTarget" (shots on target taken and faced).

    Returns one ((home off, home def), (away off, away def)) per match, using
    only earlier matches. With by_venue, the home team is rated on its home
    matches only and the away team on its away matches only. Matches on the
    same day are rated before any of them are added to the history.

    From the date adjust_from onwards (if given) the ratings are adjusted for
    opponent strength; before that, plain averages are returned, which saves
    time on matches that are never scored.

    A team with no earlier matches (at that venue, with by_venue) gets the
    league average instead: average goals per team overall, or with by_venue
    the average home or away goals as fits the rating.
    """
    # (team, venue) -> (goals for, goals against, opponent key) per earlier
    # match, oldest first. Venue is "Home", "Away" or "All".
    history = {}
    goals = {"Home": [], "Away": []}  # every earlier home and away score
    by_date = {}
    for m in matches:
        by_date.setdefault(m["Date"], []).append(m)

    def average(values):
        return sum(values) / len(values)

    result_ratings = []
    for day in sorted(by_date):
        all_goals = goals["Home"] + goals["Away"]
        league = {"Home": average(goals["Home"]) if all_goals else 1.0,
                  "Away": average(goals["Away"]) if all_goals else 1.0,
                  "All": average(all_goals) if all_goals else 1.0}
        conceded_base = {v: league[OTHER_VENUE[v]] for v in league}

        venues = ("Home", "Away") if by_venue else ("All",)
        venue_history = {k: v for k, v in history.items() if k[1] in venues}
        if adjust_from is not None and day >= adjust_from:
            offence, defence = opponent_adjusted(venue_history, window,
                                                 league, conceded_base)
        else:
            recent = {k: ms[-window:] if window is not None else ms
                      for k, ms in venue_history.items()}
            offence = {k: average([gf for gf, _, _ in ms])
                       for k, ms in recent.items()}
            defence = {k: average([ga for _, ga, _ in ms])
                       for k, ms in recent.items()}

        for m in by_date[day]:
            rated = []
            for side in ("Home", "Away"):
                key = (m[f"{side}Team"], side if by_venue else "All")
                venue = key[1]
                rated.append((offence.get(key, league[venue]),
                              defence.get(key, conceded_base[venue])))
            result_ratings.append(tuple(rated))

        for m in by_date[day]:
            home, away = int(m[f"Home{stat}"]), int(m[f"Away{stat}"])
            for team, opponent, venue, goals_for, goals_against in (
                    (m["HomeTeam"], m["AwayTeam"], "Home", home, away),
                    (m["AwayTeam"], m["HomeTeam"], "Away", away, home)):
                history.setdefault((team, "All"), []).append(
                    (goals_for, goals_against, (opponent, "All")))
                history.setdefault((team, venue), []).append(
                    (goals_for, goals_against,
                     (opponent, OTHER_VENUE[venue])))
            goals["Home"].append(home)
            goals["Away"].append(away)
    return result_ratings


def scored_only(home, away):
    return home[0], away[0]


def offence_defence(home, away):
    return (home[0] + away[1]) / 2, (away[0] + home[1]) / 2


# name -> (how ratings become a prediction, whether ratings are split by
#          venue, whether they are adjusted for opponent strength,
#          what they are averaged from)
METHODS = {
    "Scored only": (scored_only, False, False, "Goals"),
    "Offence + defence": (offence_defence, False, False, "Goals"),
    "Home/away split": (offence_defence, True, False, "Goals"),
    "Opponent-adjusted": (offence_defence, True, True, "Goals"),
    "Shots on target": (offence_defence, True, False, "ShotsOnTarget"),
}
SAVED_METHOD = "Shots on target"  # written to test_predictions_average.csv
DEFAULT_CONVERSION = 0.32  # goals per shot on target, before any data


def conversion_rates(matches):
    """League goals per shot on target, at home and away, before each match.

    Only matches on earlier days count. Returns (home rate, away rate) per
    match.
    """
    totals = {"Home": [0, 0], "Away": [0, 0]}  # venue -> [goals, on target]
    rates = {}
    for day in sorted({m["Date"] for m in matches}):
        rates[day] = tuple(goals / shots if shots else DEFAULT_CONVERSION
                           for goals, shots in totals.values())
        for m in matches:
            if m["Date"] == day:
                for venue, total in totals.items():
                    total[0] += int(m[f"{venue}Goals"])
                    total[1] += int(m[f"{venue}ShotsOnTarget"])
    return [rates[m["Date"]] for m in matches]


def predict(matches, window, method, adjust_from):
    """(match, predicted home goals, predicted away goals) before rounding."""
    formula, by_venue, adjusted, stat = method
    rated = ratings(matches, window, by_venue,
                    adjust_from if adjusted else None, stat)
    predictions = [formula(home, away) for home, away in rated]
    if stat == "ShotsOnTarget":
        # Expected shots on target -> expected goals.
        predictions = [(h * home_rate, a * away_rate) for (h, a),
                       (home_rate, away_rate)
                       in zip(predictions, conversion_rates(matches))]
    return [(m, h, a) for m, (h, a) in zip(matches, predictions)]


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
    adjust_from = validation[0]["Date"]  # no need to adjust earlier matches

    columns = {}
    unrounded = {}
    for name, method in METHODS.items():
        print(f"{name}: choosing the window on validation (result correct):")
        best = None
        for window in WINDOWS:
            rows = rounded(predict(matches, window, method, adjust_from))
            accuracy = metrics(rows[start_val:start_test])["result_accuracy"]
            label = "all" if window is None else f"last {window}"
            print(f"  {label:>8} matches: {accuracy:.1%}")
            if best is None or accuracy > best[0]:
                best = (accuracy, window, label)
        _, window, label = best
        print(f"Chosen: {label} matches\n")
        raw = predict(matches, window, method, adjust_from)[start_test:]
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
    print("\nBefore rounding (goals MAE / RMSE per team, and results correct "
          "if the side with\nhigher expected goals is picked, so never a draw):")
    for name, values in unrounded.items():
        print(f"  {name}: {values['goals_mae']:.3f} / "
              f"{values['goals_rmse']:.3f}, {values['result_accuracy']:.1%}")

    # Write one method's predictions with the ratings behind them.
    out = os.path.join(HERE, "test_predictions_average.csv")
    method = METHODS[SAVED_METHOD]
    _, by_venue, adjusted, stat = method
    test_ratings = ratings(matches, chosen_window, by_venue,
                           adjust_from if adjusted else None,
                           stat)[start_test:]
    test_predictions = predict(matches, chosen_window, method,
                               adjust_from)[start_test:]
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Season", "Date", "HomeTeam", "AwayTeam",
                         "HomeGoals", "AwayGoals", "Result",
                         f"HomeOffence{stat}", f"HomeDefence{stat}",
                         f"AwayOffence{stat}", f"AwayDefence{stat}",
                         "ExpHomeGoals", "ExpAwayGoals",
                         "PredHomeGoals", "PredAwayGoals", "PredResult"])
        for (m, h, a), (home, away) in zip(test_predictions, test_ratings):
            ph, pa = round_half_up(h), round_half_up(a)
            writer.writerow([m["Season"], m["Date"], m["HomeTeam"],
                             m["AwayTeam"], m["HomeGoals"], m["AwayGoals"],
                             m["Result"], *(f"{r:.3f}" for r in home + away),
                             f"{h:.3f}", f"{a:.3f}", ph, pa, result(ph, pa)])
    print(f"Wrote {SAVED_METHOD} predictions to {out}")


if __name__ == "__main__":
    main()
