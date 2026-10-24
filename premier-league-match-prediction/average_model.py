"""Predict scores from each team's average goals in its previous matches.

Each team has two ratings, taken from its previous matches:
    offensive rating: average goals it scored
    defensive rating: average goals it conceded (higher is worse)

Six ways of predicting a team's goals in a match are compared:
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
    "Goals + shots":     a weighted average of the home/away split and shots
                         on target predictions, with the weight chosen on
                         validation
The first two use all previous matches, home and away alike. The prediction
is rounded to the nearest whole number, and the predicted result follows
from the two rounded scores, so equal scores mean a draw.

Ratings are weighted averages: every earlier match counts, but its weight
halves every "half-life" days, so recent form matters most. A newly promoted
team is rated afresh: its matches from before promotion are ignored, and it
starts from the average record of earlier promoted teams in their first
season, counted as PROMOTED_PRIOR_MATCHES matches. Its own results take over
as the season goes on.

The half-life (and, for "Goals + shots", the weight) is chosen on the
validation seasons, separately for each method, by the error in predicted
goals before rounding (RMSE). That uses the size of every miss, so it is
much steadier than how often the rounded result is right. The test season is
then predicted once with the chosen settings. Only matches before the one
being predicted are ever used.

Run split_data.py (and predict_scores.py, for the comparison) first, then:
    python average_model.py
Output:  test_predictions_average.csv (next to this script) plus metrics.
"""

import csv
import math
import os
from datetime import date

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HALF_LIVES = [60, 120, 240, 365, 730, None]  # days; None means no decay
BLENDS = [0, 0.25, 0.5, 0.75, 1]  # share of goals in "Goals + shots"
PROMOTED_PRIOR_MATCHES = 2  # weight of a promoted team's starting rating, in matches
ADJUST_ROUNDS = 20  # enough for the opponent adjustment to settle
DEFAULT_CONVERSION = 0.32  # goals per shot on target, before any data


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


def promoted_seasons(matches):
    """team -> indices of the seasons it was newly promoted into.

    A team is promoted if it did not play in the previous season. The first
    season in the data has no previous season, so nobody is promoted there.
    """
    seasons = sorted({m["Season"] for m in matches})
    teams = [{t for m in matches if m["Season"] == s
              for t in (m["HomeTeam"], m["AwayTeam"])} for s in seasons]
    promoted = {}
    for k in range(1, len(seasons)):
        for team in teams[k] - teams[k - 1]:
            promoted.setdefault(team, []).append(k)
    return seasons, promoted


def ratings(matches, half_life, by_venue=False, adjusted=False,
            stat="Goals", rate_from=None):
    """Each team's (offensive, defensive) rating before every match.

    Returns one ((home off, home def), (away off, away def)) per match, using
    only matches on earlier days.

    half_life: days for a match's weight to halve; None weights all equally.
    by_venue: rate the home team on its home matches only and the away team
        on its away matches only.
    adjusted: adjust the ratings for opponent strength (opponent_adjusted).
    stat: what is averaged, "Goals" or "ShotsOnTarget" (taken and faced).
    rate_from: date from which to compute ratings. Earlier matches get the
        league average, which saves time on matches that are never scored.

    A team with no usable earlier matches (at that venue, with by_venue) gets
    the league average, or the promoted-team starting rating if it has just
    been promoted.
    """
    seasons, promoted = promoted_seasons(matches)
    season_index = {s: k for k, s in enumerate(seasons)}
    teams = sorted({t for m in matches for t in (m["HomeTeam"], m["AwayTeam"])})
    team_index = {t: i for i, t in enumerate(teams)}
    n_keys = len(teams) * (2 if by_venue else 1)

    def key(team, venue):  # venue 0 = home, 1 = away
        return team_index[team] * 2 + venue if by_venue else team_index[team]

    # The venue each key's ratings come from (2 = home and away together).
    key_venue = (np.tile([0, 1], len(teams)) if by_venue
                 else np.full(n_keys, 2))
    key_team = np.arange(n_keys) // (2 if by_venue else 1)

    # One record per team per earlier match: the key rated, the opponent's
    # key, the stat for and against, the day and the season.
    records = {name: [] for name in ("key", "opp", "for", "against", "day",
                                     "season", "promoted")}
    totals = np.zeros((2, 2))  # [home, away] x [stat sum, matches]
    by_date = {}
    for m in matches:
        by_date.setdefault(m["Date"], []).append(m)

    result_ratings = []
    for day in sorted(by_date):
        today = date.fromisoformat(day).toordinal()
        season = season_index[by_date[day][0]["Season"]]
        league = np.where(totals[:, 1] > 0,
                          totals[:, 0] / np.maximum(totals[:, 1], 1), 1.0)
        # Average stat for and against at each venue code 0, 1, 2.
        scored_base = np.array([league[0], league[1], league.mean()])
        conceded_base = np.array([league[1], league[0], league.mean()])

        if rate_from is not None and day < rate_from or not records["key"]:
            offence = scored_base[key_venue]
            defence = conceded_base[key_venue]
        else:
            r = {name: np.array(values) for name, values in records.items()}
            weights = (np.ones(len(r["key"])) if half_life is None
                       else 0.5 ** ((today - r["day"]) / half_life))
            # A team's matches before its latest promotion are ignored.
            latest = np.array([max([k for k in promoted.get(t, [])
                                    if k <= season], default=-1)
                               for t in teams])
            weights[r["season"] < latest[key_team[r["key"]]]] = 0

            # Starting rating for teams promoted this season: what earlier
            # promoted teams averaged in their first season, by venue.
            prior_offence = scored_base[key_venue].copy()
            prior_defence = conceded_base[key_venue].copy()
            earlier = r["promoted"] & (r["season"] < season)
            for venue in (0, 1, 2):
                mask = earlier & (key_venue[r["key"]] == venue)
                if mask.any():
                    prior_offence[key_venue == venue] = r["for"][mask].mean()
                    prior_defence[key_venue == venue] = (
                        r["against"][mask].mean())
            prior_weight = np.where(latest[key_team] == season,
                                    PROMOTED_PRIOR_MATCHES, 0.0)

            total_weight = np.bincount(r["key"], weights, n_keys)

            def average(values, prior, fallback):
                sums = np.bincount(r["key"], weights * values, n_keys)
                total = total_weight + prior_weight
                return np.where(total > 0,
                                (sums + prior_weight * prior)
                                / np.maximum(total, 1e-12), fallback)

            fallback_offence = scored_base[key_venue]
            fallback_defence = conceded_base[key_venue]
            offence = average(r["for"], prior_offence, fallback_offence)
            defence = average(r["against"], prior_defence, fallback_defence)
            if adjusted:
                offence, defence = opponent_adjusted(
                    r, average, offence, defence, prior_offence,
                    prior_defence, fallback_offence, fallback_defence,
                    scored_base[key_venue], conceded_base[key_venue])

        for m in by_date[day]:
            home, away = key(m["HomeTeam"], 0), key(m["AwayTeam"], 1)
            result_ratings.append(((offence[home], defence[home]),
                                   (offence[away], defence[away])))

        for m in by_date[day]:
            home, away = int(m[f"Home{stat}"]), int(m[f"Away{stat}"])
            for team, opponent, venue, stat_for, stat_against in (
                    (m["HomeTeam"], m["AwayTeam"], 0, home, away),
                    (m["AwayTeam"], m["HomeTeam"], 1, away, home)):
                records["key"].append(key(team, venue))
                records["opp"].append(key(opponent, 1 - venue))
                records["for"].append(stat_for)
                records["against"].append(stat_against)
                records["day"].append(today)
                records["season"].append(season)
                records["promoted"].append(season in promoted.get(team, []))
            totals[0] += (home, 1)
            totals[1] += (away, 1)
    return result_ratings


def opponent_adjusted(r, average, offence, defence, prior_offence,
                      prior_defence, fallback_offence, fallback_defence,
                      key_scored_base, key_conceded_base):
    """Offensive and defensive ratings adjusted for the opponents faced.

    Scoring 3 against a leaky defence is worth less than scoring 3 against a
    tight one. So each tally for is corrected by how far the opponent's
    defensive rating is from the league average (and each tally against by
    the opponent's offensive rating), and every team is re-rated with the
    corrected numbers. The opponents' ratings depend on their own opponents,
    so this is repeated until the ratings settle.
    """
    base_for = key_scored_base[r["key"]]
    base_against = key_conceded_base[r["key"]]
    for _ in range(ADJUST_ROUNDS):
        offence, defence = (
            average(r["for"] - (defence[r["opp"]] - base_for),
                    prior_offence, fallback_offence),
            average(r["against"] - (offence[r["opp"]] - base_against),
                    prior_defence, fallback_defence))
    return offence, defence


def scored_only(home, away):
    return home[0], away[0]


def offence_defence(home, away):
    return (home[0] + away[1]) / 2, (away[0] + home[1]) / 2


# name -> how ratings become a prediction, whether ratings are split by
# venue, whether they are adjusted for opponent strength, and what they are
# averaged from. "blend" methods mix the predictions of two other methods.
METHODS = {
    "Scored only": dict(formula=scored_only, by_venue=False,
                        adjusted=False, stat="Goals"),
    "Offence + defence": dict(formula=offence_defence, by_venue=False,
                              adjusted=False, stat="Goals"),
    "Home/away split": dict(formula=offence_defence, by_venue=True,
                            adjusted=False, stat="Goals"),
    "Opponent-adjusted": dict(formula=offence_defence, by_venue=True,
                              adjusted=True, stat="Goals"),
    "Shots on target": dict(formula=offence_defence, by_venue=True,
                            adjusted=False, stat="ShotsOnTarget"),
    "Goals + shots": dict(blend=("Home/away split", "Shots on target")),
}
SAVED_METHOD = "Goals + shots"  # written to test_predictions_average.csv


def conversion_rates(matches):
    """League goals per shot on target, at home and away, before each match.

    Only matches on earlier days count. Returns (home rate, away rate) per
    match.
    """
    totals = {"Home": [0, 0], "Away": [0, 0]}  # venue -> [goals, on target]
    by_date = {}
    for m in matches:
        by_date.setdefault(m["Date"], []).append(m)
    rates = {}
    for day in sorted(by_date):
        rates[day] = tuple(goals / shots if shots else DEFAULT_CONVERSION
                           for goals, shots in totals.values())
        for m in by_date[day]:
            for venue, total in totals.items():
                total[0] += int(m[f"{venue}Goals"])
                total[1] += int(m[f"{venue}ShotsOnTarget"])
    return [rates[m["Date"]] for m in matches]


def predict(matches, method, half_life, rate_from):
    """(match, predicted home goals, predicted away goals) before rounding.

    Not for blend methods; see predict_all.
    """
    rated = ratings(matches, half_life, method["by_venue"],
                    method["adjusted"], method["stat"], rate_from)
    predictions = [method["formula"](home, away) for home, away in rated]
    if method["stat"] == "ShotsOnTarget":
        # Expected shots on target -> expected goals.
        predictions = [(h * home_rate, a * away_rate) for (h, a),
                       (home_rate, away_rate)
                       in zip(predictions, conversion_rates(matches))]
    return [(m, h, a) for m, (h, a) in zip(matches, predictions)]


def predict_all(matches, name, rate_from):
    """Predictions for every setting of a method: {setting: rows}.

    A setting is (half-life, blend weight); the weight is None unless the
    method is a blend.
    """
    method = METHODS[name]
    if "blend" not in method:
        return {(h, None): predict(matches, method, h, rate_from)
                for h in HALF_LIVES}
    first, second = method["blend"]
    out = {}
    for h in HALF_LIVES:
        a = predict(matches, METHODS[first], h, rate_from)
        b = predict(matches, METHODS[second], h, rate_from)
        for w in BLENDS:
            out[h, w] = [(m, w * ha + (1 - w) * hb, w * aa + (1 - w) * ab)
                         for (m, ha, aa), (_, hb, ab) in zip(a, b)]
    return out


def validation_error(rows):
    """What settings are chosen by: goals RMSE before rounding (lower wins)."""
    return metrics(rows)["goals_rmse"]


def setting_label(setting):
    half_life, blend = setting
    label = "no decay" if half_life is None else f"{half_life}d"
    return label if blend is None else f"{label}, {blend:.0%} goals"


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
    rate_from = validation[0]["Date"]  # earlier matches are never scored

    columns = {}
    unrounded = {}
    for name in METHODS:
        print(f"{name}: choosing the settings on validation "
              "(goals RMSE before rounding, lower is better):")
        by_setting = predict_all(matches, name, rate_from)
        errors = {s: validation_error(rows[start_val:start_test])
                  for s, rows in by_setting.items()}
        for s, error in errors.items():
            print(f"  {setting_label(s):>20}: {error:.4f}")
        setting = min(errors, key=errors.get)
        print(f"Chosen: {setting_label(setting)}\n")
        raw = by_setting[setting][start_test:]
        columns[name] = metrics(rounded(raw))
        unrounded[name] = metrics(raw)
        if name == SAVED_METHOD:
            saved = raw

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

    out = os.path.join(HERE, "test_predictions_average.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Season", "Date", "HomeTeam", "AwayTeam",
                         "HomeGoals", "AwayGoals", "Result",
                         "ExpHomeGoals", "ExpAwayGoals",
                         "PredHomeGoals", "PredAwayGoals", "PredResult"])
        for m, h, a in saved:
            ph, pa = round_half_up(h), round_half_up(a)
            writer.writerow([m["Season"], m["Date"], m["HomeTeam"],
                             m["AwayTeam"], m["HomeGoals"], m["AwayGoals"],
                             m["Result"], f"{h:.3f}", f"{a:.3f}", ph, pa,
                             result(ph, pa)])
    print(f"Wrote {SAVED_METHOD} predictions to {out}")


if __name__ == "__main__":
    main()
