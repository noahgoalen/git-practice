"""Predict Premier League scores with a Poisson team-strength model.

Each team gets an attack rating and a defence rating. The number of goals a
team scores is modelled as Poisson with mean
    exp(base + home_advantage + attack[team] + defence[opponent]
        + promoted terms)
where a higher defence rating means the team concedes more. Newly promoted
teams get an extra learned attack/defence adjustment, because they usually
arrive with little or stale history in the league.

Fitting is weighted maximum likelihood: older matches count less (an
exponential decay with a given half-life) and team ratings are shrunk
towards average (a ridge penalty).

Predictions are made walk-forward, one week at a time: before each week the
model is refitted on every match played before it, then predicts that week's
matches. This is how it would be used during a real season, and no match is
ever predicted with knowledge of its own result or later ones.

The half-life and penalty are chosen on the validation seasons; the test
season is then predicted once with the chosen settings.

Run split_data.py first, then:  python predict_scores.py
Output:  test_predictions.csv (next to this script) plus metrics printed.
"""

import csv
import math
import os
from datetime import date

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_GOALS = 10  # scorelines up to 10-10 are enough to cover the probability
HALF_LIVES = [180, 365, 730, None]  # days; None means no decay
PENALTIES = [1, 3, 10, 30, 100]
RESULTS = ["H", "D", "A"]


def load(name):
    with open(os.path.join(HERE, f"{name}.csv"), newline="",
              encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["Date"] = date.fromisoformat(row["Date"])
        row["HomeGoals"] = int(row["HomeGoals"])
        row["AwayGoals"] = int(row["AwayGoals"])
    return rows


def mark_promoted(matches):
    """Flag teams playing their first season after promotion.

    A team is promoted if it did not play in the previous season. The first
    season in the data has no previous season, so nobody is flagged there.
    """
    teams_by_season = {}
    for m in matches:
        teams_by_season.setdefault(m["Season"], set()).update(
            (m["HomeTeam"], m["AwayTeam"]))
    seasons = sorted(teams_by_season)
    previous = {s: teams_by_season[p] for p, s in zip(seasons, seasons[1:])}
    for m in matches:
        before = previous.get(m["Season"])
        m["HomePromoted"] = before is not None and m["HomeTeam"] not in before
        m["AwayPromoted"] = before is not None and m["AwayTeam"] not in before


class Design:
    """Two rows per match (home goals, then away goals) for the Poisson fit.

    Columns: base, home advantage, promoted attack, promoted defence, then an
    attack and a defence column per team. Only team columns are penalised.
    """

    FIXED = 4

    def __init__(self, matches):
        teams = sorted({m["HomeTeam"] for m in matches}
                       | {m["AwayTeam"] for m in matches})
        self.index = {t: i for i, t in enumerate(teams)}
        n_teams = len(teams)
        self.n_params = self.FIXED + 2 * n_teams

        self.X = np.zeros((2 * len(matches), self.n_params))
        self.y = np.zeros(2 * len(matches))
        for k, m in enumerate(matches):
            for side, row in (("Home", 2 * k), ("Away", 2 * k + 1)):
                other = "Away" if side == "Home" else "Home"
                self.X[row] = self._row(m[f"{side}Team"], m[f"{other}Team"],
                                        side == "Home", m[f"{side}Promoted"],
                                        m[f"{other}Promoted"])
                self.y[row] = m[f"{side}Goals"]
        self.days = np.repeat([m["Date"].toordinal() for m in matches], 2)

    def _row(self, team, opponent, at_home, promoted, opponent_promoted):
        n_teams = (self.n_params - self.FIXED) // 2
        x = np.zeros(self.n_params)
        x[0] = 1
        x[1] = at_home
        x[2] = promoted
        x[3] = opponent_promoted
        x[self.FIXED + self.index[team]] = 1
        x[self.FIXED + n_teams + self.index[opponent]] = 1
        return x


def fit(X, y, weights, penalty, n_fixed):
    """Weighted, ridge-penalised Poisson regression by Newton's method."""
    beta = np.zeros(X.shape[1])
    beta[0] = math.log(np.average(y, weights=weights))
    ridge = np.full(X.shape[1], float(penalty))
    # A negligible penalty keeps the fit solvable when a fixed term has no
    # data yet, e.g. promoted teams before the second season.
    ridge[:n_fixed] = 1e-6
    for _ in range(50):
        mu = np.exp(X @ beta)
        gradient = X.T @ (weights * (y - mu)) - ridge * beta
        hessian = (X.T * (weights * mu)) @ X + np.diag(ridge)
        step = np.linalg.solve(hessian, gradient)
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def walk_forward(design, eval_rows, half_life, penalty):
    """Expected (home, away) goals for each match index in eval_rows.

    eval_rows are match indices into the design; they are refitted on every
    earlier match once per calendar week.
    """
    weeks = {}
    for k in eval_rows:
        day = int(design.days[2 * k])
        weeks.setdefault(date.fromordinal(day).isocalendar()[:2], []).append(k)

    expected = {}
    for week_matches in weeks.values():
        start = min(int(design.days[2 * k]) for k in week_matches)
        known = design.days < start
        age = start - design.days[known]
        weights = (np.ones_like(age, dtype=float) if half_life is None
                   else 0.5 ** (age / half_life))
        beta = fit(design.X[known], design.y[known], weights, penalty,
                   Design.FIXED)
        for k in week_matches:
            home, away = np.exp(design.X[[2 * k, 2 * k + 1]] @ beta)
            expected[k] = (home, away)
    return [expected[k] for k in eval_rows]


def score_grid(home_mean, away_mean):
    goals = np.arange(MAX_GOALS + 1)
    log_fact = np.array([math.lgamma(g + 1) for g in goals])
    home = np.exp(goals * math.log(home_mean) - home_mean - log_fact)
    away = np.exp(goals * math.log(away_mean) - away_mean - log_fact)
    grid = np.outer(home, away)
    return grid / grid.sum()


def predict(home_mean, away_mean):
    """Result probabilities plus the predicted result and scoreline.

    The predicted scoreline is the most likely score that agrees with the
    predicted result, so the two predictions never contradict each other.
    """
    grid = score_grid(home_mean, away_mean)
    h, a = np.indices(grid.shape)
    masks = {"H": h > a, "D": h == a, "A": h < a}
    probs = {r: grid[masks[r]].sum() for r in RESULTS}
    result = max(RESULTS, key=probs.get)
    best = np.argmax(np.where(masks[result], grid, -1))
    home_goals, away_goals = np.unravel_index(best, grid.shape)
    return probs, result, int(home_goals), int(away_goals), grid


def mean_log_likelihood(matches, expected):
    total = 0.0
    for m, (home_mean, away_mean) in zip(matches, expected):
        for goals, mean in ((m["HomeGoals"], home_mean),
                            (m["AwayGoals"], away_mean)):
            total += goals * math.log(mean) - mean - math.lgamma(goals + 1)
    return total / len(matches)


def metrics(matches, expected):
    n = len(matches)
    abs_err = sq_err = score_abs_err = total_err = diff_err = 0.0
    exact = correct = log_loss = brier = rps = score_log_loss = 0.0
    for m, (home_mean, away_mean) in zip(matches, expected):
        probs, result, ph, pa, grid = predict(home_mean, away_mean)
        hg, ag = m["HomeGoals"], m["AwayGoals"]
        abs_err += abs(home_mean - hg) + abs(away_mean - ag)
        sq_err += (home_mean - hg) ** 2 + (away_mean - ag) ** 2
        score_abs_err += abs(ph - hg) + abs(pa - ag)
        total_err += abs(home_mean + away_mean - hg - ag)
        diff_err += abs(home_mean - away_mean - (hg - ag))
        exact += (ph, pa) == (hg, ag)
        correct += result == m["Result"]
        log_loss -= math.log(probs[m["Result"]])
        brier += sum((probs[r] - (r == m["Result"])) ** 2 for r in RESULTS)
        # Ranked probability score: ordered H < D < A, so predicting a draw
        # for a home win is less wrong than predicting an away win.
        cum_p = np.cumsum([probs[r] for r in RESULTS])[:2]
        cum_o = np.cumsum([r == m["Result"] for r in RESULTS])[:2]
        rps += np.sum((cum_p - cum_o) ** 2) / 2
        score_log_loss -= math.log(grid[min(hg, MAX_GOALS), min(ag, MAX_GOALS)])
    return {
        "goals_mae": abs_err / (2 * n),
        "goals_rmse": math.sqrt(sq_err / (2 * n)),
        "score_mae": score_abs_err / (2 * n),
        "total_goals_mae": total_err / n,
        "goal_diff_mae": diff_err / n,
        "exact_score": exact / n,
        "score_log_loss": score_log_loss / n,
        "result_accuracy": correct / n,
        "result_log_loss": log_loss / n,
        "result_brier": brier / n,
        "result_rps": rps / n,
    }


def baseline(history, matches):
    """No-skill model: every match gets the historical average goals."""
    home = sum(m["HomeGoals"] for m in history) / len(history)
    away = sum(m["AwayGoals"] for m in history) / len(history)
    return [(home, away)] * len(matches)


def print_table(rows):
    labels = [
        ("goals_mae", "Goals MAE per team (expected goals)", "{:.3f}"),
        ("goals_rmse", "Goals RMSE per team (expected goals)", "{:.3f}"),
        ("score_mae", "Goals MAE per team (predicted score)", "{:.3f}"),
        ("total_goals_mae", "Total goals MAE per match", "{:.3f}"),
        ("goal_diff_mae", "Goal difference MAE per match", "{:.3f}"),
        ("exact_score", "Exact score correct", "{:.1%}"),
        ("score_log_loss", "Exact score log loss", "{:.3f}"),
        ("result_accuracy", "Result (H/D/A) correct", "{:.1%}"),
        ("result_log_loss", "Result log loss", "{:.3f}"),
        ("result_brier", "Result Brier score", "{:.3f}"),
        ("result_rps", "Result RPS", "{:.4f}"),
    ]
    names = list(rows)
    print(f"{'':40}" + "".join(f"{n:>12}" for n in names))
    for key, label, fmt in labels:
        print(f"{label:40}" + "".join(f"{fmt.format(rows[n][key]):>12}"
                                      for n in names))


def main():
    train, validation, test = (load(n) for n in
                               ("train", "validation", "test"))
    matches = train + validation + test
    mark_promoted(matches)
    design = Design(matches)
    val_rows = range(len(train), len(train) + len(validation))
    test_rows = range(len(train) + len(validation), len(matches))

    print("Tuning on validation (mean log-likelihood of goals, higher is "
          "better):")
    best = None
    for half_life in HALF_LIVES:
        for penalty in PENALTIES:
            expected = walk_forward(design, val_rows, half_life, penalty)
            score = mean_log_likelihood(validation, expected)
            print(f"  half-life {str(half_life):>4} days, "
                  f"penalty {penalty:>4}: {score:.4f}")
            if best is None or score > best[0]:
                best = (score, half_life, penalty)
    _, half_life, penalty = best
    print(f"Chosen: half-life {half_life} days, penalty {penalty}\n")

    expected = walk_forward(design, test_rows, half_life, penalty)
    history = train + validation
    home_rate = sum(m["Result"] == "H" for m in history) / len(history)
    always_home = metrics(test, baseline(history, test))
    print(f"Test season {test[0]['Season']} ({len(test)} matches):")
    print_table({"Model": metrics(test, expected), "Baseline": always_home})
    print("Baseline: every match predicted with the average home and away "
          f"goals of earlier seasons\n(so it always picks a home win; home "
          f"teams won {home_rate:.1%} of earlier matches).\n")

    out = os.path.join(HERE, "test_predictions.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Season", "Date", "HomeTeam", "AwayTeam",
                         "HomeGoals", "AwayGoals", "Result",
                         "ExpHomeGoals", "ExpAwayGoals",
                         "PredHomeGoals", "PredAwayGoals", "PredResult",
                         "ProbH", "ProbD", "ProbA"])
        for m, (home_mean, away_mean) in zip(test, expected):
            probs, result, ph, pa, _ = predict(home_mean, away_mean)
            writer.writerow([m["Season"], m["Date"].isoformat(),
                             m["HomeTeam"], m["AwayTeam"], m["HomeGoals"],
                             m["AwayGoals"], m["Result"],
                             f"{home_mean:.3f}", f"{away_mean:.3f}", ph, pa,
                             result, f"{probs['H']:.3f}",
                             f"{probs['D']:.3f}", f"{probs['A']:.3f}"])
    print(f"Wrote predictions to {out}")


if __name__ == "__main__":
    main()
