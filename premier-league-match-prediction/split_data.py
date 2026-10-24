"""Split the downloaded results into train, validation and test sets by season.

The split is chronological so the model is never trained on matches that
happen after the ones it is evaluated on:
    train:      the first 7 seasons
    validation: the next 2 seasons
    test:       the last season

Run download_results.py first, then:  python split_data.py
Output:  train.csv, validation.csv, test.csv (next to this script)
"""

import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT = os.path.join(HERE, "premier_league_results.csv")
SPLITS = [("train", 7), ("validation", 2), ("test", 1)]


def main():
    with open(INPUT, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        rows = list(reader)

    # Season labels like "2016-17" sort in chronological order.
    seasons = sorted({row["Season"] for row in rows})
    needed = sum(count for _, count in SPLITS)
    if len(seasons) != needed:
        raise ValueError(f"Expected {needed} seasons, found {len(seasons)}")

    start = 0
    for name, count in SPLITS:
        chosen = seasons[start:start + count]
        start += count
        split_rows = [row for row in rows if row["Season"] in chosen]

        path = os.path.join(HERE, f"{name}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(split_rows)
        print(f"{name}: {chosen[0]} to {chosen[-1]}, "
              f"{len(split_rows)} matches -> {path}")


if __name__ == "__main__":
    main()
