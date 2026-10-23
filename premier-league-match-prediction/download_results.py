"""Download Premier League results for the last 10 seasons into one CSV.

Data comes from football-data.co.uk, which publishes one CSV per season.
Run with:  python download_results.py
Output:    premier_league_results.csv (next to this script)
"""

import csv
import io
import os
import urllib.request
from datetime import datetime

# Season codes as football-data.co.uk uses them: "1617" is 2016-17.
SEASONS = ["1617", "1718", "1819", "1920", "2021",
           "2122", "2223", "2324", "2425", "2526"]
URL = "https://www.football-data.co.uk/mmz4281/{}/E0.csv"
OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "premier_league_results.csv")
COLUMNS = ["Season", "Date", "HomeTeam", "AwayTeam",
           "HomeGoals", "AwayGoals", "Result"]


def parse_date(text):
    # Older seasons use dd/mm/yy, newer ones dd/mm/yyyy.
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            pass
    raise ValueError(f"Unrecognised date: {text!r}")


def fetch_season(code):
    with urllib.request.urlopen(URL.format(code)) as response:
        raw = response.read()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")

    season = f"20{code[:2]}-{code[2:]}"
    rows = []
    for row in csv.DictReader(io.StringIO(text)):
        if not row.get("HomeTeam"):
            continue  # skip blank trailing lines
        rows.append({
            "Season": season,
            "Date": parse_date(row["Date"]),
            "HomeTeam": row["HomeTeam"],
            "AwayTeam": row["AwayTeam"],
            "HomeGoals": int(row["FTHG"]),
            "AwayGoals": int(row["FTAG"]),
            "Result": row["FTR"],  # H = home win, D = draw, A = away win
        })
    return rows


def main():
    all_rows = []
    for code in SEASONS:
        rows = fetch_season(code)
        print(f"20{code[:2]}-{code[2:]}: {len(rows)} matches")
        all_rows.extend(rows)

    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Wrote {len(all_rows)} matches to {OUTPUT}")


if __name__ == "__main__":
    main()
