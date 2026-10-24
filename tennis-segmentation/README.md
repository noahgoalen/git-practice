# Tennis Segmentation

A project for segmentation using professional tennis match data.

## Data

The [Match Charting Project](https://github.com/JeffSackmann/tennis_MatchChartingProject)
(MCP): crowdsourced, shot-by-shot records of thousands of men's and women's
professional matches, including shot type, direction, return depth and error
type for every point, plus match metadata (players, tournament, date,
surface) and match-level aggregate stats.

The data is licensed under
[CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/):
attribution is required, non-commercial use only, and anything built on it
must be shared under the same licence.

> Crowdsourced shot-by-shot professional tennis data by
> [The Tennis Abstract Match Charting Project](http://www.tennisabstract.com/charting/meta.html)
> is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike
> 4.0 International License. Based on a work at
> https://github.com/JeffSackmann/tennis_MatchChartingProject

## Usage

```
pip install -r requirements.txt
jupyter notebook eda.ipynb             # exploratory data analysis
jupyter notebook elo_prediction.ipynb  # Elo model predicting ATP match winners
```

The notebook downloads the data files it needs (about 240 MB) into `data/`
on its first run. `data/` is not committed to git.

### ATP results for the Elo notebook

`elo_prediction.ipynb` uses every ATP tour-level match from the yearly files
at [tennis-data.co.uk](http://www.tennis-data.co.uk/alldata.php), stored in
`atp_data/` (one `.xls`/`.xlsx` per year). The site blocks automated
downloads, so new years have to be downloaded in a browser and added to that
folder; the notebook picks up every file there.
