# Premier League Match Prediction

A project for predicting the results of Premier League matches.

## Usage

```
pip install -r requirements.txt
python download_results.py   # 10 seasons -> premier_league_results.csv
python split_data.py         # -> train.csv (7), validation.csv (2), test.csv (1)
python predict_scores.py     # Poisson model; prints test metrics, writes test_predictions.csv
```
