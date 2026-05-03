# Parkinson Voice Analysis

This workspace contains a reproducible pipeline for the Parkinson voice-classification practice.

## Files

- `analysis_parkinson.py`: full experiment pipeline.
- `pd_speech_features.csv`: dataset used in the practice.
- `results/`: generated metrics, plots, synthetic data and Markdown report.

## How to run

```bash
python analysis_parkinson.py
```

You can also change the number of CTGAN epochs if you want a faster or slower synthetic-data run:

```bash
python analysis_parkinson.py --ctgan-epochs 100
```

## What the script does

1. Loads the Parkinson voice dataset and removes the `id` column.
2. Creates a stratified train/test split.
3. Selects the most relevant features with a combined mutual-information and RandomForest ranking.
4. Tunes several classical models on real data.
5. Fits CTGAN on the selected feature space, generates a purely synthetic training set and repeats the tuning process.
6. Exports plots, tables and a final report to `results/analysis_report.md`.