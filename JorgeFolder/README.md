# JorgeFolder

End-to-end machine learning project for one-hour-ahead Alberta electricity price spike prediction using AESO hourly data from 2020 to 2025.

## Structure

- `src/`: reproducible preprocessing, EDA, model training, evaluation, and report generation
- `models/mlp`, `models/cnn`, `models/lstm`: separate model folders with checkpoints and predictions
- `outputs/data`: processed data, split summary, and data dictionary
- `outputs/figures`: EDA and evaluation figures
- `outputs/metrics`: model comparison and error analysis
- `outputs/report`: APA-style report in Markdown and DOCX

## Run

```powershell
python -m JorgeFolder.src.run_all
```

## Modeling Design

- Target: `spike_t_plus_1 = 1` if next-hour pool price exceeds `$200/MWh`
- Fixed time-based splits:
  - Train: `2020-01-02 00:00:00` to `2023-11-05 23:00:00`
  - Validation: `2023-11-06 00:00:00` to `2024-12-11 23:00:00`
  - Test: `2024-12-12 00:00:00` to `2025-07-01 23:00:00`
- Hyperparameter tuning: `TimeSeriesSplit(n_splits=5)` on the pre-test horizon only
- Primary metric: `F1-score`
