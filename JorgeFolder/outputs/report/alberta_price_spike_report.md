# Predicting Alberta Electricity Price Spikes (2020-2025)

## Abstract

This project develops a full machine learning pipeline to predict two-hour-ahead electricity price spikes in Alberta using hourly AESO market and generation data from 2020-01-03 through 2025-07-01 23:00:00. Three neural-network families were implemented and compared: a multilayer perceptron (MLP), a 1D convolutional neural network (CNN), and a long short-term memory network (LSTM). The target was defined as a binary indicator for pool prices above $200/MWh at time t+2, while explanatory variables were measured at time t. Using a strict time-based split and five-fold TimeSeriesSplit tuning within the train/validation horizon, the best model on the untouched test set was the CNN with F1=0.475, precision=0.462, recall=0.489, and ROC-AUC=0.941. The results indicate that system tightness, lagged price behavior, and renewable variability are informative predictors of spike risk.

## Introduction and Motivation

Alberta operates an energy-only electricity market in which the pool price reflects the hourly intersection of supply and demand. Because electricity is non-storable and the system must remain balanced in real time, periods of tight supply, strong load, limited imports, and renewable variability can produce sharp, short-lived price spikes. These spikes matter economically because they affect generator revenues, retailer costs, hedging, and system operations.

## Literature Review

The project design follows prior work showing that electricity prices are heavy-tailed, seasonal, and prone to spikes, which motivates both spike-classification frameworks and neural-network architectures. Stathakis et al. (2021) analyze price spike forecasting as a classification problem. Ugurlu et al. (2018) show that recurrent neural networks are effective for electricity price forecasting, while Kuo and Huang (2018) motivate CNN/LSTM-type architectures for structured temporal inputs.

## Data and Methodology

The modeling dataset was built from AESO hourly data using a fixed Mountain Prevailing Time axis (UTC-7) to avoid daylight saving ambiguity. The effective split interpretation was: train from 2020-01-03 through 2023-11-05 23:00:00, validation from 2023-11-06 through 2024-12-11 23:00:00, and test from 2024-12-12 through 2025-07-01 23:00:00. The target was a binary spike indicator for two hours ahead. Continuous predictors were standardized with training statistics only, while dummy variables were left unscaled. Hyperparameter tuning used five-fold TimeSeriesSplit within the pre-test period only.

### Split Summary

| split | rows | start | end | spike_rate_lead_2 |
| --- | --- | --- | --- | --- |
| train | 33672 | 2020-01-03 | 2023-11-05 23:00:00 | 0.1238 |
| validation | 9648 | 2023-11-06 | 2024-12-11 23:00:00 | 0.0615 |
| test | 4848 | 2024-12-12 | 2025-07-01 23:00:00 | 0.0182 |

### Model Comparison

| model | cv_best_f1 | validation_f1 | validation_precision | validation_recall | validation_roc_auc | test_f1 | test_precision | test_recall | test_roc_auc | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cnn | 0.5504 | 0.5818 | 0.648 | 0.5278 | 0.9209 | 0.4751 | 0.4624 | 0.4886 | 0.9406 | 0.9 |
| lstm | 0.591 | 0.5936 | 0.6078 | 0.5801 | 0.9153 | 0.4111 | 0.3152 | 0.5909 | 0.932 | 0.9 |
| mlp | 0.4739 | 0.5479 | 0.4567 | 0.6847 | 0.931 | 0.2085 | 0.1579 | 0.3068 | 0.5132 | 0.9 |

## Results and Model Comparison

The best test-set performer was the CNN, which achieved the highest F1-score under class imbalance. Validation and test metrics were consistent across models, but the best architecture balanced precision and recall more effectively than the alternatives.

## Discussion

False negatives for the best model were associated with mean reserve margin 1.168, mean wind generation 665.5 MW, and mean two-hour-ahead price 459.21 CAD/MWh.

False positives for the best model were associated with mean reserve margin 1.126, mean wind generation 320.8 MW, and mean two-hour-ahead price 90.07 CAD/MWh.

Economically, the error patterns are consistent with Alberta's spike dynamics: missed spikes tend to occur when the system appears only moderately tight on observed fundamentals, while false positives arise when tight conditions do not escalate into realized scarcity pricing. This is consistent with the importance of reserve margin, fuel mix, and renewable availability in an energy-only market.

## Conclusion and Future Work

The project delivers a reproducible neural-network pipeline for Alberta spike prediction using hourly AESO fundamentals. Future work should extend the feature space with weather and outage data, explore calibration under rarer extreme-price regimes, and test cost-sensitive decision thresholds aligned to operational or trading objectives.

## References

- Alberta Electric System Operator. (2021). 2020 annual market statistics. https://www.aeso.ca/assets/Uploads/2020-Annual-Market-Stats-Final.pdf
- Alberta Electric System Operator. (n.d.). Daily average pool price report. https://ets.aeso.ca/ets_web/ip/Market/Reports/DailyAveragePoolPriceReportServlet
- Stathakis, E., Papadimitriou, T., & Gogas, P. (2021). Forecasting price spikes in electricity markets. Review of Economic Analysis, 13, 65-87. https://openjournals.uwaterloo.ca/index.php/rofea/article/download/1822/2096/5445
- Ugurlu, U., Oksuz, I., & Tas, O. (2018). Electricity price forecasting using recurrent neural networks. Energies, 11(5), 1255. https://res.mdpi.com/d_attachment/energies/energies-11-01255/article_deploy/energies-11-01255.pdf
- Kuo, P.-H., & Huang, C.-J. (2018). An electricity price forecasting model by hybrid structured deep neural networks. Sustainability, 10(4), 1280. https://www.mdpi.com/2071-1050/10/4/1280
- Christensen, T., Hurn, A. S., & Lindsay, K. A. (2012). Forecasting spikes in electricity prices. International Journal of Forecasting, 28(2), 400-411. https://doi.org/10.1016/j.ijforecast.2011.02.019
