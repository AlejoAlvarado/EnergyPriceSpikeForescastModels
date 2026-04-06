# Predicting Alberta Electricity Price Spikes: Evidence From Supply, Demand, and Renewable Conditions

Aravindh Palaniguru, Alejandro Jose Alvarado Barrera, and Jorge Gutierrez Barajas

DATA 607

April 5, 2026

## Abstract

Extreme hourly price spikes in Alberta's energy-only electricity market create operational and financial risk for market participants, which makes short-horizon spike prediction a relevant applied machine learning problem. This report evaluates whether publicly observable AESO market and generation data can classify whether the Alberta pool price will exceed CAD 200/MWh at t+2. Two public AESO datasets were merged at the hourly level into a file with 48,887 hourly observations covering January 2020 to July 2025, while the LSTM and CNN notebook variants operated on 48,839 rows after their preprocessing steps. Exploratory analysis shows a heavy right tail in pool prices and a consistent scarcity signature: future spike hours are associated with higher current prices, higher net load, lower wind output, lower renewable share, and thinner reserve margins. Three neural-network classifiers were compared: a multilayer perceptron (MLP), a long short-term memory network (LSTM), and a one-dimensional convolutional neural network (CNN). Using the rounded metrics presented in Data607_EnergySpike_Presentation_v2.pptx, the CNN achieved the strongest thresholded test performance (F1 = 0.416, precision = 0.340, recall = 0.535, ROC-AUC = 0.941), followed by the LSTM (F1 = 0.394, precision = 0.311, recall = 0.535, ROC-AUC = 0.945) and the MLP (F1 = 0.363, precision = 0.275, recall = 0.535, ROC-AUC = 0.934). The results indicate that sequence-aware neural architectures can improve short-horizon spike classification, although the gains remain modest because the task is rare-event prediction in a market whose structure changes over time.

## Background and Introduction

Alberta's wholesale electricity market is an energy-only market in which the hourly pool price is formed by the interaction of supply and demand. Because electricity cannot be stored economically at scale and system balance must be maintained continuously, periods of tight capacity, high demand, weak renewable output, or intertie stress can produce abrupt price spikes. These episodes matter for generators, large industrial consumers, retailers, and system operators because they alter dispatch incentives, hedging costs, and operating risk.

The present project asks whether public system-level information is sufficient to anticipate short-run spike risk. That question is practical as well as methodological: if a model can identify high-risk hours before scarcity pricing materializes, market participants can adapt bidding, dispatch, or load scheduling decisions accordingly. Alberta is also an attractive applied setting because its market is province-specific, publicly documented, and not as overused in classroom projects as more generic benchmark datasets.

The modeling choice is motivated by prior work showing that electricity prices are nonlinear, seasonal, and spike-prone. Lago et al. (2018) show that deep-learning methods often outperform traditional price forecasting approaches. Alberta-specific studies likewise motivate this comparison: Manfre Jaimes et al. (2023) use neural methods for multi-day price forecasting, while Zamudio Lopez et al. (2024) examine spike occurrence directly. In that context, the present report emphasizes transparent comparison across three neural architectures rather than claiming a production-ready forecasting system.

## Data Source and Preparation

The report uses two publicly available AESO datasets. The first is the Hourly Metered Volumes and Pool Price and AIL file, which provides pool price, Alberta Internal Load, and intertie flow information (Alberta Electric System Operator, 2025a). The second is the Historical Generation Data (CSD), which records hourly generation and system capability by fuel type (Alberta Electric System Operator, 2025b). Because both sources are distributed publicly by AESO, the group had permission to use them for academic analysis under the operator's public reporting framework.

The two datasets were aligned on a common hourly Mountain Prevailing Time axis, aggregated to the province level where necessary, and then merged into a unified modeling table. The merged AESO file used for exploratory analysis contains 48,887 rows and 108 columns. The model notebooks then apply slightly different preprocessing choices: the MLP notebook reports 59 features and retains 48,887 rows, whereas the LSTM and CNN notebooks report 32 and 92 features respectively and operate on 48,839 rows.

To avoid another mismatch between exploratory material and the model write-up, this report uses the presentation deck for the final rounded performance metrics and the notebooks for model-specific details such as feature counts, selected thresholds, and pipeline row counts.

| Model | Features | Rows | Train | Validation | Test | Threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MLP | 59 | 48,887 | 33,696 | 9,648 | 5,543 | 0.91 |
| LSTM | 32 | 48,839 | 33,672 | 9,624 | 5,543 | 0.94 |
| CNN | 92 | 48,839 | 33,672 | 9,624 | 5,543 | 0.77 |

## Preliminary Analyses

Exploratory analysis shows that Alberta pool prices are strongly right-skewed rather than approximately Gaussian. In the source data, the median hourly price is CAD 45.34/MWh, the 95th percentile is CAD 431.00/MWh, and the observed maximum is CAD 999.99/MWh. This heavy upper tail justifies a spike classification framing rather than a narrow focus on average price behavior.

Temporal structure is also visible. Mean prices rise most sharply in late afternoon and early evening, and higher-price months cluster in the summer period. Year-level spike prevalence is similarly uneven: 2020 = 2.16%, 2021 = 9.93%, 2022 = 18.94%, 2023 = 17.59%, 2024 = 5.73%, 2025 = 1.94%. This variability indicates that the market regime is not stationary across the full sample.

The descriptive comparisons in the final deck also support a scarcity narrative. Relative to hours that are not followed by a spike, hours followed by a spike at t+2 have substantially higher current prices (CAD 366.57 versus CAD 65.02), higher net load (9,599 MW versus 8,518 MW), lower wind output (407 MW versus 1,103 MW), lower renewables share (6.7% versus 13.0%), and lower reserve margin (0.722 versus 0.851). These patterns are economically coherent because price spikes tend to emerge when dispatchable supply must absorb more load while renewable support weakens.

**Figure 1.** *Distribution and temporal structure of Alberta pool prices.*

![Figure 1](../figures/figure_1_price_dynamics.png)

*Note.* Left panel shows the heavy right tail in hourly pool prices. Right panel shows mean hourly prices by month and hour of day, highlighting seasonal and intraday structure.

**Figure 2.** *Mean system conditions for hours with and without a future spike at t+2.*

![Figure 2](../figures/figure_2_spike_conditions.png)

*Note.* Future spike hours exhibit higher current prices and net load, but lower wind output, lower renewables share, and lower reserve margin.

**Figure 3.** *Correlation matrix for selected market and system-stress variables.*

![Figure 3](../figures/figure_3_correlation_matrix.png)

*Note.* The exploratory matrix summarizes pairwise linear association and is used only to describe structure, not to make causal claims.

## Problem Statement and Working Hypotheses

The central problem statement is whether publicly observable market and generation conditions contain enough information to identify a future price spike before it occurs. In operational terms, the presentation frames the task as a short-horizon warning problem; in the modeling pipeline itself, the binary target is whether the pool price exceeds CAD 200/MWh at t+2.

Two working hypotheses guided the analysis. First, sequence-aware models should outperform a flat multilayer perceptron because short-run ramp events and evolving system tightness are inherently temporal. Second, future spike hours should be associated with higher load pressure and lower renewable availability, which should make variables such as current price, net load, reserve margin, and wind output informative predictors of the target.

## Formal Analyses

All models were trained and evaluated with time-ordered data splits in order to avoid look-ahead bias. The final workflow used a chronological train-validation-test partition, coupled with TimeSeriesSplit cross-validation inside the pre-test horizon. Because spike hours are relatively rare, F1-score was used as the primary metric, with precision, recall, PR-AUC, and ROC-AUC used to qualify model behaviour.

The analytical storyline began with a simple MLP baseline. The MLP notebook used 59 tabular features, selected a validation threshold of 0.91, and produced the weakest final F1 among the three models. This baseline was important pedagogically because it established how much could be learned from a flat feature vector before adding explicit temporal structure.

The LSTM and CNN then introduced sequence modelling. The LSTM notebook used 32 features, removed manual lag variables that were redundant in a recurrent architecture, and selected a validation threshold of 0.94. The CNN notebook used 92 features with a 24-hour lookback window, selected a validation threshold of 0.77, and treated convolutional filters as detectors of local pre-spike patterns.

In practical terms, the main comparison in the project is therefore not between a naive heuristic and the final CNN, but between a simple MLP baseline and temporally structured neural models, especially the CNN.

## Results and Interpretation

Using the rounded values shown in the presentation deck, test F1 rises monotonically from the simple MLP (0.363) to the LSTM (0.394) and then to the CNN (0.416). That progression supports the core modelling claim of the project: adding temporal structure helps, and the CNN is the strongest of the three final models on the thresholded spike-classification task.

The improvement is driven more by precision than by recall. The deck reports essentially the same recall for all three neural models, approximately 0.535, while precision improves from 0.275 for the MLP to 0.311 for the LSTM and 0.340 for the CNN. In operational terms, the sequence-aware models do not capture more spikes than the baseline; rather, they issue fewer false alarms while preserving the same hit rate.

ROC-AUC values remain high across the three models, with 0.934 for the MLP, 0.945 for the LSTM, and 0.941 for the CNN. This pattern is instructive. The LSTM slightly outperforms the CNN as a ranker of spike risk, but the CNN achieves the best thresholded F1 once the decision rule is fixed. That distinction is important because the project's applied objective is not just ranking hours by risk, but making a usable classification decision under class imbalance.

| Model | Test F1 | Precision | Recall | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| Naive | 0.384 | 0.384 | 0.384 | n/a |
| MLP | 0.363 | 0.275 | 0.535 | 0.934 |
| LSTM | 0.394 | 0.311 | 0.535 | 0.945 |
| CNN | 0.416 | 0.340 | 0.535 | 0.941 |

**Figure 4.** *Test performance comparison reported in Data607_EnergySpike_Presentation_v2.pptx.*

![Figure 4](../figures/figure_4_model_results.png)

*Note.* Panel A compares the rounded test F1 values reported in the presentation deck. Panel B shows that precision improves from MLP to LSTM to CNN while recall remains at approximately 0.535, so most of the F1 gain comes from reducing false positives rather than capturing additional spike hours.

## Conclusions

This project shows that publicly available AESO data contain meaningful information about short-horizon electricity price spike risk in Alberta. The formal comparison across an MLP, an LSTM, and a CNN shows that temporally structured neural models are preferable to a simple tabular baseline, with the CNN emerging as the best overall model on the final thresholded F1 metric reported in the presentation.

At the same time, the results should be interpreted carefully. The sample exhibits pronounced class imbalance, the market changed materially over the 2020-2025 period, and the final gains over the MLP baseline remain limited in absolute terms. Future work should therefore extend the predictor set with weather forecasts, outage information, and offer-stack or merit-order variables, while also evaluating longer forecast horizons and cost-sensitive thresholds that better reflect operational priorities.

## Task Division

Jorge Gutierrez Barajas led the LSTM workflow, including recurrent-model configuration, threshold-tuning experiments, and interpretation of the sequence-model results. Alejandro Jose Alvarado Barrera led the CNN workflow, including convolutional architecture design, class-weight calibration, and interpretation of the final best-performing model. Aravindh Palaniguru led the MLP baseline, the initial feature-engineering workflow, and the baseline comparison logic. All three members contributed jointly to data acquisition, data cleaning, exploratory analysis, interpretation of findings, and preparation of the final presentation and report.

## References

- Alberta Electric System Operator. (2025a). Hourly metered volumes and pool price and AIL data 2001 to July 2025 [Data set]. https://www.aeso.ca/market/market-and-system-reporting/data-requests/hourly-generation-metered-volumes-and-pool-price-and-ail-data-2001-to-july-2025/

- Alberta Electric System Operator. (2025b). Historical generation data (CSD) [Data set]. https://www.aeso.ca/market/market-and-system-reporting/data-requests/historical-generation-data/

- Lago, J., De Ridder, F., & De Schutter, B. (2018). Forecasting spot electricity prices: Deep learning approaches and empirical comparison of traditional algorithms. Applied Energy, 221, 386-405.

- Manfre Jaimes, D., Zamudio Lopez, M., Zareipour, H., & Quashie, M. (2023). A hybrid model for multi-day-ahead electricity price forecasting considering price spikes. Forecasting, 5(3), 499-521.

- Saha, C. (2025). Developing a statistical risk assessment and grid prediction tool for power system reliability (Master's capstone project). University of Calgary. https://ucalgary.scholaris.ca/server/api/core/bitstreams/2660b5fd-7457-4720-b2c1-f4127a579d62/content

- Zamudio Lopez, M., et al. (2024). Forecasting the occurrence of electricity price spikes: A statistical-economic investigation study. Forecasting, 6(1), 7.
