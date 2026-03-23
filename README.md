# dengue-forecasting-xgboost
Dengue fever case prediction for San Juan and Iquitos using XGBoost with time-series feature engineering, log transform, and Optuna hyperparameter tuning. Built as part of the DrivenData DengAI competition.
# dengue-forecasting-xgboost

Dengue fever case prediction for **San Juan, Puerto Rico** and **Iquitos, Peru** using XGBoost with time-series feature engineering, log transformation, and Optuna hyperparameter tuning.

Built as part of the [DrivenData DengAI competition](https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/).

---

## Results

| City | Model | Validation MAE | Notes |
|------|-------|---------------|-------|
| Iquitos (IQ) | XGBoost + Optuna | **3.26** | Cases range 0–20/week |
| San Juan (SJ) | XGBoost + log transform + Optuna | **8.62** | Cases range 0–461/week |
| **Combined** | **DrivenData leaderboard** | **32.15** | Tested on 5 yrs SJ + 3 yrs IQ |

---

## Problem Statement

Dengue fever is a mosquito-borne disease affecting millions across tropical regions. Early prediction of outbreak size helps public health officials allocate resources before cases spike.

The goal is to predict `total_cases` for each `(city, year, weekofyear)` using environmental and climate features including temperature, humidity, precipitation, and vegetation indices.

---

## Approach

### Why separate models per city?
San Juan and Iquitos have fundamentally different outbreak profiles. San Juan peaks at 461 cases/week while Iquitos peaks at ~20. Training one model on both cities would confuse the scale — separate models learn each city's unique patterns.

### Feature Engineering
Every feature was built from domain knowledge about the dengue transmission cycle:

| Feature Group | Columns Created | Why It Helps |
|--------------|----------------|--------------|
| Lag features | lag1 → lag16 | Captures 4 months of outbreak persistence |
| Rolling statistics | rolling_mean_4/8, rolling_max_4, rolling_std_4 | Captures trend, peak, and volatility |
| Seasonal encoding | week_sin, week_cos | Circular encoding — week 52 and week 1 are neighbours |
| Trend counter | trend | Captures long-term drift over 18 years |
| Climate lags | humidity_lag4/8, precip_lag4/8 | Rain today → mosquito breeding → cases in 4-8 weeks |
| Outbreak momentum | outbreak_momentum, outbreak_flag | Explicit signal when cases are accelerating |

### Why log transform for San Juan?
San Juan has extreme spikes (461 cases). XGBoost predicts group averages — a spike of 461 in 764 training rows is too rare to average well. Applying `log1p(cases)` compresses 461 → 6.13, making all values comparable in scale. Predictions are reversed with `expm1()` after inference.

### Hyperparameter Tuning
[Optuna](https://optuna.org/) was used for both cities with 100 trials for Iquitos and 200 trials for San Juan. Key parameters tuned: `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `min_child_weight`, `reg_alpha`, `reg_lambda`.

---

## Model Performance

### Iquitos — MAE journey
```
18.74  →  data leakage bug present (wrong validation set)
 3.78  →  after fixing data leakage
 3.68  →  after fixing week_cos encoding bug  
 3.26  →  after Optuna tuning (100 trials)
```

### San Juan — MAE journey
```
17.80  →  baseline XGBoost + log transform
 9.23  →  after Optuna tuning (100 trials)
 8.62  →  after adding outbreak momentum feature + 200 Optuna trials
```

---

## Project Structure

```
dengue-forecasting-xgboost/
│
├── dengai_XGboost_model.ipynb   # Main notebook — full pipeline
├── submission/
│   └── dengai_submission_final.csv  # DrivenData submission file
├── requirements.txt             # Python dependencies
└── README.md
```

> **Note:** Raw data files (`dengue_features_train.csv`, `dengue_labels_train.csv`, `dengue_features_test.csv`) are not included. Download them from the [DrivenData competition page](https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/data/).

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/dengue-forecasting-xgboost.git
cd dengue-forecasting-xgboost
pip install -r requirements.txt
```

Then download the data from DrivenData, place the CSV files in the project root, and run the notebook top to bottom.

---

## Requirements

```
xgboost==3.1.2
optuna==4.8.0
scikit-learn==1.7.1
pandas==3.0.0
numpy==2.4.2
matplotlib==3.10.8
```

---

## Key Learnings

- **Data leakage is silent and dangerous** — the original notebook had the IQ validation set accidentally reading from SJ data, producing a fake MAE of 18.74. After fixing it, MAE dropped to 3.78 instantly.
- **Feature engineering matters more than model choice** — going from raw climate features to engineered lag/rolling/seasonal features improved results significantly before any tuning.
- **Log transform is essential for skewed targets** — without it, XGBoost on San Juan undershoots every spike.
- **Separate models per city outperform one combined model** — the cities have different scales, seasonality, and outbreak patterns.

---

## What's Next

- [ ] LightGBM comparison on both cities
- [ ] LSTM model to better capture long-term outbreak memory
- [ ] Fix recursive forecasting dampening for better leaderboard score
- [ ] Ensemble XGBoost + LSTM predictions

---

## Competition

**Platform:** [DrivenData](https://www.drivendata.org/)  
**Competition:** [DengAI: Predicting Disease Spread](https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/)  
**Metric:** Mean Absolute Error (MAE)  
**Leaderboard Score:** 32.15
