# Market Stress Regime Detection

## Overview

Financial markets experience periods of instability characterized by increased volatility, rapid drawdowns, and structural regime changes. Detecting these periods early is an important problem in quantitative finance, risk management, and portfolio allocation.

This project investigates whether machine learning models can identify and anticipate periods of **market stress** using historical financial time-series data.

Using more than a decade of SPY data, we construct statistical features describing market dynamics and train classification models to estimate the probability that the market is entering a stress regime.

The goal is not necessarily to predict exact crashes, but to **detect transitions toward high-risk market regimes**.

---

## Dataset

The dataset is constructed using historical daily data for the **SPY ETF**, downloaded via the `yfinance` API.

SPY is used because it tracks the **S&P 500**, one of the most liquid and widely traded equity indices in the world, making it a relevant benchmark for global market conditions.

The dataset includes approximately **13 years of daily observations**.

---

## Feature Engineering

Several statistical and technical indicators are constructed from historical price data in order to capture the dynamics of market behavior.

Examples include:

- Daily returns
- Rolling volatility
- Momentum indicators
- Short-term and long-term return statistics
- Volatility clustering indicators

These features aim to capture patterns associated with:

- market instability
- volatility clustering
- regime transitions

---

## Stress Event Definition

Market stress events are defined using a rule-based approach based on **large negative returns over short horizons**.

This approach allows the creation of a binary classification target:


target = 1 → stress regime
target = 0 → normal market conditions


Although simplified, this definition captures periods of significant market turbulence such as sharp corrections and volatility spikes.

---

## Baseline Model

Before training machine learning models, a simple **volatility-based baseline** is implemented.

The baseline assumes that high historical volatility corresponds to increased stress probability.

This provides a reference point to evaluate whether machine learning models provide additional predictive power.

---

## Machine Learning Models

Two classification models are trained:

### Logistic Regression
A simple linear probabilistic classifier that provides interpretable probability estimates.

### Random Forest
A tree-based ensemble model capable of capturing nonlinear relationships between features.

Models are evaluated using standard classification metrics:

- Accuracy
- Precision
- Recall
- ROC-AUC
- Confusion Matrix

---

## Walk-Forward Validation

Because financial time-series are inherently temporal, standard random cross-validation would introduce **look-ahead bias**.

To address this, the project implements **walk-forward validation**, where models are repeatedly trained on past data and evaluated on future periods.

This methodology more closely resembles real-world trading conditions.

---

## Model Calibration

Predicted probabilities are calibrated to ensure that model outputs correspond to meaningful probabilities.

Calibration curves are used to compare predicted probabilities with observed frequencies of stress events.

---

## Results

The results show that machine learning models are able to identify periods of elevated market stress.

However, the model primarily acts as a **stress regime detector** rather than a precise predictor of future crashes.

Predicted probabilities tend to increase during:

- market drawdowns
- periods of high volatility
- turbulent market regimes

This behavior reflects the well-known phenomenon of **volatility clustering** in financial markets.

---

## Visualization

Several visualizations are produced to analyze the model’s behavior:

- Market stress probability over time
- Comparison between predicted stress probability and real stress events
- Detected stress periods on the SPY price series
- Feature importance analysis

These plots help interpret how the model reacts to changing market conditions.

---

## Key Insights

The model captures important characteristics of financial markets:

- Volatility clustering
- Market regime shifts
- Stress periods associated with large drawdowns

While predicting crashes remains extremely difficult, detecting transitions into risky regimes can still be valuable for:

- portfolio risk management
- exposure reduction
- regime-based investment strategies

---

## Project Structure


market-stress-regime-detection
│
├── data
│ ├── raw
│ └── processed
│
├── src
│ ├── data_loader.py
│ ├── feature_engineering.py
│ ├── models.py
│ ├── walk_forward.py
│ ├── calibration.py
│ ├── baseline.py
│ ├── visualization.py
│ └── feature_importance.py
│
├── results
│ └── plots
│
└── main.py


---

## Limitations

Financial markets are noisy and difficult to predict.

The current model mainly captures **volatility regimes** rather than predicting exact future crashes.

Possible improvements include incorporating additional sources of information and refining the definition of stress events.

---

## Future Improvements

Several directions could improve the predictive power of the model:

### Improve the Target Definition
Instead of detecting current stress events, future versions of the model could attempt to predict **future drawdowns** or **future volatility spikes**, such as:


stress_event = return_{t+5} < -3%


This would transform the model from a regime detector into a true **forward-looking predictor**.

---

### Incorporate Additional Market Indicators

Additional financial variables could provide stronger predictive signals, including:

- VIX (implied volatility index)
- Credit spreads
- Market breadth indicators
- Put-call ratios
- Volatility term structure

These variables often contain information about **market expectations and risk sentiment**.

---

### Explore More Advanced Models

Future work could test more advanced machine learning methods such as:

- Gradient Boosting (XGBoost / LightGBM)
- Neural networks
- Hidden Markov Models for regime detection

These models may capture more complex nonlinear relationships in financial data.

---

### Multi-Asset Analysis

Instead of focusing only on SPY, the model could be extended to include:

- multiple equity indices
- bond markets
- commodities
- volatility indices

This would allow the model to capture **cross-market stress signals**.

---

### Strategy Backtesting

The stress probability signal could be used to build a simple **risk-management strategy**, such as:

- reducing equity exposure when stress probability rises
- increasing hedging during high-risk regimes

Backtesting such strategies would provide a practical evaluation of the signal’s usefulness.

---

## Conclusion

This project demonstrates how machine learning can be applied to financial time-series in order to detect periods of elevated market stress.

Although precise prediction of crashes remains challenging, identifying high-risk regimes can provide valuable insights for quantitative risk management and trading strategies.