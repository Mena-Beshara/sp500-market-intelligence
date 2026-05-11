# S&P 500 Market Intelligence System

**A comprehensive financial data science project** combining **price forecasting** with **market anomaly & regime detection** — built for fintech, banking, and risk management roles.

![Python](https://img.shields.io/badge/Python-3.11-blue) 
![Pandas](https://img.shields.io/badge/Pandas-2.x-orange)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-4CAF50)
![yfinance](https://img.shields.io/badge/Data-yfinance-yellow)

---

## 🎯 Project Overview

This project analyzes the **S&P 500 Index (^GSPC)** from 2000 to present with two complementary angles:
- **Forecasting** (classical + deep learning)
- **Risk & Anomaly Detection** (volatility modeling + outlier/regime detection)

The goal is to demonstrate **production-grade financial data science skills**: data diligence, statistical rigor, domain knowledge, and reproducible workflows.

---

## 📋 Notebook Structure

| # | Notebook | Description |
|---|---------|-------------|
| **01** | [`01_eda.ipynb`](01_eda.ipynb) | **Exploratory Data Analysis & Data Quality**<br>Data loading, cleaning, OHLCV explanation, returns, volatility, drawdowns, seasonality, stationarity |
| **02** | [`02_statistical_diagnostics.ipynb`](02_statistical_diagnostics.ipynb) | **Statistical Diagnostics & Stylized Facts**<br>ACF/PACF, Q-Q plots, normality tests, skewness/kurtosis, Ljung-Box, ARCH effect testing, white noise diagnostics |
| **03** | [`03_feature_engineering.ipynb`](03_feature_engineering.ipynb) | **Feature Engineering & Technical Indicators**<br>Rolling stats, lag features, momentum (RSI, MACD), volatility (ATR, Bollinger Bands), calendar features, regime indicators |
| **04** | [`04_classical_forecasting.ipynb`](04_classical_forecasting.ipynb) | **Classical Time Series Forecasting**<br>ARIMA/SARIMA family, Prophet, walk-forward validation, baseline forecasting |
| **05** | [`05_garch_volatility_modeling.ipynb`](05_garch_volatility_modeling.ipynb) | **Volatility Modeling with GARCH**<br>ARCH/GARCH, EGARCH, GJR-GARCH, conditional volatility forecasting, Value-at-Risk (VaR) |
| **06** | [`06_deep_learning.ipynb`](06_deep_learning.ipynb) | **Deep Learning Forecasting**<br>LSTM, GRU, sequence preparation, multi-step forecasting |
| **07** | [`07_anomaly_detection.ipynb`](07_anomaly_detection.ipynb) | **Anomaly & Regime Detection**<br>Isolation Forest, Autoencoders, market regime detection, extreme-event identification |
| **08** | [`08_model_evaluation.ipynb`](08_model_evaluation.ipynb) | **Model Evaluation & Comparison**<br>RMSE, MAE, MAPE, directional accuracy, residual diagnostics, backtesting, model comparison & final recommendations |

---

## 🛠 Tech Stack

**Core**  
- **Python 3.11** • pandas • NumPy • yfinance  
- **Visualization**: Plotly (interactive) + Matplotlib/Seaborn  

**Modeling**  
- statsmodels • pmdarima • Prophet  
- TensorFlow/Keras (LSTM, GRU)  
- scikit-learn (Isolation Forest, metrics)  

**Environment**  
- Conda (`environment.yml`) for full reproducibility  

---

## 🚀 How to Run the Project

```bash
# 1. Clone the repository
git clone https://github.com/Mena-Beshara/sp500-market-intelligence.git
cd sp500-market-intelligence

# 2. Create environment
conda env create -f environment.yml
conda activate sp500-intel

# 3. Run notebooks (recommended order)
jupyter lab