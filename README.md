# PowerGuard-BD

Official implementation and dataset for **PowerGuard-BD: Explainable Multi-Task Learning for Electricity Demand and Load-Shedding Risk Forecasting to Support Business Continuity and Operational Resilience in Bangladesh**.

## Overview

PowerGuard-BD is an explainable multi-task forecasting framework for the Bangladesh power system. The framework jointly performs:

- Electricity demand forecasting
- Load-shedding risk prediction
- Power-deficit magnitude estimation

The proposed **PowerGuardX-CMoE-RiskGuard** framework combines leakage-safe multi-scale historical inputs, a feature-gated Transformer-based temporal core, fuel-mix and exogenous context representations, contextual mixture-of-experts selection, and validation-gated risk decision logic.

## Dataset

The PowerGuard-BD dataset contains approximately **94,934 hourly observations** spanning **April 2015 to March 2026**.

The dataset integrates:

- Electricity demand
- Electricity generation
- Load shedding
- Power deficit
- Fuel-wise generation
- Cross-border electricity imports
- Meteorological variables
- Calendar and temporal variables
- Leakage-safe lag and persistence features

Dataset file:

`PowerGuard_BD_Dataset.zip`

## Research Notebook

The complete modeling, evaluation, explainability, ablation, statistical testing, and robustness-analysis pipeline is provided in:

`PowerGuard.ipynb`

## Main Model

**PowerGuardX-CMoE-RiskGuard**

The framework contains:

- Feature-gated historical representation
- Learnable lag embeddings
- Transformer encoder
- Temporal attention pooling
- Fuel-context branch
- Exogenous-context branch
- Multi-task prediction heads
- Contextual Mixture-of-Experts (CMoE)
- RiskGuard validation-gated classification module

## Evaluation Tasks

### Electricity Demand Forecasting
Metrics include:
- MAE
- RMSE
- MAPE
- R²

### Load-Shedding Risk Prediction
Metrics include:
- Accuracy
- Balanced Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Brier Score

### Power-Deficit Estimation
Metrics include:
- MAE
- RMSE
- MAPE
- R²

## Final Test Results

The final PowerGuardX-CMoE-RiskGuard framework achieved:

- **Demand MAE:** 350.46 MW
- **Demand RMSE:** 485.55 MW
- **Demand MAPE:** 3.16%
- **Demand R²:** 0.9523
- **Load-Shedding Risk Accuracy:** 93.33%
- **Risk F1-score:** 91.58%
- **Risk ROC-AUC:** 0.9615
- **Risk PR-AUC:** 0.9375
- **Power-Deficit MAE:** 72.13 MW
- **Power-Deficit RMSE:** 199.17 MW

## Explainability and Robustness

The repository includes analyses based on:

- Temporal attention
- Learned variable gates
- SHAP feature importance
- Ablation analysis
- Bootstrap confidence analysis
- Statistical significance testing
- Year-wise robustness
- Seasonal robustness
- Peak-demand-period robustness

## Requirements

Main Python packages include:

```text
numpy
pandas
scipy
scikit-learn
matplotlib
xgboost
lightgbm
shap
statsmodels
torch
