# Regime-Uncertainty Weighted Bayesian LASSO

This repository contains the code, result tables, figures, and manuscript materials for the project:

**Regime-Uncertainty Weighted Bayesian LASSO for Financial Return Prediction**

The project studies whether sparse Bayesian regression can be improved by making shrinkage depend on both:

1. market-regime uncertainty estimated from a hidden Markov model, and
2. regime-specific feature evidence estimated from predictive feature-importance models.

The key empirical conclusion is conditional. The proposed prior performs strongly when regime-specific sparse feature structures are present, especially in simulation. On real financial data, simpler sparse baselines such as Adaptive LASSO can achieve slightly lower prediction error, while the proposed method provides regime-specific posterior inference and uncertainty-aware feature selection.

## Repository Structure

```text
github_ready_package/
├── important_code/
│   ├── First.py
│   ├── regime_deep_analysis.py
│   ├── validate_regimes.py
│   ├── regime_feature_importance.py
│   ├── bayesian_lasso_experiment.py
│   ├── prior_feasibility_check.py
│   ├── prior_sensitivity_analysis.py
│   ├── simulation_study.py
│   ├── hyperparameter_selection.py
│   ├── robust_real_data_validation.py
│   ├── repeated_simulation_study.py
│   ├── rolling_window_validation.py
│   ├── statistical_significance_tests.py
│   ├── second_market_validation.py
│   └── requirements.txt
├── all_results/
│   ├── data/
│   ├── paper_figures/
│   └── manuscript_data_used/
├── paper/
│   ├── manuscript.tex
│   ├── manuscript.pdf
│   ├── PAPER_DATA_MANIFEST.md
│   ├── REGIME_UNCERTAINTY_WEIGHTED_BAYESIAN_LASSO.md
│   ├── REGIME_ANALYSIS_GUIDE.md
│   └── PUBLICATION_UPGRADE_PLAN.md
└── README.md
```

## Main Code Files

The main scripts are in `important_code/`.

| File | Purpose |
|---|---|
| `First.py` | Downloads market data, computes technical indicators, estimates HMM regimes, and builds the main dataset. |
| `regime_deep_analysis.py` | Produces deeper descriptive regime diagnostics and crisis-window checks. |
| `validate_regimes.py` | Validates whether detected regimes have distinct market behavior. |
| `regime_feature_importance.py` | Estimates regime-wise feature evidence using predictive models. |
| `bayesian_lasso_experiment.py` | Runs Classical LASSO, Standard Bayesian LASSO, and regime-adaptive Bayesian LASSO experiments. |
| `prior_feasibility_check.py` | Computes regime uncertainty, feature evidence, and the shrinkage matrix. |
| `prior_sensitivity_analysis.py` | Tests whether shrinkage rankings are stable across gamma and delta values. |
| `simulation_study.py` | Runs synthetic strong-regime and weak-regime experiments. |
| `hyperparameter_selection.py` | Performs validation-based selection of gamma and delta. |
| `robust_real_data_validation.py` | Adds stronger real-data baselines and ablation variants. |
| `repeated_simulation_study.py` | Repeats simulations across random seeds and reports average performance. |
| `rolling_window_validation.py` | Runs annual walk-forward validation on real data. |
| `statistical_significance_tests.py` | Computes paired significance tests and effect sizes for repeated simulations. |
| `second_market_validation.py` | Tests the method on a second market universe of sector ETFs. |

## Results Folder

The `all_results/` folder contains CSV, TXT, and PNG outputs.

Important result subfolders include:

| Folder | Contents |
|---|---|
| `all_results/data/bayesian_lasso/` | Main Bayesian LASSO coefficients, predictions, selected features, diagnostics, and model comparison. |
| `all_results/data/prior_feasibility_check/` | Regime uncertainty scores, feature evidence scores, and shrinkage matrix. |
| `all_results/data/prior_sensitivity_analysis/` | Gamma-delta sensitivity tables, rank correlations, lambda matrices, and heatmaps. |
| `all_results/data/simulation_study/` | Synthetic-data comparison results and feature-recovery outputs. |
| `all_results/data/repeated_simulation_study/` | Replication-level and average simulation performance. |
| `all_results/data/robust_real_data_validation/` | Real-data baselines, ablation results, selected features, and feature evidence. |
| `all_results/data/rolling_window_validation/` | Annual rolling-window validation tables and RMSE plot. |
| `all_results/data/statistical_significance_tests/` | Paired tests, confidence intervals, and effect-size summaries. |
| `all_results/data/second_market_validation/` | Sector ETF validation data and results. |
| `all_results/paper_figures/` | Figures used in the manuscript. |
| `all_results/manuscript_data_used/` | Clean result tables copied into the manuscript workflow. |

## Installation

Create and activate a Python environment, then install dependencies:

```bash
cd important_code
pip install -r requirements.txt
```

The project uses:

- pandas
- numpy
- yfinance
- hmmlearn
- scikit-learn
- matplotlib
- xgboost
- shap

## Suggested Run Order

Run the scripts from inside `important_code/`.

```bash
python First.py
python validate_regimes.py
python regime_deep_analysis.py
python regime_feature_importance.py
python bayesian_lasso_experiment.py
python prior_feasibility_check.py
python prior_sensitivity_analysis.py
python simulation_study.py
python hyperparameter_selection.py
python robust_real_data_validation.py
python repeated_simulation_study.py
python rolling_window_validation.py
python statistical_significance_tests.py
python second_market_validation.py
```

Note: the scripts download financial data through `yfinance`, so exact results may change if the upstream data provider revises historical prices.

## Manuscript Materials

The `paper/` folder contains:

- `manuscript.tex`: LaTeX source for the paper.
- `manuscript.pdf`: compiled manuscript snapshot.
- `REGIME_UNCERTAINTY_WEIGHTED_BAYESIAN_LASSO.md`: methodology and publication notes.
- `PAPER_DATA_MANIFEST.md`: mapping between manuscript tables/figures and output files.
- `PUBLICATION_UPGRADE_PLAN.md`: publication-readiness checklist and reviewer-facing improvements.

## Main Research Claim

The proposed method is not presented as a universal replacement for simpler sparse baselines. Its value is strongest when regime-specific feature relevance is meaningful. In those settings, the method provides:

- regime-specific posterior coefficients,
- uncertainty-aware feature selection,
- interpretable shrinkage values for each feature-regime pair,
- simulation evidence for improved prediction and feature recovery under strong regime-dependent sparsity.

On noisy real financial data, Adaptive LASSO and Elastic Net may remain better pure-prediction baselines. The proposed contribution is therefore methodological and inferential, not only predictive.

## Data and Reproducibility

The package includes generated CSV outputs and figures for review. If this repository is made public, consider adding:

- a Zenodo archive for the replication package,
- a DOI in the README,
- a license file,
- a citation file such as `CITATION.cff`.

