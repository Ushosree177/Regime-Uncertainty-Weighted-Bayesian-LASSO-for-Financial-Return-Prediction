# Does Regime-Aware Shrinkage Improve Equity-Return Forecasts?

This repository contains the analysis code and saved results for **“Does Regime-Aware Shrinkage Improve Equity-Return Forecasts? A Leakage-Safe Evaluation Finds No Robust Gain.”** The manuscript is not included. The study asks whether regime-aware, data-dependent Bayesian shrinkage improves five-day equity-return forecasts beyond well-tuned pooled and regime-specific models. Across the reported financial tests, it finds **no detectable forecasting gain** from the proposed weighting.

## Quick links

- [Download the code-and-results ZIP](https://github.com/Ushosree177/Regime-Uncertainty-Weighted-Bayesian-LASSO-for-Financial-Return-Prediction/raw/refs/heads/main/reproducibility_package_code_results.zip)
- [GitHub repository](https://github.com/Ushosree177/Regime-Uncertainty-Weighted-Bayesian-LASSO-for-Financial-Return-Prediction)
- [GitHub releases](https://github.com/Ushosree177/Regime-Uncertainty-Weighted-Bayesian-LASSO-for-Financial-Return-Prediction/releases)
- [Zenodo concept DOI (resolves to the latest version)](https://doi.org/10.5281/zenodo.23560477)

## What is in this archive

- `important_code/` contains Python scripts for preparing data, fitting and tuning models, running simulations, evaluating forecasts, and checking diagnostics.
- `data/` contains saved aggregate results, simulation replication outputs, tuning records, and diagnostic summaries used to check the paper's reported results.
- `important_code/PLAIN_LANGUAGE_ANALYSIS_GUIDE.md` explains the main analysis steps in accessible language.
- `important_code/requirements.txt` lists Python package dependencies.

This is a **code-and-results archive**, not a complete raw-data mirror. The manuscript, manuscript figures, downloaded market-price files, derived ticker-level market panels, and row-level financial prediction files are not included. This avoids redistributing market data. The data scripts download observations through `yfinance` (`auto_adjust=False`). A provider may revise historical data or ticker coverage, so a fresh download can differ from the archived results.

## Study data checked against the manuscript

The primary equity panel reported in the manuscript contains 160,197 ticker-date observations from 31 U.S. symbols, from 1 February 2005 to 31 December 2025. Its reported regime counts are:

| Regime | Observations |
| --- | ---: |
| Recovery | 65,621 |
| Bull | 40,970 |
| Bear | 36,541 |
| Crisis | 17,065 |
| **Total** | **160,197** |

The sector-market evaluation uses 11 sector ETFs, with SPY used only as the market-regime proxy and excluded from the ETF regression panel. The final manuscript reports 11 ETFs and 8,734 test forecasts, but does not report the total input-panel row count. A local downloaded panel snapshot has 44,508 ticker-date rows in total: 4,004 for SPY and 40,504 across the 11 ETFs. This local count is provided for transparency and is not a manuscript-reported sample size; the derived panel and individual forecasts are not included in this public archive, and a fresh download can have different coverage.

## Environment and reproducibility notes

The dependency file is intentionally unpinned. The following versions were observed in the Windows environment used for the current local package check; they are provided as a reference, not as a lockfile or a claim that every archived result was generated under precisely these versions:

- Python 3.13.6
- NumPy 2.2.6; pandas 3.0.3; scikit-learn 1.9.0; hmmlearn 0.3.3
- yfinance 1.3.0; SciPy 1.17.1; statsmodels 0.14.5
- Matplotlib 3.10.9; XGBoost 3.1.1; SHAP 0.52.0
- Operating system: Windows 11 (build 26200); installed RAM: approximately 7.8 GB
- Full end-to-end runtime: not recorded. It depends on data-download speed and the time required for repeated simulations, Bayesian tuning and sampling, and rolling validation.

The repeated simulation study uses 30 replications in each of seven designs. Main simulation seed constants and derived seeds are defined in the relevant scripts (the repeated simulation uses base seed 42). Four-chain diagnostic seeds are listed in `final_setting_mcmc_diagnostics.py`. For exact details, consult the constants and seed construction in each script; not every script uses the same random-number generator or seed schedule.

## Run instructions (Windows)

### 1. Extract to a short folder

Download the ZIP from the repository or Zenodo and extract it to a short path such as `C:\rp`. Short paths help avoid Windows path-length errors. Open PowerShell and change into the extracted archive root (the folder containing `important_code` and `data`):

```powershell
Set-Location C:\rp
```

### 2. Create an environment and install the packages

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\important_code\requirements.txt
```

If PowerShell blocks environment activation, use `C:\rp\.venv\Scripts\python.exe` in place of `python` for the following commands.

### 3. Reproduce the paired simulation tests

The archive contains saved simulation outputs. Recreate the seven paired full-prior versus regime-specific Bayesian LASSO RMSE tests and their Holm adjustment with:

```powershell
python .\important_code\paired_replication_tests.py
```

The paired-test script runs directly from the bundled replication outputs. The simulation table updater also uses bundled CSVs, but edits `springer_manuscript.tex`, which is intentionally not included. To use it, place your own manuscript copy one folder above the archive root; it edits that copy in place, so work from a backup. Neither script reruns real-market model fits.

### 4. Rerun an analysis

The analysis scripts create intermediate data and may download market observations. Run them from the archive root so their relative `data/` paths work. Follow the run order in `important_code/PLAIN_LANGUAGE_ANALYSIS_GUIDE.md`, and inspect a script's input/output paths before running it. Some scripts require files generated by earlier steps. A complete fresh rerun can take substantial time; no verified end-to-end runtime is available.

Useful commands for individual analyses include:

```powershell
python .\important_code\First.py
python .\important_code\final_setting_mcmc_diagnostics.py
python .\important_code\repeated_simulation_study.py
python .\important_code\pooled_regime_interaction_simulation.py
python .\important_code\robust_real_data_validation.py
python .\important_code\rolling_window_validation.py
python .\important_code\rolling_multiyear_tuning.py
python .\important_code\second_market_validation.py
```

Some analyses need large computation or outputs from preceding scripts. The commands above are not a promise that each script can be run independently or in any arbitrary order. The detailed guide and the input/output paths in each script are the source for dependencies and run sequence.

## Where the paper's results come from

Paths below are relative to the archive root. “Rebuild without data download?” distinguishes reproducing a result from inspecting the saved output: for real-market results, the summary CSV may be available even though the row-level inputs needed to recompute it are excluded.

| Paper result | Main script(s) | Saved output in `data/` | Rebuild without data download? |
| --- | --- | --- | --- |
| Table 1: regime counts | `First.py` | Primary ticker-date panel is not included | No; the market panel must be downloaded and rebuilt |
| Table 2: regime characteristics | `regime_deep_analysis.py` | Summary CSV is not included in this archive version | No; the market panel must be downloaded and rebuilt |
| Table 3: regime uncertainty scores | `robust_real_data_validation.py` | `robust_real_data_validation/real_data_uncertainty_scores.csv` | Inspect saved scores; recomputation requires market data |
| Table 4: four-chain MCMC diagnostics | `final_setting_mcmc_diagnostics.py` | `robust_real_data_validation/final_four_chain_mcmc_diagnostics.csv` | Inspect saved diagnostics; rerunning requires market data |
| Table 5: primary-equity prior-setting validation | `tune_bayesian_lambda0.py` | `bayesian_lambda0_tuning/gamma_delta_validation_scores.csv`, `selected_model_config.json` | Inspect saved scores; recomputation requires market data |
| Tables 6–9: simulation prediction performance | `repeated_simulation_study.py`; `pooled_regime_interaction_simulation.py`; `update_simulation_tables.py` | `repeated_simulation_study/average_model_metrics.csv`; `pooled_regime_interaction_simulation/summary.csv`; `moderate_summary.csv` | Bundled simulation summaries are inspectable; synthetic simulations rerun without market data. To insert rebuilt TeX tables, supply the manuscript file (not included). |
| Tables 10–13: simulation feature recovery | `repeated_simulation_study.py`; `pooled_regime_interaction_simulation.py`; `update_simulation_tables.py` | `repeated_simulation_study/average_feature_recovery.csv`; interaction-simulation summaries | Bundled simulation summaries are inspectable; synthetic simulations rerun without market data. To insert rebuilt TeX tables, supply the manuscript file (not included). |
| Table 14: paired simulation tests | `paired_replication_tests.py` | `repeated_simulation_study/paired_contrast_tests.csv`; replication-level inputs are included | Yes; rerun the paired tests from bundled simulation outputs |
| Table 15: primary equity comparison | `robust_real_data_validation.py` | `robust_real_data_validation/real_data_stronger_baselines.csv` | Inspect saved summary only; recomputation requires market data and excluded row-level forecasts |
| Table 16: primary forecast-comparison tests | `forecast_comparison_tests.py` | `robust_real_data_validation/fixed_split_predictions_dm_tests.csv` | Inspect saved tests only; prediction-level inputs are excluded |
| Table 17: annual rolling-window comparison | `rolling_window_validation.py` | `rolling_window_validation/rolling_window_results.csv`, `rolling_window_summary.csv` | Inspect saved results only; recomputation requires market data. The year-level file also records selected λ₀, γ, and δ for each model/year. |
| Table 18: pooled rolling forecast tests | `forecast_comparison_tests.py` | `rolling_window_validation/rolling_window_predictions_dm_tests.csv` | Inspect saved tests only; prediction-level inputs are excluded |
| Table 19: one-year versus three-year tuning robustness | `rolling_multiyear_tuning.py` | `rolling_multiyear_tuning/rolling_results.csv`, `rolling_summary.csv` | Inspect saved results only; recomputation requires market data |
| Table 20: sector ETF comparison | `second_market_validation.py` | `second_market_validation/sector_etf_model_comparison.csv` | Inspect saved summary only; recomputation requires market data and excluded row-level inputs |
| Figure 1: evaluation workflow | `draw_pipeline_figure.py` | Figure image is in the manuscript materials, not this archive | Regenerate the figure without a data download |
| Figure 2: SPY regime timeline | `regime_deep_analysis.py` | Figure image is in the manuscript materials, not this archive | Requires market data |
| Figure 3: feature-evidence heatmap | `regime_feature_importance.py` | Figure image is in the manuscript materials, not this archive | Requires market data |
| Figure 4: shrinkage-weight illustration | `prior_sensitivity_analysis.py` | Heatmap image is in the manuscript materials, not this archive | Regenerate the illustrative heatmap without a data download |

Table numbers refer to the manuscript as submitted. The compiled manuscript table labels run from 1 through 20, and there is no separate table of year-by-year selected hyperparameters. The yearly output for Table 17 includes the selected values of λ₀, γ, and δ.

## What the available checks do and do not establish

The saved aggregate and replication-level outputs allow inspection of the reported results. The paired simulation test script can be rerun from its saved input data. The simulation table-generation script can update a local manuscript copy from bundled CSVs, but the manuscript itself is not included. This is not the same as reproducing every estimate from raw market data in a clean environment: the raw vendor data, derived market panels, and row-level financial predictions are excluded, and the online provider can change its history.

## Licences

- Python code is covered by the [MIT License](LICENSE).
- Documentation and saved result files are covered by [CC BY 4.0](LICENSE-CC-BY-4.0.md), unless a file states otherwise. Third-party market data are not included or relicensed by these terms.
## How to cite

Raha, U. (2026). *Code and results for the regime-aware shrinkage study* [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.23560477

The DOI above is the concept DOI and resolves to the latest version. The exact version DOI is shown on that version's Zenodo record.



## Contact

Ushosree Raha  
Department of Engineering Sciences, Atal Bihari Vajpayee Indian Institute of Information Technology and Management, Gwalior, India  
Email: [ms_2025027@iiitm.ac.in](mailto:ms_2025027@iiitm.ac.in)

