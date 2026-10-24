# COSMOS: Constrained Sparse Model Selection

Code, synthetic data, and per-instance results for the paper

> Ting-Yu Liao, Chia-Cheng Chen, and Chih-Hua Hsu. *Constrained Sparse Model Selection: Unifying Exact Sparsity and Physics-Informed Constraints for Interpretable High-Dimensional Regression.* Submitted to the *European Journal of Operational Research*.

COSMOS solves sparse linear regression with exact cardinality control, linear inequality constraints on the coefficients (`A beta >= b`), and ridge regularization. The inner phase solves each fixed-support quadratic program exactly through a small nonnegative least-squares problem; the outer phase updates the support by gradient-guided swaps. The implementation is the class `Constrained_Sparse_Model_Selection` in [`Python/models_.py`](Python/models_.py).

## Repository layout

| Path | Contents |
|---|---|
| `Python/models_.py` | COSMOS, Holistic Regression (HR, Gurobi), PLS, SPLS, and the L0Learn wrapper (R via rpy2) |
| `Python/global_.py` | Normalization and evaluation metrics (test R², constraint satisfaction, sparsity) |
| `Python/wat_constants.py` | Reads the WAT proportionality constants `a1`, `a2` (see below) |
| `Python/create_data.py` | Synthetic WAT data generator (Section 5.1.1) |
| `Python/CV_baseline.py` | Baseline scenario, p = 6, grid search (Table 8) |
| `Python/Simulation_CV.py` | Optuna-tuned 5-fold CV for p = 530 and 5030 (Tables 9, 10, 12) |
| `Python/Simulation_HR.py` | HR at the k and λ selected for COSMOS (Tables 11, 13) |
| `Python/SA_Warm_Start.py`, `Python/SA_alpha.py` | Sensitivity analyses (Tables 14, 15) |
| `Python/Real_CV.py` | Corn and riboflavin applications (Table 17) |
| `Python/NIR_Corn_constraints.py`, `Python/Riboflavin_constraints.py` | Build the constraint files of Table 16 |
| `Python/Simulation_Result.py`, `Python/Real_Result.py` | Turn per-fold results into summary tables |
| `verify_inner_solver.py` | Numerical check of the inner solver against Clarabel and the KKT conditions (Remark 1) |
| `run_simulation.py`, `run_sensitivity.py`, `run_real_data.py` | Entry points for Sections 5.3–5.4, 5.5, and 6 |
| `Data/` | Synthetic datasets, constraint files, and scripts to prepare and check the public datasets |
| `Output/` | Per-instance (per-fold) results reported in the paper |
| `Results/Tables/` | Per-fold summaries and averaged tables behind Tables 8–17 |

## Installation

The experiments in the paper were run on Windows with Python 3.11.16, R 4.6.1, L0Learn 2.1.0, and Gurobi 13.0.3 (gurobipy); the Python package versions are pinned in `requirements.txt`.

1. Python 3.11 or newer (the pinned pandas 3.0 requires it): `pip install -r requirements.txt`.
2. **Gurobi** (for HR) with a valid licence; free academic licences are available from Gurobi.
3. **R** with the package `L0Learn`: in R, run `install.packages("L0Learn")`. `Python/models_.py` locates R automatically; if it does not, set the environment variable `R_HOME`.

On Windows with a non-English system locale (for example Traditional Chinese, code page cp950), rpy2 can fail with a `UnicodeDecodeError` when it starts R. Run Python in UTF-8 mode to avoid this: `set PYTHONUTF8=1` (cmd) or `$env:PYTHONUTF8=1` (PowerShell) before running the scripts.

Note that `Python/models_.py` imports `gurobipy` and loads L0Learn through `rpy2` when it is imported, so Gurobi, R, and L0Learn must be installed even if you only run COSMOS.

All scripts must be run **from the repository root**, either directly (`python run_simulation.py`) or as modules (`python -m Python.CV_baseline`).

## Using COSMOS on your own data

```python
import numpy as np
from Python.global_ import Normalize
from Python.models_ import Constrained_Sparse_Model_Selection

# Centre the data and scale every column of X to unit length, using training statistics.
X_tr, X_te, y_tr, y_te = Normalize( X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test )

# Linear constraints A beta >= b, one row per constraint. Example: beta_0 >= 1e-4 and beta_1 - beta_2 >= 1e-4.
# Every feature in a row with b_i > 0 is mandatory and always kept in the model.
p = X_tr.shape[1]
A = np.zeros( ( 2, p ) ); A[0, 0] = 1; A[1, 1], A[1, 2] = 1, -1
b = np.full( 2, 1e-4 )

model = Constrained_Sparse_Model_Selection( k = 10, alpha_ = 0.1, lambda_ = 1e-3, A = A, b = b )
model.fit_model( X_tr, y_tr, initial_sol = model.Initial_Sol( X_train = X_tr, y_train = y_tr ) )
y_pred = model.predict( X_te )
beta = model.beta_          # the nonzero entries are the selected features
```

`k` is the maximum model size, `alpha_` the weight of the covariance-alignment term, and `lambda_` the ridge parameter (Section 3.1); in the paper they are tuned by cross-validation. The optional argument `group_sparsity` (a list of index lists) allows at most one feature per group.

## WAT proportionality constants

The WAT constraints (6)–(7) of the paper, `beta_1 >= a1 * beta_3 + eps` and `beta_2 >= a2 * beta_3 + eps`, use constants that come from the second author's domain expertise. **Their values are withheld to protect trade secrets** (footnote 2 of the paper) and do not appear anywhere in this repository. To run the WAT experiments, set your own values:

```bash
export COSMOS_A1=1.2   # example values only
export COSMOS_A2=1.0
```

Proposition 3 of the paper requires `a2 >= 0` and `a1 <= a2 + 1`. The real-data experiments do not use these constants. Because the constants are withheld, re-running the WAT experiments reproduces the method and the protocol, but not the exact numbers in Tables 8–15; the numbers themselves are provided in `Output/` and `Results/Tables/`.

## Reproducing the paper

| Paper | Command |
|---|---|
| Synthetic data, Section 5.1.1 | Included in `Data/` (see below) |
| Table 8 (baseline, p = 6) | `python -m Python.CV_baseline` |
| Tables 9, 10, 12 (p = 530, 5030) | `python run_simulation.py` |
| Tables 11, 13 (HR at fixed hyperparameters) | `python -m Python.Simulation_HR 500` and `python -m Python.Simulation_HR 5000` |
| Tables 14, 15 (sensitivity analysis) | `python run_sensitivity.py` |
| Table 16 (real-data constraints) | `python -m Python.NIR_Corn_constraints` and `python -m Python.Riboflavin_constraints` |
| Table 17 (corn, riboflavin) | `python run_real_data.py` |
| Remark 1 (inner-solver check) | `python verify_inner_solver.py` (needs `COSMOS_A1`/`COSMOS_A2` and the real-world data; overwrites `Output/Verify_Inner_Solver/`) |
| Summary tables | `python -m Python.Simulation_Result` and `python -m Python.Real_Result` |

All random draws (data generation, cross-validation folds, Optuna's TPE sampler) use the fixed seed `11220426`. Wall-clock times, the Optuna pruning sequence, and HR fits that stop at the one-hour Gurobi time limit depend on the hardware and software versions, so these quantities can differ between machines; the values reported in the paper are in `Output/` and `Results/Tables/`.

**Synthetic data.** All datasets used in the paper are included: the base datasets (p = 6) in `Data/Base Data/` and the high-dimensional datasets (p = 530 and 5030) in `Data/Irrelevant Nonlinear/`. Use these files to reproduce the paper's numbers. The generator `Python/create_data.py` documents the design of Section 5.1.1:

- `python -m Python.create_data 500 --overwrite` (and `5000`) builds the high-dimensional datasets from the base datasets in under a minute. The response, the six WAT parameters, and the 30 nonlinear terms match the shipped files to numerical precision. The distractors follow the same design but are a different random draw: the shipped files were generated with this procedure, but the random seed used for their distractors was not recorded. Without `--overwrite`, the shipped files are left untouched.
- Setting `base_data = True` in the script reruns the full weight search of Section 5.1.1 for the base datasets, which takes several hours. The shipped base datasets were selected by the same criterion over several seeded runs, so a rerun gives datasets of the same quality but not necessarily the same files.

The scripts write new results with a new timestamp (`Optimal_<timestamp>.csv`) next to the published ones. `Simulation_HR.py` overwrites `HR_Simulation.csv`, and the post-processing scripts write to `Results/Tables_regenerated/`, so the published tables are kept.

## Real-world data

The two public datasets are not redistributed here. Before running `run_real_data.py`, `verify_inner_solver.py`, or the constraint scripts:

- **Corn**: download `corn.mat` from [Eigenvector Research](https://eigenvector.com/resources/data-sets/nir-of-corn-samples-for-standardization-benchmarking/), save it as `Data/NIR Corn/corn.mat`, and run `python Data/prepare_corn.py`.
- **Riboflavin**: run `python Data/prepare_riboflavin.py`. The data were distributed with the R package `hdi`, which was archived on CRAN in July 2026; the script downloads the archived source package `hdi_0.1-9.tar.gz` from the CRAN archive and reads the data file from it, so `hdi` itself does not need to be installed. `Rscript Data/prepare_riboflavin.R` does the same without Python.

Then confirm that you have exactly the data used in the paper:

```bash
python Data/check_real_data.py
```

The check compares the shape, the column names and order, the sample ids and **row order** (the cross-validation folds depend on it), and the values of each file against `Data/real_data_fingerprint.json`, allowing only for differences in how numbers are written to text.

The constraint files used in the paper are included in `Data/NIR Corn/Constraints/` and `Data/Riboflavin/Constraints/`. The files ending in `_optional` add constraints that were considered but not used in the paper.

## Result files

Each `Optimal_<timestamp>.csv` in `Output/` contains one row per method, sample size, and cross-validation fold, for the hyperparameters selected by cross-validation:

| Column | Meaning |
|---|---|
| `Model` | Method (`COSMOS`, `HR`, `PLS`, `SPLS`, `L0Learn_<penalty>_<algorithm>`) |
| `Param1`–`Param3` | Selected hyperparameters: COSMOS (k, α, λ); HR (k, Γ, Γ); PLS (components); SPLS (K, η, η); L0Learn (support size, λ, γ) |
| `CV`, `Sample size`, `Features` | Fold index (1–5), n, and p |
| `Time` | Fit time in seconds for this fold |
| `train/test MAPE`, `train/test R2` | Prediction metrics |
| `Constraint Satisfaction` | Fraction of the constraint rows satisfied by the fitted coefficients |
| `Sparsity` | Number of nonzero coefficients |
| `X1`, `X2`, … (or wavelengths / gene names) | Fitted coefficients |

**WAT result files:** in the WAT scenarios (`Output/Baseline/`, `Output/Irrelevant Nonlinear/`), the columns `X1`–`X6` hold **selection indicators** (1 = nonzero, 0 = zero) instead of the coefficients of the six WAT parameters, because the fitted values would reveal the withheld constants `a1` and `a2`. All other coefficients are reported as fitted.

`HR_Simulation.csv` has the same layout for HR at fixed hyperparameters and adds the Gurobi `MIPGap`. The sensitivity results (`Output/SA_Warm_Start/`, `Output/SA_alpha/`) and the inner-solver check (`Output/Verify_Inner_Solver/`) are summaries per configuration. The per-trial Optuna logs are not included because of their size; the total search time of each run (`CV Time`) is reported in `Results/Tables/*_Final.xlsx`.

## Citation

If you use this code, please cite the paper above (full reference to be added on publication). Citation metadata is in [`CITATION.cff`](CITATION.cff), which GitHub shows as "Cite this repository".

## Contact

Chih-Hua Hsu (corresponding author), Department of Industrial and Systems Engineering, Chung Yuan Christian University — chhsu135@cycu.edu.tw
