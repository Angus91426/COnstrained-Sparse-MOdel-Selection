"""
Simulation study, Section 5.3-5.4 (Tables 9, 10 and 12).

Run from the repository root:

    python run_simulation.py

Every method (PLS, SPLS, COSMOS and the six L0Learn variants) is tuned by 300 Optuna trials of
5-fold cross-validation for n = 55, 110, ..., 330. Each call writes
Output/Irrelevant Nonlinear/_<p>/Optimal_<timestamp>.csv (one row per method, sample size and
fold) and the per-trial logs under Log/.

Related scripts (also run from the repository root):
    python -m Python.CV_baseline               # baseline scenario p = 6, Table 8 (grid search)
    python -m Python.Simulation_HR 500         # HR at COSMOS's k and lambda, p = 530, Table 11
    python -m Python.Simulation_HR 5000        # same for p = 5030, Table 13
    python -m Python.Simulation_Result         # per-fold summaries and averaged tables

Requirements: COSMOS_A1 / COSMOS_A2 (see Python/wat_constants.py), Gurobi for HR, and
R with the L0Learn package for L0Learn.
"""
from Python.Simulation_CV import main

if __name__ == '__main__':
    for num_features in ( 500, 5000 ):           # p = 530 and p = 5030 with the 30 nonlinear terms
        main( sim_name = "Irrelevant Nonlinear", num_features = num_features, n_splits = 5,
              multiplier_range = range( 1, 7 ),  # n = 55 * multiplier
              run_HR = False, run_Relax_HR = False,
              run_PLS = True, run_SPLS = True, run_COSMOS = True, run_L0Learn = True )
