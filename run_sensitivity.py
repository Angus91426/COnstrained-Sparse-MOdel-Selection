"""
Sensitivity analysis, Section 5.5 (Tables 14 and 15).

Run from the repository root:

    python run_sensitivity.py

Both analyses reuse the hyperparameters selected in the simulation runs whose results are
stored in Output/Irrelevant Nonlinear/_<p>/Optimal_<optimal_id>.csv. After re-running
run_simulation.py, replace the ids below with the timestamps of your own runs.
"""
from Python.SA_Warm_Start import main as warm_start_main
from Python.SA_alpha import main as alpha_main

OPTIMAL_IDS = {
    500:  "20260915-151925",   # p = 530
    5000: "20260915-153244",   # p = 5030
}

if __name__ == '__main__':
    for num_features, optimal_id in OPTIMAL_IDS.items():
        warm_start_main( sim_name = "Irrelevant Nonlinear", num_features = num_features, optimal_id = optimal_id )
        alpha_main( sim_name = "Irrelevant Nonlinear", num_features = num_features, optimal_id = optimal_id )
