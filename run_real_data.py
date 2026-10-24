"""
Real-world applications, Section 6 (Tables 16 and 17).

Prepare the data first (see README, "Real-world data"), then run from the repository root:

    python run_real_data.py
    python -m Python.Real_Result        # per-fold summaries and averaged tables

Each response writes Output/Real/<dataset>/<response>/Optimal_<timestamp>.csv.
L0Learn requires R with the L0Learn package.
"""
from Python.Real_CV import main

if __name__ == '__main__':
    methods = dict( run_HR = False, run_Relax_HR = False, run_PLS = True, run_SPLS = True,
                    run_COSMOS = True, run_L0Learn = True, n_trials = 300 )

    # Near-infrared corn spectra (m5 instrument), four responses.
    main( dataset = "NIR Corn", responses = ['moisture', 'oil', 'protein', 'starch'], n_splits = 5,
          log_coef = True, **methods )

    # Riboflavin production (p = 4088); coefficient logging is switched off to keep the logs small.
    main( dataset = "Riboflavin", responses = ['riboflavin'], n_splits = 5,
          log_coef = False, **methods )
