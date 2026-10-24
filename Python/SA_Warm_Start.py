import warnings
warnings.filterwarnings( "ignore" )

import numpy as np, pandas as pd
import os, time, glob

from rich import print
from rich.text import Text
from sklearn.model_selection import KFold
from Python.global_ import Normalize, Metric
from Python.wat_constants import a1, a2
from Python.models_ import Constrained_Sparse_Model_Selection

# ──────────────────────────────────────────────────────────────────────────────
# SA_COSMOS.py
#
# Sensitivity analysis on the initial solution for the COSMOS model.
#
# For each simulation regime (sim_name × num_features), the script:
#   1. Reads the optimal (k, alpha, lambda) for COSMOS at each sample size
#      from the previously saved Output/{sim_name}/_{num_features}/Optimal_*.csv.
#   2. Skips any sample size whose optimal k == 6: when k == 6 there are no
#      optional features to select (k − 6 = 0), so every strategy produces an
#      identical initial support and the sensitivity analysis is uninformative.
#   3. For sample sizes with k > 6, re-runs 5-fold CV with four strategies:
#        • Correlation  – correlation-based screening (Initial_Sol)        [1 run]
#        • Ridge        – ridge-regression-based screening                 [1 run]
#        • Gradient     – add-score / gradient-based screening             [1 run]
#        • Random       – random feature sampling (n_random runs, seeded)
#   4. Records per-fold metrics in a parquet log and writes a CSV summary
#      with per-run means aggregated across CV folds, then across runs for
#      the Random strategy.
#
# Outputs
# -------
# Log/SA_COSMOS/{sim_name}/_{num_features}/{sample_size}_{csv_id}.parquet
# Output/SA_COSMOS/{sim_name}/_{num_features}/Summary_{csv_id}.csv
# ──────────────────────────────────────────────────────────────────────────────

# Matrix form constraints (identical to simulation_CV.py)
_A = np.array([
    [0,  0,     1, 0,  0,  0],
    [0,  0,     0, 0,  0,  1],
    [1, -1,     0, 0,  0,  0],
    [1,  0, -a1(), 0,  0,  0],
    [0,  1, -a2(), 0,  0,  0],
    [0,  0,     0, 1, -1,  0],
    [0,  0,     0, 0,  1, -1]
])
_epsilon = 1e-4
_b = np.array( [_epsilon] * 7 )


def main(
    sim_name:     str,
    num_features: int,
    optimal_id:   str,
    n_splits:     int = 5,
    n_random:     int = 30
):
    """
    Run COSMOS initial-solution sensitivity analysis for one simulation regime.

    Parameters
    ----------
    sim_name     : Regime name matching the folder structure, e.g. "Random",
                   "Irrelevant", "Random Nonlinear", "Irrelevant Nonlinear".
    num_features : Number of features (500 or 5000).
    n_splits     : Number of CV folds (default 5).
    n_random     : Number of random-initialisation repetitions (default 30).
    """

    csv_id = time.strftime( "%Y%m%d-%H%M%S", time.localtime() )

    # List of indices for group sparsity selection constraints
    if 'Nonlinear' in sim_name:
        group_sparsity = [
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
            [26, 27, 28, 29, 30],
            [31, 32, 33, 34, 35]
        ]
    else:
        group_sparsity = None

    # ── Create output directories ────────────────────────────────────────────
    log_dir = f'Log/SA_Warm_Start/{sim_name}/_{num_features}'
    out_dir = f'Output/SA_Warm_Start/{sim_name}/_{num_features}'
    os.makedirs( log_dir, exist_ok = True )
    os.makedirs( out_dir, exist_ok = True )

    # ── Load optimal hyperparameters ─────────────────────────────────────────
    opt_pattern = f'Output/{sim_name}/_{num_features}/Optimal_{optimal_id}.csv'
    opt_files   = sorted( glob.glob( opt_pattern ) )
    if not opt_files:
        raise FileNotFoundError(
            f"No Optimal_{optimal_id}.csv found at '{opt_pattern}'. "
            "Run simulation_CV.main() for this regime first."
        )
    # Use the most recent file
    opt_path = opt_files[-1]
    print( Text( f"Loading optimal hyperparameters from: {opt_path}", style = "bright_cyan" ) )

    opt_full = pd.read_csv( opt_path )

    # One unique (k, alpha, lambda) per sample size for COSMOS
    cosmos_rows = opt_full[ opt_full['Model'] == 'COSMOS' ].copy()
    if cosmos_rows.empty:
        raise ValueError( f"No COSMOS rows found in {opt_path}." )

    cosmos_params = (
        cosmos_rows
        .drop_duplicates( subset = ['Sample size'] )
        [['Sample size', 'Param1', 'Param2', 'Param3']]
        .rename( columns = { 'Param1': 'k', 'Param2': 'alpha', 'Param3': 'lam' } )
        .sort_values( 'Sample size' )
        .reset_index( drop = True )
    )
    print( Text( f"Found optimal COSMOS params for {len( cosmos_params )} sample size(s).", style = "bright_cyan" ) )
    print( cosmos_params.to_string( index = False ) )

    # ── Cross-validation splitter (same seed as simulation_CV.py) ────────────
    kf = KFold( n_splits = n_splits, shuffle = True, random_state = 11220426 )

    # Seeds for the deterministic strategies (seed only affects the normal
    # noise used for the initial coefficient values, not the support selection)
    fixed_seed = 11220426
    rng = np.random.default_rng( seed = fixed_seed )

    # ── Initial-solution strategies ──────────────────────────────────────────
    # Each entry: ( display_name, method_name_on_instance, seeds_list )
    # The method is called as: getattr(cosmos_instance, method_name)(X_tr, y_tr, seed=s)
    strategies = [
        ( 'Correlation', 'Initial_Sol',            [fixed_seed] ),
        ( 'Ridge',       '_ridge_initial_sol',     [fixed_seed] ),
        ( 'Gradient',    '_gradient_initial_sol',  [fixed_seed] ),
        ( 'Random',      '_random_initial_sol',    rng.integers( 0, 2**32, size = n_random ).tolist() ),
    ]

    # ── Main loop: iterate over each sample size ─────────────────────────────
    all_summaries = pd.DataFrame()

    for _, param_row in cosmos_params.iterrows():
        sample_size = int(   param_row['Sample size'] )
        k           = int(   param_row['k']           )
        alpha_      = float( param_row['alpha']       )
        lambda_     = float( param_row['lam']         )

        # When k == 6 all selected features are the mandatory six; no optional
        # features are included (k - 6 = 0), so every initial-solution strategy
        # produces an identical support.  Sensitivity analysis is meaningless here.
        if k == 6:
            print( Text(
                f"\nSample size: {sample_size} | k = {k}  →  skipped "
                f"(k = 6: no optional features selected, all strategies identical).",
                style = "bright_yellow"
            ) )
            continue

        print()
        print( Text( f"{'=' * 80}", style = "bright_white" ) )
        print( Text(
            f"Sample size: {sample_size} | k = {k}, alpha = {alpha_:.6f}, lambda = {lambda_:.6f}",
            style = "bright_white"
        ) )
        print( Text( f"{'=' * 80}", style = "bright_white" ) )

        # Load data
        data_path = f'Data/{sim_name}/{num_features}_p/{sample_size}.csv'
        data = pd.read_csv( data_path )
        X = data.iloc[:, 1:].values
        y = data.iloc[:, 0].values

        log_df = pd.DataFrame()

        for ( method_name, attr_name, seeds ) in strategies:
            n_runs = len( seeds )
            print( Text(
                f"  [{method_name}]  {n_runs} run(s)",
                style = "bright_yellow"
            ) )

            for run_idx, seed in enumerate( seeds ):
                test_R2_rec = np.zeros( n_splits )

                for split_idx, ( train_idx, test_idx ) in enumerate( kf.split( X ) ):
                    # Data splitting
                    X_train, X_test = X[train_idx, :], X[test_idx, :]
                    y_train, y_test = y[train_idx], y[test_idx]

                    # Normalisation
                    X_train, X_test, y_train, y_test = Normalize(
                        X_train = X_train, y_train = y_train,
                        X_test  = X_test,  y_test  = y_test
                    )

                    # Build model and generate initial solution
                    cosmos = Constrained_Sparse_Model_Selection(
                        k       = k,
                        alpha_  = alpha_,
                        lambda_ = lambda_,
                        A = _A, b = _b,
                        group_sparsity = group_sparsity
                    )
                    init_sol = getattr( cosmos, attr_name )( X_train, y_train, seed = seed )

                    # Fit
                    start = time.time()
                    cosmos.fit_model( X = X_train, y = y_train, initial_sol = init_sol )
                    end   = time.time()

                    # Predict and evaluate
                    train_pred = cosmos.predict( X_train )
                    test_pred  = cosmos.predict( X_test  )

                    metric_ = Metric(
                        train_true = y_train, train_pred = train_pred,
                        test_true  = y_test,  test_pred  = test_pred,
                        A = _A, b = _b, beta = cosmos.beta_
                    )

                    info_ = {
                        'Initial_Sol': method_name,
                        'Run':         run_idx + 1,
                        'Seed':        seed,
                        'Sample size': X.shape[0],
                        'Features':    X.shape[1],
                        'k':           k,
                        'alpha':       alpha_,
                        'lambda':      lambda_,
                        'CV':          split_idx + 1,
                        'Time':        end - start,
                        'Iterations':  cosmos.iter_
                    }

                    log_df = pd.concat(
                        [log_df, pd.DataFrame( {**info_, **metric_}, index = [0] )],
                        ignore_index = True
                    )
                    test_R2_rec[split_idx] = metric_['test R2']

                mean_R2 = np.mean( test_R2_rec )
                print( Text(
                    f"    run {run_idx + 1:>3d} / {n_runs}  (seed = {seed})  |  "
                    f"mean test R2 = {mean_R2:.4f}",
                    style = "bright_white"
                ) )

        # ── Save detailed log for this sample size ────────────────────────────
        log_path = f'{log_dir}/{sample_size}_{csv_id}.parquet'
        log_df.to_parquet( log_path, engine = 'pyarrow' )
        print( Text( f"  Log saved → {log_path}", style = "bright_cyan" ) )

        # ── Build summary ─────────────────────────────────────────────────────
        # Step 1: average metric values over the n_splits CV folds within each run
        per_run = (
            log_df
            .groupby( ['Initial_Sol', 'Run', 'Seed'] )
            .agg(
                test_R2      = ( 'test R2',                'mean' ),
                test_R2_std  = ( 'test R2',                'std'  ),   # CV fold std
                train_R2     = ( 'train R2',               'mean' ),
                test_MAPE    = ( 'test MAPE',              'mean' ),
                train_MAPE   = ( 'train MAPE',             'mean' ),
                CS           = ( 'Constraint Satisfaction', 'mean' ),
                Sparsity     = ( 'Sparsity',               'mean' ),
                Time         = ( 'Time',                   'mean' ),
                Iterations   = ( 'Iterations',             'mean' ),
            )
            .reset_index()
        )

        # Step 2: average (and std) across runs within each strategy
        #   For deterministic methods this produces NaN for run_std (only 1 run).
        #   For Random, run_std captures variability across different initialisations.
        summary = (
            per_run
            .groupby( 'Initial_Sol' )
            .agg(
                test_R2         = ( 'test_R2',     'mean' ),
                test_R2_run_std = ( 'test_R2',     'std'  ),   # across-run std
                test_R2_cv_std  = ( 'test_R2_std', 'mean' ),   # avg within-run CV std
                train_R2        = ( 'train_R2',    'mean' ),
                test_MAPE       = ( 'test_MAPE',   'mean' ),
                train_MAPE      = ( 'train_MAPE',  'mean' ),
                CS              = ( 'CS',          'mean' ),
                Sparsity        = ( 'Sparsity',    'mean' ),
                Time            = ( 'Time',        'mean' ),
                Iterations      = ( 'Iterations',  'mean' ),
                Iterations_std  = ( 'Iterations',  'std'  ),   # across-run std
                n_runs          = ( 'test_R2',     'count' ),
            )
            .reset_index()
            .rename( columns = {
                'test_R2':         'test R2',
                'test_R2_run_std': 'test R2 run std',
                'test_R2_cv_std':  'test R2 CV std',
                'train_R2':        'train R2',
                'test_MAPE':       'test MAPE',
                'train_MAPE':      'train MAPE',
                'CS':              'Constraint Satisfaction',
                'Iterations_std':  'Iterations run std',
            } )
        )
        summary.insert( 0, 'Sample size', sample_size )
        summary.insert( 1, 'k',           k           )
        summary.insert( 2, 'alpha',        alpha_      )
        summary.insert( 3, 'lambda',       lambda_     )

        all_summaries = pd.concat( [all_summaries, summary], ignore_index = True )

    # ── Save combined summary ─────────────────────────────────────────────────
    summary_path = f'{out_dir}/Summary_{csv_id}.csv'
    all_summaries.to_csv( summary_path, index = False )
    print()
    print( Text( f"Summary saved → {summary_path}", style = "bright_green" ) )
