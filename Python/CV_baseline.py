import warnings
warnings.filterwarnings( "ignore" )

import numpy as np, pandas as pd
import os, time

from rich import print
from rich.text import Text
from itertools import product
from sklearn.model_selection import KFold
from Python.global_ import Normalize, Metric
from Python.wat_constants import a1, a2
from Python.models_ import Holistic_Regression, Partial_Least_Squares, Sparse_Partial_Least_Squares, \
                            Constrained_Sparse_Model_Selection, L0Learn_Model

####################################################################################################
#                                                                                                  #
#                            Simulation with the base data (6 features)                            #
#                                                                                                  #
#   Grid search over the non-sparsity hyperparameters. Every grid lies inside the corresponding    #
#   Optuna search space used for p = 530 / 5,030 (Simulation_CV.py), so both designs search the    #
#   same spaces at different resolutions. k is fixed at 6 for COSMOS and HR because all six WAT    #
#   parameters must be retained.                                                                   #
#                                                                                                  #
####################################################################################################

SEED    = 11220426
K_FIXED = 6

#* Hyperparameter grids (all within the Optuna ranges)
# COSMOS : alpha in [0, 1], lambda in [1e-8, 1] (log)
COSMOS_ALPHA  = np.round( np.linspace( 0.0, 1.0, 21 ), 2 ).tolist()           # 0, 0.05, ..., 1
COSMOS_LAMBDA = np.logspace( -8, 0, 30 ).tolist()                             # 30 values, log
# HR     : Gamma in [0, 1]
HR_GAMMA      = np.round( np.linspace( 0.0, 1.0, 51 ), 2 ).tolist()           # 0, 0.02, ..., 1
# SPLS   : eta in [0.1, 0.9]
SPLS_ETA      = np.round( np.linspace( 0.1, 0.9, 9 ), 1 ).tolist()            # 0.1, 0.2, ..., 0.9
# L0Learn: lambda in [1e-6, 1] (log), gamma in [1e-4, 1] (log)
L0_LAMBDA     = np.logspace( -6, 0, 20 ).tolist()
L0_GAMMA      = np.logspace( -4, 0, 20 ).tolist()
# PLS / SPLS component caps follow Simulation_CV.py: min(n_train, p, 15)
MAX_COMP_CAP  = 15


def run_model(
        model: object,
        X_train: np.ndarray, X_test: np.ndarray,
        y_train: np.ndarray, y_test: np.ndarray,
        A: np.ndarray, b: np.ndarray,
        initial_sol: np.ndarray = None
    ):
    start = time.time()
    if initial_sol is not None:
        model.fit_model( X = X_train, y = y_train, initial_sol = initial_sol )
    else:
        model.fit_model( X = X_train, y = y_train )
    end = time.time()

    train_pred = model.predict( X_train )
    test_pred  = model.predict( X_test )

    metric_ = Metric( train_true = y_train, train_pred = train_pred,
                      test_true  = y_test,  test_pred  = test_pred,
                      A = A, b = b, beta = model.beta_ )

    coef_ = { f'X{i + 1}': model.beta_[i] for i in range( X_train.shape[1] ) }
    return metric_, coef_, ( end - start )


def run_search(
        model_name: str, candidates,
        model_factory, info_fn, message_fn,
        X: np.ndarray, y: np.ndarray, kf: KFold, A: np.ndarray, b: np.ndarray,
        sample_size: int, csv_id: str,
        needs_initial_sol: bool = False,
        per_split_model: bool = False,
    ):
    """
    Exhaustive grid search with k-fold CV: fit every candidate on all folds, keep the
    candidate with the highest mean test R2, and log every candidate to a parquet file.

    model_factory( candidate )         -> a fresh, unfitted model instance
    info_fn( candidate, fitted_model ) -> dict with 'Param1' / 'Param2' / 'Param3'
    message_fn( candidate )            -> human-readable description for the "new best" print

    per_split_model=True re-creates the model for every fold instead of once per candidate.
    needs_initial_sol=True calls model.Initial_Sol(...) before each fold's fit.
    """
    n_splits = kf.get_n_splits()
    opt_     = { 'R2': -np.inf, 'df': None }
    log_df   = pd.DataFrame()

    print( Text( f'\n[{model_name}] Sample: {sample_size} | Grid search: {len( candidates )} candidates', style = "bright_white" ) )

    for candidate in candidates:
        test_R2_rec = np.zeros( n_splits )
        split_df    = pd.DataFrame()

        model = None if per_split_model else model_factory( candidate )

        for split_idx, ( train_idx, test_idx ) in enumerate( kf.split( X ) ):
            X_train, X_test = X[train_idx, :], X[test_idx, :]
            y_train, y_test = y[train_idx], y[test_idx]

            X_train, X_test, y_train, y_test = Normalize( X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test )

            m = model_factory( candidate ) if per_split_model else model
            initial_sol = m.Initial_Sol( X_train = X_train, y_train = y_train ) if needs_initial_sol else None

            metric_, coef_, time_ = run_model( model = m, X_train = X_train, X_test = X_test,
                                               y_train = y_train, y_test = y_test, A = A, b = b,
                                               initial_sol = initial_sol )

            info_ = {
                'Model': model_name, **info_fn( candidate, m ),
                'CV': split_idx + 1, 'Sample size': X.shape[0], 'Features': X.shape[1], 'Time': time_
            }
            split_df = pd.concat( [split_df, pd.DataFrame( {**info_, **metric_, **coef_}, index = [0] )], ignore_index = True )
            test_R2_rec[split_idx] = metric_['test R2']

        log_df = pd.concat( [log_df, split_df], axis = 0, ignore_index = True )

        mean_R2 = np.mean( test_R2_rec )
        if mean_R2 > opt_['R2']:
            print( Text( f'Update optimal result for sample size {sample_size} with model {model_name} and R2 {mean_R2:.4f} ({message_fn( candidate )}).', style = "bright_magenta" ) )
            opt_['R2'] = mean_R2
            opt_['df'] = split_df

    log_df.to_parquet( f'Log/Baseline/_6/{sample_size}_{csv_id}_{model_name}.parquet', engine = 'pyarrow' )
    return opt_['df']


if __name__ == '__main__':
    #* Create necessary folders
    for folder in ( 'Output/Baseline/_6', 'Log/Baseline/_6', 'Figure/Baseline/_6' ):
        os.makedirs( folder, exist_ok = True )

    csv_id = time.strftime( "%Y%m%d-%H%M%S", time.localtime() )

    # Matrix form constraints
    A = np.array([
        [0,  0,     1, 0,  0,  0],
        [0,  0,     0, 0,  0,  1],
        [1, -1,     0, 0,  0,  0],
        [1,  0, -a1(), 0,  0,  0],
        [0,  1, -a2(), 0,  0,  0],
        [0,  0,     0, 1, -1,  0],
        [0,  0,     0, 0,  1, -1]
    ])
    epsilon = 1e-4
    b = np.array( [epsilon] * 7 )

    opt_df   = pd.DataFrame()
    n_splits = 5
    kf = KFold( n_splits = n_splits, shuffle = True, random_state = SEED )

    for multiplier in range( 1, 7 ):
        sample_size = multiplier * 55
        data = pd.read_csv( f'Data/Base Data/{sample_size}/Synthetic.csv' )
        X = data.iloc[:, 1:].values
        y = data.iloc[:, 0].values

        # Training-fold dimensions (used for the PLS / SPLS component caps)
        train_idx, _ = next( iter( kf.split( X ) ) )
        n_samples_, n_features_ = X[train_idx, :].shape
        max_comp = min( n_samples_, n_features_, MAX_COMP_CAP )

        common_kwargs = dict( X = X, y = y, kf = kf, A = A, b = b, sample_size = sample_size, csv_id = csv_id )

        # #! Holistic Regression  (k fixed at 6, Gamma in [0, 1])
        # opt_df = pd.concat( [opt_df, run_search(
        #     model_name    = "HR",
        #     candidates    = HR_GAMMA,
        #     model_factory = lambda gamma: Holistic_Regression( k = K_FIXED, Gamma = gamma, constant_1 = a1(), constant_2 = a2(), time_limit = 3600 ),
        #     info_fn       = lambda gamma, model: { 'Param1': K_FIXED, 'Param2': gamma, 'Param3': gamma },
        #     message_fn    = lambda gamma: f'k={K_FIXED}, gamma={gamma}',
        #     **common_kwargs
        # )], axis = 0, ignore_index = True )

        # #! Partial Least Squares  (ncomp in {1, ..., min(n_train, p, 15)})
        # opt_df = pd.concat( [opt_df, run_search(
        #     model_name    = "PLS",
        #     candidates    = list( range( 1, max_comp ) ),
        #     model_factory = lambda ncomp: Partial_Least_Squares( ncomp = ncomp ),
        #     info_fn       = lambda ncomp, model: { 'Param1': ncomp, 'Param2': ncomp, 'Param3': ncomp },
        #     message_fn    = lambda ncomp: f'ncomp={ncomp}',
        #     **common_kwargs
        # )], axis = 0, ignore_index = True )

        # #! Sparse Partial Least Squares  (K in {1, ..., min(n_train, p, 15) - 1}, eta in [0.1, 0.9])
        # opt_df = pd.concat( [opt_df, run_search(
        #     model_name    = "SPLS",
        #     candidates    = list( product( range( 1, max_comp ), SPLS_ETA ) ),
        #     model_factory = lambda cand: Sparse_Partial_Least_Squares( K = cand[0], eta = cand[1] ),
        #     info_fn       = lambda cand, model: { 'Param1': cand[0], 'Param2': cand[1], 'Param3': cand[1] },
        #     message_fn    = lambda cand: f'K={cand[0]}, eta={cand[1]}',
        #     **common_kwargs
        # )], axis = 0, ignore_index = True )

        #! COSMOS  (k fixed at 6, alpha in [0, 1], lambda in [1e-8, 1] log)
        opt_df = pd.concat( [opt_df, run_search(
            model_name        = "COSMOS",
            candidates        = list( product( COSMOS_ALPHA, COSMOS_LAMBDA ) ),
            model_factory     = lambda cand: Constrained_Sparse_Model_Selection( k = K_FIXED, alpha_ = cand[0], lambda_ = cand[1], A = A, b = b ),
            info_fn           = lambda cand, model: { 'Param1': K_FIXED, 'Param2': cand[0], 'Param3': cand[1] },
            message_fn        = lambda cand: f'k={K_FIXED}, alpha={cand[0]}, lambda={cand[1]:.2e}',
            needs_initial_sol = True,
            **common_kwargs
        )], axis = 0, ignore_index = True )

        # #! L0Learn  (penalties: L0 / L0L1 / L0L2 x algorithms: CD / CDPSI)
        # for ( penalty_, algorithm_ ) in product( [ "L0", "L0L1", "L0L2" ], [ "CD", "CDPSI" ] ):
        #     candidates = [ ( lam, 0.0 ) for lam in L0_LAMBDA ] if penalty_ == "L0" else list( product( L0_LAMBDA, L0_GAMMA ) )

        #     opt_df = pd.concat( [opt_df, run_search(
        #         model_name      = f"L0Learn_{penalty_}_{algorithm_}",
        #         candidates      = candidates,
        #         model_factory   = lambda cand, penalty_ = penalty_, algorithm_ = algorithm_: L0Learn_Model(
        #                                 penalty = penalty_, algorithm = algorithm_, lambda_val = cand[0], gamma_val = cand[1] ),
        #         info_fn         = lambda cand, model: { 'Param1': model.supp_size_, 'Param2': cand[0], 'Param3': cand[1] },
        #         message_fn      = lambda cand: f'lambda={cand[0]:.2e}, gamma={cand[1]:.2e}',
        #         per_split_model = True,
        #         **common_kwargs
        #     )], axis = 0, ignore_index = True )

        # Save after every sample size so partial results survive an interruption
        opt_df.to_csv( f'Output/Baseline/_6/Optimal_{csv_id}.csv', index = False )