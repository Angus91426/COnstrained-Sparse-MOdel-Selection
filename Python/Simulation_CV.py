import warnings
warnings.filterwarnings( "ignore" )

import numpy as np, pandas as pd
import os, time
import optuna

from rich import print
from rich.text import Text
from dataclasses import dataclass
from typing import Callable, Optional
from itertools import product
from sklearn.model_selection import KFold
from Python.global_ import Normalize, Metric
from Python.wat_constants import a1, a2
from Python.models_ import Holistic_Regression, Partial_Least_Squares, Sparse_Partial_Least_Squares, \
                            Constrained_Sparse_Model_Selection, L0Learn_Model

optuna.logging.set_verbosity( optuna.logging.WARNING )

SEED = 11220426

def run_model(
        model: object,
        X_train: np.ndarray, X_test: np.ndarray,
        y_train: np.ndarray, y_test: np.ndarray,
        A: np.ndarray, b: np.ndarray,
        initial_sol: np.ndarray = None
    ):
    # Model training
    start = time.time()
    if initial_sol is not None:
        model.fit_model( X = X_train, y = y_train, initial_sol = initial_sol )
    else:
        model.fit_model( X = X_train, y = y_train )
    end = time.time()

    # Prediction
    train_pred = model.predict( X_train )
    test_pred = model.predict( X_test )

    # Metric computing
    metric_ = Metric( train_true = y_train, train_pred = train_pred, \
                        test_true = y_test, test_pred = test_pred, \
                        A = A, b = b, beta = model.beta_ )

    # Extracting feature coefficient
    coef_ = { f'X{feature_idx + 1}': model.beta_[feature_idx] for feature_idx in range( X_train.shape[1] ) }
    return metric_, coef_, ( end - start )

@dataclass
class ModelSpec:
    """Everything that differs between models; the Optuna search loop itself is shared (see `optuna_search`)."""
    name: str
    space: Callable[[optuna.Trial, dict], dict]           # (trial, ctx) -> suggested raw params
    build: Callable[[dict, dict], object]                 # (params, ctx) -> unfitted model instance
    log_params: Callable[[dict, object], dict]            # (params, fitted_model) -> {'Param1', 'Param2', 'Param3'}
    describe: Optional[Callable[[dict], str]] = None      # (ctx) -> human-readable search-space string
    needs_initial_sol: bool = False
    n_trials: int = 300

def format_params( params: dict ) -> str:
    return ", ".join( f"{name_} = {value_:.4f}" if isinstance( value_, float ) else f"{name_} = {value_}" for name_, value_ in params.items() )

def optuna_search(
        spec: ModelSpec, X: np.ndarray, y: np.ndarray, kf: KFold,
        A: np.ndarray, b: np.ndarray, ctx: dict,
        sim_name: str, num_features: int, multiplier: float, csv_id: str
    ) -> pd.DataFrame:
    """Run one Optuna TPE (+ MedianPruner) search over `spec`'s hyperparameters, log every trial,
    and return the best trial's per-fold results (mirrors the COSMOS search in the original code,
    generalized so every model shares the same driver instead of duplicating the CV/logging loop)."""

    # trial_splits stores each trial's per-fold DataFrame, keyed by trial number.
    # Pruned trials will have a partial DataFrame (< n_splits rows); completed
    # trials will have a full n_splits-row DataFrame.
    trial_splits: dict = {}

    def objective( trial ):
        params = spec.space( trial, ctx )
        model = spec.build( params, ctx )

        split_df = pd.DataFrame()
        test_R2_rec = np.zeros( kf.get_n_splits() )

        for split_idx, ( train_idx, test_idx ) in enumerate( kf.split( X ) ):
            # Data splitting
            X_train, X_test = X[train_idx, :], X[test_idx, :]
            y_train, y_test = y[train_idx], y[test_idx]

            # Data normalization
            X_train, X_test, y_train, y_test = Normalize( X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test )

            initial_sol = model.Initial_Sol( X_train = X_train, y_train = y_train ) if spec.needs_initial_sol else None
            metric_, coef_, time_ = run_model(
                model = model, X_train = X_train, X_test = X_test,
                y_train = y_train, y_test = y_test, A = A, b = b, initial_sol = initial_sol
            )

            info_ = {
                'Model': spec.name,
                **spec.log_params( params, model ),
                'CV': split_idx + 1,
                'Sample size': X.shape[0],
                'Features': X.shape[1],
                'Time': time_
            }

            split_df = pd.concat( [split_df, pd.DataFrame( {**info_, **metric_, **coef_}, index = [0] )], ignore_index = True )
            test_R2_rec[split_idx] = metric_['test R2']

            # Report the running mean R² after each fold so the pruner can
            # abort unpromising trials before all n_splits folds are done.
            trial.report( float( np.mean( test_R2_rec[:split_idx + 1] ) ), step = split_idx )
            if trial.should_prune():
                trial_splits[trial.number] = split_df   # save partial record
                raise optuna.TrialPruned()

        trial_splits[trial.number] = split_df
        return float( np.mean( test_R2_rec ) )

    def callback( study, trial ):
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        if trial.value >= study.best_value - 1e-9:
            print( Text(
                f"[Trial {(trial.number + 1):>4d}] New best {spec.name} | {format_params( trial.params )} | R2 = {trial.value:.4f}",
                style = "bright_magenta"
            ) )
        else:
        # elif (trial.number + 1) % 20 == 0 or (trial.number + 1) == spec.n_trials:
            best = study.best_trial
            print( Text(
                f"[Trial {(trial.number + 1):>4d}] {format_params( trial.params )} | R2 = {trial.value:.4f} | "
                f"Best R2 = {study.best_value:.4f} ({format_params( best.params )})",
                style = "bright_cyan"
            ) )

    # TPESampler: Bayesian adaptive search (same seed as KFold for full reproducibility).
    # MedianPruner: after n_startup_trials warm-up, prune a trial at fold step s if its
    # intermediate R² is below the median of all completed trials at the same step.
    # n_warmup_steps=2 means the first two folds are always run before pruning is considered.
    sampler = optuna.samplers.TPESampler( seed = SEED )
    pruner  = optuna.pruners.MedianPruner( n_startup_trials = 20, n_warmup_steps = 2 )
    study   = optuna.create_study( direction = 'maximize', sampler = sampler, pruner = pruner )

    search_desc = spec.describe( ctx ) if spec.describe else ""
    print( Text(
        f"\n[{spec.name}] Sample: {multiplier * 55} | Optuna TPE search: {spec.n_trials} trials"
        + ( f" | {search_desc}" if search_desc else "" ),
        style = "bright_white"
    ) )

    study.optimize( objective, n_trials = spec.n_trials, callbacks = [callback], show_progress_bar = False )

    # Aggregate all trial records into the log file
    log_df = pd.concat( trial_splits.values(), axis = 0, ignore_index = True ) if trial_splits else pd.DataFrame()
    log_df.to_parquet(
        f'Log/{sim_name}/_{num_features}/{multiplier * 55}_{csv_id}_{spec.name}.parquet',
        engine = 'pyarrow'
    )

    # Report and return the best trial's per-fold results
    best_t = study.best_trial
    n_pruned = len( [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED] )
    print( Text(
        f"\n[{spec.name}] Best result: {format_params( best_t.params )} | R2 = {best_t.value:.4f}  "
        f"(trial #{best_t.number}, {n_pruned} pruned / {spec.n_trials} total)",
        style = "bright_green"
    ) )

    return trial_splits[best_t.number]

def build_specs( ctx: dict, n_trials: int, run_HR: bool, run_Relax_HR: bool, run_PLS: bool, \
                run_SPLS: bool, run_COSMOS: bool, run_L0Learn: bool, \
                run_COSMOS_Unconstrained: bool = False ) -> list:
    """Assemble the list of ModelSpecs to run for the current (multiplier-derived) `ctx`.
    Each model only needs to describe its search space / constructor / logged params;
    the CV loop, Optuna study, logging and best-result bookkeeping are shared."""
    specs = []

    #! Holistic Regression
    if run_HR:
        specs.append( ModelSpec(
            name = "HR",
            space = lambda trial, ctx: {
                'k':     trial.suggest_int(   'k',     6,   ctx['k_max'] ),
                'Gamma': trial.suggest_float( 'Gamma', 0.0, 1.0 )
            },
            build = lambda p, ctx: Holistic_Regression( k = p['k'], Gamma = p['Gamma'], constant_1 = a1(), constant_2 = a2(), time_limit = 3600 ),
            log_params = lambda p, model: { 'Param1': p['k'], 'Param2': p['Gamma'], 'Param3': p['Gamma'] },
            describe = lambda ctx: f"k ∈ [6, {ctx['k_max']}], Gamma ∈ [0, 1]",
            n_trials = n_trials
        ) )

    #! Relaxed Holistic Regression
    if run_Relax_HR:
        specs.append( ModelSpec(
            name = "Relax_HR",
            space = lambda trial, ctx: {
                'k':     trial.suggest_int(   'k',     3,   ctx['k_max'] ),
                'Gamma': trial.suggest_float( 'Gamma', 0.0, 1.0 )
            },
            build = lambda p, ctx: Holistic_Regression( k = p['k'], Gamma = p['Gamma'], constant_1 = a1(), constant_2 = a2(), relax = True, time_limit = 3600 ),
            log_params = lambda p, model: { 'Param1': p['k'], 'Param2': p['Gamma'], 'Param3': p['Gamma'] },
            describe = lambda ctx: f"k ∈ [3, {ctx['k_max']}], Gamma ∈ [0, 1]",
            n_trials = n_trials
        ) )

    #! Partial Least Squares
    if run_PLS:
        specs.append( ModelSpec(
            name = "PLS",
            space = lambda trial, ctx: { 'ncomp': trial.suggest_int( 'ncomp', 1, ctx['max_ncomp'] - 1 ) },
            build = lambda p, ctx: Partial_Least_Squares( ncomp = p['ncomp'] ),
            log_params = lambda p, model: { 'Param1': p['ncomp'], 'Param2': p['ncomp'], 'Param3': p['ncomp'] },
            describe = lambda ctx: f"ncomp ∈ [1, {ctx['max_ncomp'] - 1}]",
            n_trials = n_trials
        ) )

    #! Sparse Partial Least Squares
    if run_SPLS:
        specs.append( ModelSpec(
            name = "SPLS",
            space = lambda trial, ctx: {
                'K':   trial.suggest_int(   'K',   1,   ctx['max_K'] - 1 ),
                'eta': trial.suggest_float( 'eta', 0.1, 0.9 )
            },
            build = lambda p, ctx: Sparse_Partial_Least_Squares( K = p['K'], eta = p['eta'] ),
            log_params = lambda p, model: { 'Param1': p['K'], 'Param2': p['eta'], 'Param3': p['eta'] },
            describe = lambda ctx: f"K ∈ [1, {ctx['max_K'] - 1}], eta ∈ [0.1, 0.9]",
            n_trials = n_trials
        ) )

    #! Constrained Sparse Model Selection (COSMOS)
    if run_COSMOS:
        specs.append( ModelSpec(
            name = "COSMOS",
            space = lambda trial, ctx: {
                # 'k':      trial.suggest_int(   'k',      ctx['l_mandatory'], ctx['k_max'] ),
                'k':      trial.suggest_int(   'k',      ctx['l_mandatory'], 500 ),
                'alpha':  trial.suggest_float( 'alpha',  0.0,  1.0 ),
                'lambda': trial.suggest_float( 'lambda', 1e-8, 1.0, log = True )
            },
            build = lambda p, ctx: Constrained_Sparse_Model_Selection(
                k = p['k'], alpha_ = p['alpha'], lambda_ = p['lambda'],
                A = ctx['A'], b = ctx['b'], group_sparsity = ctx['group_sparsity']
            ),
            log_params = lambda p, model: { 'Param1': p['k'], 'Param2': p['alpha'], 'Param3': p['lambda'] },
            # describe = lambda ctx: f"k ∈ [{ctx['l_mandatory']}, {ctx['k_max']}], alpha ∈ [0, 1], lambda ∈ [1e-8, 1.0] (log-scale)",
            describe = lambda ctx: f"k ∈ [{ctx['l_mandatory']}, 500], alpha ∈ [0, 1], lambda ∈ [1e-8, 1.0] (log-scale)",
            needs_initial_sol = True,
            n_trials = n_trials
        ) )

    #! Unconstrained COSMOS -- the same solver with the linear constraints switched off.
    if run_COSMOS_Unconstrained:
        specs.append( ModelSpec(
            name = "COSMOS_Unconstrained",
            space = lambda trial, ctx: {
                'k':      trial.suggest_int(   'k',      1,    ctx['k_max'] ),
                'alpha':  trial.suggest_float( 'alpha',  0.0,  1.0 ),
                'lambda': trial.suggest_float( 'lambda', 1e-8, 1.0, log = True )
            },
            build = lambda p, ctx: Constrained_Sparse_Model_Selection(
                k = p['k'], alpha_ = p['alpha'], lambda_ = p['lambda'],
                A = np.zeros( ( 0, ctx['p'] ) ), b = np.zeros( 0 ),
                group_sparsity = ctx['group_sparsity']
            ),
            log_params = lambda p, model: { 'Param1': p['k'], 'Param2': p['alpha'], 'Param3': p['lambda'] },
            describe = lambda ctx: f"k ∈ [1, {ctx['k_max']}], alpha ∈ [0, 1], lambda ∈ [1e-8, 1.0] (log-scale), no A*beta >= b",
            needs_initial_sol = True,
            n_trials = n_trials
        ) )

    #! L0Learn (penalties: L0 / L0L1 / L0L2 x algorithms: CD / CDPSI)
    # For "L0", gamma is not a regularization parameter (always 0), so only lambda is tuned.
    # Param1 is logged from the fitted model's actual support size, not the raw hyperparameter.
    if run_L0Learn:
        for ( penalty_, algorithm_ ) in product( ["L0", "L0L1", "L0L2"], ["CD", "CDPSI"] ):
            has_gamma = penalty_ != "L0"

            def make_space( has_gamma ):
                def space( trial, ctx ):
                    params = { 'lambda_val': trial.suggest_float( 'lambda_val', 1e-6, 1.0, log = True ) }
                    if has_gamma:
                        params['gamma_val'] = trial.suggest_float( 'gamma_val', 1e-4, 1.0, log = True )
                    return params
                return space

            def make_build( penalty_, algorithm_ ):
                return lambda p, ctx: L0Learn_Model(
                    penalty = penalty_, algorithm = algorithm_,
                    lambda_val = p['lambda_val'], gamma_val = p.get( 'gamma_val', 0.0 )
                )

            def make_describe( has_gamma ):
                return lambda ctx: "lambda ∈ [1e-6, 1] (log-scale)" + ( ", gamma ∈ [1e-4, 1] (log-scale)" if has_gamma else "" )

            specs.append( ModelSpec(
                name = f"L0Learn_{penalty_}_{algorithm_}",
                space = make_space( has_gamma ),
                build = make_build( penalty_, algorithm_ ),
                log_params = lambda p, model: {
                    'Param1': model.supp_size_, 'Param2': p['lambda_val'], 'Param3': p.get( 'gamma_val', 0.0 )
                },
                describe = make_describe( has_gamma ),
                n_trials = n_trials
            ) )

    return specs

def main( sim_name: str, num_features: int, n_splits: int, multiplier_range: list, \
        run_HR: bool, run_Relax_HR: bool, run_PLS: bool, run_SPLS: bool, run_COSMOS: bool, run_L0Learn: bool,
        run_COSMOS_Unconstrained: bool = False, n_trials: int = 300 ):
    # Use the current time as the CSV file's ID
    csv_id = time.strftime( "%Y%m%d-%H%M%S", time.localtime() )

    #* Create necessary folder
    if not os.path.exists( f'Output/{sim_name}/_{num_features}' ):
        os.makedirs( f'Output/{sim_name}/_{num_features}', exist_ok = True )
    if not os.path.exists( f'Log/{sim_name}/_{num_features}' ):
        os.makedirs( f'Log/{sim_name}/_{num_features}', exist_ok = True )
    if not os.path.exists( f'Figure/{sim_name}/_{num_features}' ):
        os.makedirs( f'Figure/{sim_name}/_{num_features}', exist_ok = True )

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
    b = np.array( [epsilon, epsilon, epsilon, epsilon, epsilon, epsilon, epsilon] )

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

    # Number of mandatory features = columns of A that participate in at least one constraint.
    # COSMOS derives this identically from A internally; we mirror it here for the k search range.
    l_mandatory = int( np.sum( np.any( A != 0, axis = 0 ) ) )  # 6 for WAT

    # Cross-validation
    kf = KFold( n_splits = n_splits, shuffle = True, random_state = SEED )

    opt_df = pd.DataFrame()

    for multiplier in multiplier_range:
        data = pd.read_csv( f'Data/{sim_name}/{num_features}_p/{55 * multiplier}.csv' )
        X = data.iloc[:, 1:].values
        y = data.iloc[:, 0].values

        for train_idx, _ in kf.split( X ):
            n_samples_, n_features_ = X[train_idx, :].shape[0], X[train_idx, :].shape[1]
            break

        # Per-multiplier context shared by every model's search space
        ctx = {
            'A': A, 'b': b, 'group_sparsity': group_sparsity, 'l_mandatory': l_mandatory,
            'p':          n_features_,   # width of the empty A built by COSMOS_Unconstrained
            'k_max':      min( int( n_samples_ / np.log10( n_samples_ ) ), n_features_ ),
            'max_ncomp':  min( n_samples_, n_features_, 15 ),
            'max_K':      min( n_samples_, n_features_, 15 )
        }

        specs = build_specs(
            ctx, n_trials,
            run_HR = run_HR, run_Relax_HR = run_Relax_HR, run_PLS = run_PLS, run_SPLS = run_SPLS,
            run_COSMOS = run_COSMOS, run_L0Learn = run_L0Learn,
            run_COSMOS_Unconstrained = run_COSMOS_Unconstrained
        )

        for spec in specs:
            best_split_df = optuna_search(
                spec, X, y, kf, A, b, ctx,
                sim_name = sim_name, num_features = num_features, multiplier = multiplier, csv_id = csv_id
            )
            opt_df = pd.concat( [opt_df, best_split_df], axis = 0, ignore_index = True )

        # Save files
        opt_df.to_csv( f'Output/{sim_name}/_{num_features}/Optimal_{csv_id}.csv', index = False )
