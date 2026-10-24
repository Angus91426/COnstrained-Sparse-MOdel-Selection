"""
Real-data cross-validation driver (NIR corn + riboflavin).

Same experiment as Python/simulation_CV.py -- 5-fold CV wrapped in an Optuna TPE search,
one study per model -- but run on the two real-world applications instead of the synthetic
WAT designs:

  * NIR Corn   : Data/NIR Corn/corn_m5spec.csv      (n = 80, p = 700 wavelength channels,
                 instrument m5) against one of the four responses in corn_propvals.csv.
  * Riboflavin : Data/Riboflavin/riboflavin_x.csv   (n = 71, p = 4088 gene probes)
                 against the log production rate in riboflavin_y.csv.

Differences from simulation_CV.py, and why:

  1. (A, b) is read from disk instead of being hard-coded.  simulation_CV.py inlines the
     6-feature WAT matrix; here every response has its own literature-derived constraint set,
     written by Python/NIR_Corn_constraints.py and Python/Riboflavin_constraints.py into
     Data/<dataset>/Constraints/<response>.csv.  Each such file holds p constraint columns
     (named exactly like the predictor columns) plus a trailing 'b' column, so row i is the
     constraint  a_i' beta >= b_i.

  2. A local Metric_real() replaces global_.Metric(), which hard-codes beta[:6] and therefore
     only works for the 6-column WAT A.  Here A already has p columns, so the constraint is
     evaluated on the full beta.  (Same fix as Backup/simulation_Engel.py::Metric_Engel.)

  3. Coefficients are logged under their real names (wavelength in nm / gene probe id) rather
     than X1...Xp, so the selected support can be read off the output directly.

Everything else -- the model search spaces, the Optuna study/pruner configuration and the
per-trial logging -- is imported from simulation_CV.py (ModelSpec / build_specs), so the
simulated and real experiments stay directly comparable.

Usage (from the repository root, mirroring 302_Main.py):

    from Python.Real_CV import main
    main( dataset = 'NIR Corn', responses = ['moisture', 'oil', 'protein', 'starch'],
          run_PLS = True, run_SPLS = True, run_COSMOS = True, n_trials = 300 )

    main( dataset = 'Riboflavin', responses = ['riboflavin'],
          run_PLS = True, run_SPLS = True, run_COSMOS = True, n_trials = 300,
          log_coef = False )
"""

import warnings
warnings.filterwarnings( "ignore" )

import numpy as np, pandas as pd
import os, time
import optuna

from rich import print
from rich.text import Text
from dataclasses import dataclass
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from Python.global_ import Normalize
from Python.Simulation_CV import SEED, ModelSpec, build_specs, format_params

optuna.logging.set_verbosity( optuna.logging.WARNING )

####################################################################################################
#                                                                                                  #
#                                       Dataset registry                                           #
#                                                                                                  #
####################################################################################################

@dataclass
class RealDataset:
    """One real-world dataset.

    `responses` maps a response key -> the column holding that response in `y_path`.
    The response key doubles as the stem of its constraint file, i.e.
    `{constraint_dir}/{response}.csv`, so a response is always paired with exactly one
    constraint set and adding a new one is a single entry here plus the matching CSV.
    """
    name: str
    X_path: str
    y_path: str
    constraint_dir: str
    responses: dict
    id_column: str = 'sample'

DATASETS = {
    #! NIR corn -- instrument m5 (swap to corn_mp5spec.csv / corn_mp6spec.csv for the other two).
    'NIR Corn': RealDataset(
        name           = 'NIR Corn',
        X_path         = 'Data/NIR Corn/corn_m5spec.csv',
        y_path         = 'Data/NIR Corn/corn_propvals.csv',
        constraint_dir = 'Data/NIR Corn/Constraints',
        responses      = {
            'moisture': 'Moisture',   # N1, N2
            'oil':      'Oil',        # N3, N4, N5 + optional N6
            'protein':  'Protein',    # N7, N8
            'starch':   'Starch',     # optional N9
        },
    ),

    #! Riboflavin production in B. subtilis.
    'Riboflavin': RealDataset(
        name           = 'Riboflavin',
        X_path         = 'Data/Riboflavin/riboflavin_x.csv',
        y_path         = 'Data/Riboflavin/riboflavin_y.csv',
        constraint_dir = 'Data/Riboflavin/Constraints',
        responses      = {
            'riboflavin': 'y',   # R1-R4 + optional MCA upper bounds R5-R8
        },
    ),
}

####################################################################################################
#                                                                                                  #
#                                   Data / constraint loading                                      #
#                                                                                                  #
####################################################################################################

def load_constraints( path: str, feature_names: list )-> tuple[np.ndarray, np.ndarray]:
    """Read one `Data/<dataset>/Constraints/<response>.csv` into the matrix form (A, b).

    The file's last column is 'b'; the remaining columns are named after the predictors.
    They are re-indexed onto `feature_names` so A's column order always matches X's column
    order regardless of how the file was written, and any predictor the file omits simply
    gets an all-zero (unconstrained) column.
    """
    constraint_df = pd.read_csv( path )

    if constraint_df.columns[-1] != 'b':
        raise ValueError( f"{path}: expected the last column to be 'b', got {constraint_df.columns[-1]!r}" )

    constraint_cols = constraint_df.columns[:-1].tolist()
    known = set( feature_names )
    unknown = [c for c in constraint_cols if c not in known]
    if unknown:
        raise ValueError(
            f"{path}: {len( unknown )} constraint column(s) are not predictors of X "
            f"(first few: {unknown[:5]}). The constraint file and the design matrix are out of sync."
        )

    A = constraint_df.reindex( columns = feature_names, fill_value = 0.0 ).to_numpy( dtype = float )
    b = constraint_df['b'].to_numpy( dtype = float )
    return A, b

def count_mandatory( A: np.ndarray, b: np.ndarray ) -> int:
    """Number of *mandatory* features, i.e. those touched by at least one row with b_i > 0.

    A row with b_i <= 0 (a sign constraint, or riboflavin's upper bounds -beta_j >= -u_j) is
    satisfied by beta_j = 0 and therefore forces nothing into the support.  This mirrors
    Constrained_Sparse_Model_Selection's own C_mandatory derivation exactly, so the lower end
    of COSMOS's k range is always attainable.  (simulation_CV.py uses the looser
    `np.any( A != 0, axis = 0 )`, which coincides for WAT because every WAT b_i is +epsilon.)
    """
    return int( np.sum( np.any( ( A != 0 ) & ( b[:, np.newaxis] > 0 ), axis = 0 ) ) )

# Column names produced by optuna_search() for bookkeeping/metrics. A predictor sharing one of
# these names would be silently overwritten in the results frame, so we refuse to run instead.
RESERVED_COLUMNS = {
    'Dataset', 'Response', 'Model', 'Param1', 'Param2', 'Param3', 'CV',
    'Sample size', 'Features', 'Time',
    'train MAPE', 'test MAPE', 'train R2', 'test R2', 'Constraint Satisfaction', 'Sparsity',
}

def load_real_dataset( dataset: str, response: str ):
    """Load X, y and the (A, b) pair for one (dataset, response) pair.

    X and y are aligned on the dataset's id column (`sample`) rather than on row order, and
    the id column itself is dropped from X -- it is a label, not a predictor.
    """
    if dataset not in DATASETS:
        raise KeyError( f"Unknown dataset {dataset!r}. Available: {sorted( DATASETS )}" )
    spec = DATASETS[dataset]

    if response not in spec.responses:
        raise KeyError( f"Unknown response {response!r} for {dataset!r}. Available: {sorted( spec.responses )}" )
    y_column = spec.responses[response]

    X_df = pd.read_csv( spec.X_path )
    y_df = pd.read_csv( spec.y_path )

    for path_, df_ in ( ( spec.X_path, X_df ), ( spec.y_path, y_df ) ):
        if spec.id_column not in df_.columns:
            raise ValueError( f"{path_}: missing the id column {spec.id_column!r}" )

    # Align responses to the spectra/expression rows by sample id.
    y_aligned = y_df.set_index( spec.id_column ).reindex( X_df[spec.id_column] )
    if y_aligned[y_column].isna().any():
        missing = X_df[spec.id_column][y_aligned[y_column].isna().to_numpy()].tolist()
        raise ValueError( f"{spec.y_path}: no {y_column!r} value for sample(s) {missing[:5]}" )

    feature_names = [c for c in X_df.columns if c != spec.id_column]
    clashing = sorted( set( feature_names ) & RESERVED_COLUMNS )
    if clashing:
        raise ValueError(
            f"{spec.X_path}: predictor name(s) {clashing} collide with the result columns "
            f"written by optuna_search(); rename them before running."
        )

    X = X_df[feature_names].to_numpy( dtype = float )
    y = y_aligned[y_column].to_numpy( dtype = float )

    A, b = load_constraints( os.path.join( spec.constraint_dir, f'{response}.csv' ), feature_names )
    return X, y, A, b, feature_names

####################################################################################################
#                                                                                                  #
#                                    Metric / single model run                                     #
#                                                                                                  #
####################################################################################################

def Metric_real( train_true: np.ndarray, train_pred: np.ndarray, \
                test_true: np.ndarray, test_pred: np.ndarray, \
                A: np.ndarray, b: np.ndarray, beta: np.ndarray ) -> dict:
    """Identical to global_.Metric, except the constraints are checked on the full beta.

    global_.Metric evaluates `A @ beta[:6]`, which is only correct for the 6-feature WAT
    design; here A already spans all p predictors.
    """
    #! Predictive performance
    train_r2   = r2_score( train_true, train_pred )
    train_mape = mean_absolute_percentage_error( train_true, train_pred )
    test_r2    = r2_score( test_true, test_pred )
    test_mape  = mean_absolute_percentage_error( test_true, test_pred )

    #! Explainability -- percentage of constraints satisfied
    percentage = np.count_nonzero( np.round( A @ np.asarray( beta ).ravel(), 5 ) >= b ) / A.shape[0]

    #! Interpretability -- support size
    nonzero = np.count_nonzero( np.round( beta, 6 ) )

    return {
        'train MAPE': train_mape,
        'test MAPE': test_mape,
        'train R2': train_r2,
        'test R2': test_r2,
        'Constraint Satisfaction': percentage,
        'Sparsity': nonzero
    }

def run_model(
        model: object,
        X_train: np.ndarray, X_test: np.ndarray,
        y_train: np.ndarray, y_test: np.ndarray,
        A: np.ndarray, b: np.ndarray,
        feature_names: list,
        initial_sol: np.ndarray = None
    ):
    """Fit one model on one fold and return (metrics, coefficients, fit time).

    Mirrors simulation_CV.run_model; the coefficient dict is keyed by the real predictor name
    (wavelength / gene probe) instead of X1...Xp.
    """
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
    metric_ = Metric_real( train_true = y_train, train_pred = train_pred, \
                            test_true = y_test, test_pred = test_pred, \
                            A = A, b = b, beta = model.beta_ )

    # Extracting feature coefficient
    beta_ = np.asarray( model.beta_ ).ravel()
    coef_ = { name_: beta_[feature_idx] for feature_idx, name_ in enumerate( feature_names ) }
    return metric_, coef_, ( end - start )

####################################################################################################
#                                                                                                  #
#                                       Optuna CV search                                           #
#                                                                                                  #
####################################################################################################

def optuna_search(
        spec: ModelSpec, X: np.ndarray, y: np.ndarray, kf: KFold,
        A: np.ndarray, b: np.ndarray, ctx: dict, feature_names: list,
        dataset: str, response: str, log_dir: str, csv_id: str, log_coef: bool = True
    ) -> pd.DataFrame:
    """Run one Optuna TPE (+ MedianPruner) search over `spec`'s hyperparameters, log every trial,
    and return the best trial's per-fold results.

    Same driver as simulation_CV.optuna_search; only the labels and the log path are
    real-data specific.  Per-fold rows are accumulated in a list and turned into a DataFrame
    once per trial rather than pd.concat'ed row by row -- with p = 4088 riboflavin probes the
    incremental concat dominates the runtime."""

    # trial_splits stores each trial's per-fold DataFrame, keyed by trial number.
    # Pruned trials will have a partial DataFrame (< n_splits rows); completed
    # trials will have a full n_splits-row DataFrame.
    trial_splits: dict = {}

    def objective( trial ):
        params = spec.space( trial, ctx )
        model = spec.build( params, ctx )

        split_rows = []
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
                y_train = y_train, y_test = y_test, A = A, b = b,
                feature_names = feature_names, initial_sol = initial_sol
            )

            info_ = {
                'Dataset': dataset,
                'Response': response,
                'Model': spec.name,
                **spec.log_params( params, model ),
                'CV': split_idx + 1,
                'Sample size': X.shape[0],
                'Features': X.shape[1],
                'Time': time_
            }

            split_rows.append( { **info_, **metric_, **coef_ } )
            test_R2_rec[split_idx] = metric_['test R2']

            # Report the running mean R² after each fold so the pruner can
            # abort unpromising trials before all n_splits folds are done.
            trial.report( float( np.mean( test_R2_rec[:split_idx + 1] ) ), step = split_idx )
            if trial.should_prune():
                trial_splits[trial.number] = pd.DataFrame( split_rows )   # save partial record
                raise optuna.TrialPruned()

        trial_splits[trial.number] = pd.DataFrame( split_rows )
        return float( np.mean( test_R2_rec ) )

    def callback( study, trial ):
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        if trial.value >= study.best_value - 1e-9:
            print( Text(
                f"[Trial {(trial.number + 1):>4d}] New best {spec.name} | {format_params( trial.params )} | R2 = {trial.value:.4f}",
                style = "bright_magenta"
            ) )
        # else:
        elif (trial.number + 1) % 20 == 0 or (trial.number + 1) == spec.n_trials:
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
        f"\n[{spec.name}] {dataset} / {response} | n = {X.shape[0]}, p = {X.shape[1]} | "
        f"Optuna TPE search: {spec.n_trials} trials"
        + ( f" | {search_desc}" if search_desc else "" ),
        style = "bright_white"
    ) )

    study.optimize( objective, n_trials = spec.n_trials, callbacks = [callback], show_progress_bar = False )

    # Aggregate all trial records into the log file
    log_df = pd.concat( trial_splits.values(), axis = 0, ignore_index = True ) if trial_splits else pd.DataFrame()
    if not log_coef and not log_df.empty:
        # p = 4088 coefficients x n_trials x n_splits makes the riboflavin logs very large;
        # the per-fold coefficients of the best trial are still kept in the Output CSV.
        log_df = log_df.drop( columns = feature_names, errors = 'ignore' )
    log_df.to_parquet( f'{log_dir}/{csv_id}_{spec.name}.parquet', engine = 'pyarrow' )

    # Report and return the best trial's per-fold results
    best_t = study.best_trial
    n_pruned = len( [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED] )
    print( Text(
        f"\n[{spec.name}] Best result: {format_params( best_t.params )} | R2 = {best_t.value:.4f}  "
        f"(trial #{best_t.number}, {n_pruned} pruned / {spec.n_trials} total)",
        style = "bright_green"
    ) )

    return trial_splits[best_t.number]

####################################################################################################
#                                                                                                  #
#                                            Main                                                  #
#                                                                                                  #
####################################################################################################

def main( dataset: str, responses: list = None, n_splits: int = 5, \
        run_HR: bool = False, run_Relax_HR: bool = False, run_PLS: bool = True, \
        run_SPLS: bool = True, run_COSMOS: bool = True, run_L0Learn: bool = False, \
        run_COSMOS_Unconstrained: bool = False, \
        n_trials: int = 300, log_coef: bool = True ):
    """Run the CV + Optuna experiment on one real dataset, once per response.

    Parameters
    ----------
    dataset : str
        Key into DATASETS -- 'NIR Corn' or 'Riboflavin'.
    responses : list, optional
        Response keys to run. Defaults to every response registered for `dataset`.
        Each key selects both the y column and the constraint file of the same name.
    n_splits : int
        Folds of the (shuffled, seeded) KFold, matching simulation_CV.py.
    run_* : bool
        Which model families to fit. HR / Relax_HR default to False here: they are MIQPs over
        all p features, which is not tractable at p = 700 / 4088 within the per-fit time limit.
        L0Learn defaults to False because it needs R + rpy2; turn it on when that is available.
    run_COSMOS_Unconstrained : bool
        Also fit COSMOS with the linear constraints switched off, as a separate
        'COSMOS_Unconstrained' model. Its metrics are still scored against this response's real
        (A, b), so the output shows directly what the constraints cost in R2 and buy back in
        constraint satisfaction. Turn it on together with run_COSMOS to get the pair.
    n_trials : int
        Optuna trials per model family.
    log_coef : bool
        Keep the per-trial coefficient columns in the parquet logs. Set False for riboflavin
        (p = 4088) to keep the log files small; the best trial's coefficients are always
        written to the Output CSV either way.
    """
    if dataset not in DATASETS:
        raise KeyError( f"Unknown dataset {dataset!r}. Available: {sorted( DATASETS )}" )
    spec_ = DATASETS[dataset]
    responses = list( spec_.responses ) if responses is None else list( responses )

    # Use the current time as the CSV file's ID
    csv_id = time.strftime( "%Y%m%d-%H%M%S", time.localtime() )

    for response in responses:
        #* Load the real data and its literature-derived constraints
        X, y, A, b, feature_names = load_real_dataset( dataset, response )

        #* Create necessary folder
        out_dir = f'Output/Real/{dataset}/{response}'
        log_dir = f'Log/Real/{dataset}/{response}'
        fig_dir = f'Figure/Real/{dataset}/{response}'
        for folder in ( out_dir, log_dir, fig_dir ):
            os.makedirs( folder, exist_ok = True )

        # Number of mandatory features = predictors forced into the support by a b_i > 0 row.
        # COSMOS derives this identically from (A, b) internally; we mirror it for the k range.
        l_mandatory = count_mandatory( A, b )

        # Real data carries no dummy-variable groups, unlike the Nonlinear simulations.
        group_sparsity = None

        # Cross-validation
        kf = KFold( n_splits = n_splits, shuffle = True, random_state = SEED )

        for train_idx, _ in kf.split( X ):
            n_samples_, n_features_ = X[train_idx, :].shape
            break

        # Per-response context shared by every model's search space
        ctx = {
            'A': A, 'b': b, 'group_sparsity': group_sparsity, 'l_mandatory': l_mandatory,
            'p':          n_features_,   # width of the empty A built by COSMOS_Unconstrained
            'k_max':      min( int( n_samples_ / np.log10( n_samples_ ) ), n_features_ ),
            'max_ncomp':  min( n_samples_, n_features_, 15 ),
            'max_K':      min( n_samples_, n_features_, 15 )
        }

        if ctx['k_max'] < l_mandatory:
            raise ValueError(
                f"{dataset} / {response}: k_max = {ctx['k_max']} is below the {l_mandatory} mandatory "
                f"feature(s) implied by the constraints, so COSMOS has no feasible support size."
            )

        print( Text(
            f"\n{'=' * 96}\n"
            f"[{dataset} / {response}] n = {X.shape[0]}, p = {X.shape[1]} | "
            f"constraints: {A.shape[0]} row(s), {l_mandatory} mandatory feature(s) | "
            f"{n_splits}-fold CV, k_max = {ctx['k_max']}\n"
            f"{'=' * 96}",
            style = "bold bright_yellow"
        ) )

        specs = build_specs(
            ctx, n_trials,
            run_HR = run_HR, run_Relax_HR = run_Relax_HR, run_PLS = run_PLS, run_SPLS = run_SPLS,
            run_COSMOS = run_COSMOS, run_L0Learn = run_L0Learn,
            run_COSMOS_Unconstrained = run_COSMOS_Unconstrained
        )

        opt_df = pd.DataFrame()
        for spec in specs:
            best_split_df = optuna_search(
                spec, X, y, kf, A, b, ctx, feature_names,
                dataset = dataset, response = response,
                log_dir = log_dir, csv_id = csv_id, log_coef = log_coef
            )
            opt_df = pd.concat( [opt_df, best_split_df], axis = 0, ignore_index = True )

            # Save files (rewritten after every model so a long run can be inspected early)
            opt_df.to_csv( f'{out_dir}/Optimal_{csv_id}.csv', index = False )
