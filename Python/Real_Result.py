"""
Post-processing of the real-data experiments (Python/real_CV.py).

Same job as Simulation_Result.py -- turn one `Optimal_<csv_id>.csv` per run into a per-fold
Summary sheet and a per-model Final sheet -- but for the two real applications instead of the
synthetic WAT designs.  Differences, and why:

  1. Paths.  real_CV.py writes to Output/Real/<dataset>/<response>/ and
     Log/Real/<dataset>/<response>/, and its log files are named `<csv_id>_<model>.parquet`
     (no `<sample size>_` prefix -- a real dataset has exactly one sample size).

  2. Metadata width.  The real result frame carries two extra leading columns, 'Dataset' and
     'Response', so the bookkeeping block is the first 16 columns, not 14.

  3. Feature grouping.  There is no WAT / Nonlinear / Irrelevant partition on real data.  The
     literature-derived constraint file Data/<dataset>/Constraints/<response>.csv plays that
     role instead: its non-zero columns are the *constrained* predictors (the wavelengths /
     gene probes the domain literature says must enter the model), everything else is
     unconstrained.  So each fold gets

         Constrained    -- how many constrained predictors the model actually selected
         Unconstrained  -- how many of the remaining p - l predictors it selected

     plus one 0/1 indicator column per constrained predictor, which the Final sheet sums over
     the folds -- the direct analogue of the per-nonlinear-term columns in Simulation_Result.py.
     'Constrained' counts selection only; whether the *sign* the constraint asks for was
     honoured is already measured by 'Constraint Satisfaction'.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt, seaborn as sns
import os

from itertools import product

# Set font size for matplotlib
plt.rcParams.update({"font.size": 7})

dict_ = {
    'NIR_Corn_moisture': {
        'csv_id': '20260922-163316',
        'dataset': 'NIR Corn',
        'response': 'moisture',
        'compute_average': True,
        'compute_std': True,
        'visualize': False,
        'title': 'NIR Corn / Moisture'
    },

    'NIR_Corn_oil': {
        'csv_id': '20260922-163316',
        'dataset': 'NIR Corn',
        'response': 'oil',
        'compute_average': True,
        'compute_std': True,
        'visualize': False,
        'title': 'NIR Corn / Oil'
    },

    'NIR_Corn_protein': {
        'csv_id': '20260922-163316',
        'dataset': 'NIR Corn',
        'response': 'protein',
        'compute_average': True,
        'compute_std': True,
        'visualize': False,
        'title': 'NIR Corn / Protein'
    },

    'NIR_Corn_starch': {
        'csv_id': '20260922-163316',
        'dataset': 'NIR Corn',
        'response': 'starch',
        'compute_average': True,
        'compute_std': True,
        'visualize': False,
        'title': 'NIR Corn / Starch'
    },

    'Riboflavin': {
        'csv_id': '20260922-183533',
        'dataset': 'Riboflavin',
        'response': 'riboflavin',
        'compute_average': True,
        'compute_std': True,
        'visualize': False,
        'title': 'Riboflavin'
    },
}

# Number of leading bookkeeping / metric columns written by real_CV.optuna_search():
# Dataset, Response, Model, Param1-3, CV, Sample size, Features, Time, train/test MAPE,
# train/test R2, Constraint Satisfaction, Sparsity.  Everything after them is a coefficient.
N_META_COLS = 16

def load_constraint_features( dataset: str, response: str ) -> tuple[list, list]:
    """Return (constrained, mandatory) predictor names for one (dataset, response) pair.

    `constrained` are the predictors touched by any row of A -- i.e. the ones the literature
    constrains in either direction.  `mandatory` are those touched by a row with b_i > 0, the
    subset a feasible model is *forced* to keep in its support (a row with b_i <= 0, such as
    riboflavin's MCA upper bounds -beta_j >= -u_j, is satisfied by beta_j = 0 and therefore
    mandates nothing).  This mirrors real_CV.count_mandatory().
    """
    path = f'Data/{dataset}/Constraints/{response}.csv'
    constraint_df = pd.read_csv( path )

    if constraint_df.columns[-1] != 'b':
        raise ValueError( f"{path}: expected the last column to be 'b', got {constraint_df.columns[-1]!r}" )

    A = constraint_df.iloc[:, :-1]
    b = constraint_df['b'].to_numpy( dtype = float )

    constrained = A.columns[( A != 0 ).any( axis = 0 )].tolist()
    mandatory   = A.columns[( ( A != 0 ) & ( b[:, np.newaxis] > 0 ) ).any( axis = 0 )].tolist()
    return constrained, mandatory

for name, value in dict_.items():
    model_order = [
        "COSMOS",
        "COSMOS_Relax",
        "COSMOS_Unconstrained",
        "PLS",
        "SPLS",
        "HR",
        "L0Learn_L0_CD",
        "L0Learn_L0_CDPSI",
        "L0Learn_L0L1_CD",
        "L0Learn_L0L1_CDPSI",
        "L0Learn_L0L2_CD",
        "L0Learn_L0L2_CDPSI",
    ]

    csv_save_path = 'Results/Tables_regenerated'   # keeps the published tables in Results/Tables intact

    for path in [csv_save_path]:
        if not os.path.exists( path ):
            os.makedirs( path, exist_ok = True )

    csv_path = f'Output/Real/{value["dataset"]}/{value["response"]}/Optimal_{value["csv_id"]}.csv'

    optimal_df = pd.read_csv( csv_path )
    summary_df = optimal_df.iloc[:, :N_META_COLS].copy()

    # Predictor (coefficient) columns = everything past the bookkeeping block.
    feature_cols = optimal_df.columns[N_META_COLS:].tolist()

    #* The constraint file replaces the WAT / Nonlinear / Irrelevant split of the simulations.
    constrained_cols, mandatory_cols = load_constraint_features( value["dataset"], value["response"] )

    missing = [c for c in constrained_cols if c not in feature_cols]
    if missing:
        raise ValueError(
            f'{csv_path}: constraint column(s) {missing[:5]} are absent from the result frame; '
            f'the constraint file and this run are out of sync.'
        )

    unconstrained_cols = [c for c in feature_cols if c not in set( constrained_cols )]

    selected = ( optimal_df[feature_cols] != 0 )
    summary_df['Constrained']   = selected[constrained_cols].sum( axis = 1 )
    summary_df['Mandatory']     = selected[mandatory_cols].sum( axis = 1 )
    summary_df['Unconstrained'] = selected[unconstrained_cols].sum( axis = 1 )

    # One 0/1 indicator per constrained predictor, named after the wavelength / gene probe.
    constrained_col_names = [f'beta_{c}' for c in constrained_cols]
    Constrained_Coef = selected[constrained_cols].astype( int ).copy()
    Constrained_Coef.columns = constrained_col_names
    summary_df = pd.concat( [summary_df, Constrained_Coef], axis = 1 )

    summary_df['Constraint Satisfaction'] = ( summary_df['Constraint Satisfaction'] * 100 ).round( 2 )
    summary_df.to_excel( f'{csv_save_path}/Real_{value["dataset"]}_{value["response"]}_Summary.xlsx', index = False )

    df_ = summary_df.drop( columns = ['Dataset', 'Response', 'Features', 'Param1', 'Param2', 'Param3', 'CV'] )
    # Compute average and std grouped by Model and Sample size
    final_df = df_.groupby(['Model', 'Sample size']).agg({
        'Time': 'mean',
        'test R2': ['mean', 'std'],
        'Constraint Satisfaction': 'mean',
        'Sparsity': 'mean',
        **{col: 'mean' for col in ['Constrained', 'Mandatory', 'Unconstrained']},
        **{col: 'sum' for col in constrained_col_names}
    }).reset_index()

    # Flatten column names
    final_df.columns = ['Model', 'Sample size', 'Time', 'test R2', 'test R2 std', 'Constraint Satisfaction', 'Sparsity'] + \
                    ['Constrained', 'Mandatory', 'Unconstrained'] + \
                    constrained_col_names

    for (model_, sample_) in product( final_df['Model'].unique(), final_df['Sample size'].unique() ):
        log_path = f'Log/Real/{value["dataset"]}/{value["response"]}/{value["csv_id"]}_{model_}.parquet'
        # The per-trial logs are not part of the repository; the CV Time of the original runs is
        # kept in the tables under Results/Tables. When the logs are absent, CV Time is left empty.
        if os.path.exists( log_path ):
            CV_time = pd.read_parquet( log_path )['Time'].sum()
        else:
            CV_time = np.nan
        final_df.loc[(final_df['Model'] == model_) & (final_df['Sample size'] == sample_), 'CV Time'] = CV_time

    final_df['Model'] = pd.Categorical( final_df['Model'], categories = model_order, ordered = True )
    final_df = final_df.sort_values( by = ['Sample size', 'Model'] ).reset_index( drop = True )
    final_df[['Time', 'CV Time', 'test R2', 'test R2 std']] \
        = final_df[['Time', 'CV Time', 'test R2', 'test R2 std']].round( 4 )
    final_df[['Constraint Satisfaction']] \
        = final_df[['Constraint Satisfaction']].round()
    final_df[['Sparsity']] \
        = final_df[['Sparsity']].round( 1 )

    # Keep the dataset / response identity on the aggregated sheet -- it is constant within a
    # run, so it is re-attached here rather than carried through the groupby.
    final_df.insert( 0, 'Response', value["response"] )
    final_df.insert( 0, 'Dataset', value["dataset"] )
    final_df.to_excel( f'{csv_save_path}/Real_{value["dataset"]}_{value["response"]}_Final.xlsx', index = False )

    print( f'Finished processing {value["dataset"]} / {value["response"]} '
        f'({len( feature_cols )} features, {len( constrained_cols )} constrained, '
        f'{len( mandatory_cols )} mandatory).' )
