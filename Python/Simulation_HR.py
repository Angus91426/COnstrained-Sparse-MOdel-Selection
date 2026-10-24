import os, sys, time
import numpy as np, pandas as pd

from sklearn.model_selection import KFold
from Python.global_ import Normalize, Metric
from Python.wat_constants import a1, a2
from Python.models_ import Holistic_Regression

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
    metric_['MIPGap'] = model.MIPGap_
    
    # Extracting feature coefficient
    coef_ = { f'X{feature_idx + 1}': model.beta_[feature_idx] for feature_idx in range( X_train.shape[1] ) }
    return metric_, coef_, ( end - start )

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

# Usage (from the repository root):
#     python -m Python.Simulation_HR 500     # Table 11, p = 530
#     python -m Python.Simulation_HR 5000    # Table 13, p = 5030
# k and Gamma are taken from the COSMOS optimum of the run identified by csv_id.
sim_name = "Irrelevant Nonlinear"
num_features = int( sys.argv[1] ) if len( sys.argv ) > 1 else 5000
csv_id = { 500: "20260915-151925", 5000: "20260915-153244" }[num_features]
n_splits = 5
SEED = 11220426

optimal_df = pd.read_csv( f"Output/{sim_name}/_{num_features}/Optimal_{csv_id}.csv" )
optimal_df = optimal_df[(optimal_df["Model"] == "COSMOS") & (optimal_df["CV"] == 1)]
hyper_param = {
    'k': optimal_df["Param1"].values.tolist(),
    'Gamma': optimal_df["Param3"].values.tolist(),
}

kf = KFold( n_splits = n_splits, shuffle = True, random_state = SEED )
record_df = pd.DataFrame()

for multiplier in range( 1, 7 ):
    data = pd.read_csv( f'Data/{sim_name}/{num_features}_p/{55 * multiplier}.csv' )
    X = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values
    
    k = hyper_param['k'][multiplier - 1]
    Gamma = hyper_param['Gamma'][multiplier - 1]
    
    HR = Holistic_Regression( k = k, Gamma = Gamma, constant_1 = a1(), constant_2 = a2(), time_limit = 3600 )
    print( f'Sample size: {multiplier * 55} | k: {k} | Gamma: {Gamma}' )
    for split_idx, ( train_idx, test_idx ) in enumerate( kf.split( X ) ):
        # Data splitting
        X_train, X_test = X[train_idx, :], X[test_idx, :]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Data normalization
        X_train, X_test, y_train, y_test = Normalize( X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test )
        
        metric_, coef_, time_ = run_model( model = HR, X_train = X_train, X_test = X_test,
                                            y_train = y_train, y_test = y_test, A = A, b = b )
        
        info_ = {
                'Model': 'HR',
                'Param1': k,
                'Param2': Gamma,
                'CV': split_idx + 1,
                'Sample size': X.shape[0],
                'Features': X.shape[1],
                'Time': time_
            }
        record_df = pd.concat( [record_df, pd.DataFrame( {**info_, **metric_, **coef_}, index = [0] )], ignore_index = True )
        record_df.to_csv( f'Output/{sim_name}/_{num_features}/HR_Simulation.csv', index = False )
        print( f'CV {split_idx + 1} completed | Test R²: {metric_["test R2"]:.4f} | Time: {time_:.4f} seconds' )
    print( "=." * 100 )