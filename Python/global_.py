import numpy as np, pandas as pd

from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split

np.set_printoptions( suppress = True )

#* Function for data preprocessing
def Data_Preprocessing( X: pd.DataFrame, \
                        drop_first: bool = True, drop_single: bool = True, drop_perfect: bool = True ) -> ( pd.DataFrame, list ):
    
    # 1. Split into numerical and categorical features
    numerical_features = X.select_dtypes( exclude = ['object'] )
    num_col = numerical_features.columns
    categorical_features = X.select_dtypes( include = ['object'] )
    
    # 2. One-hot encoding for categorical features
    if not categorical_features.empty:
        encoded_categorical_df = pd.get_dummies( categorical_features, dtype = int, drop_first = drop_first )
        
        # 3. Remove encoded columns with only one '1'
        if drop_single:
            unique_cols = encoded_categorical_df.columns[encoded_categorical_df.sum( axis = 0 ) > 1]
            encoded_categorical_df = encoded_categorical_df[unique_cols]
    else:
        encoded_categorical_df = pd.DataFrame( columns = [] )
    
    # 4. Concat numerical and categorical features
    X = pd.concat( [numerical_features, encoded_categorical_df], axis = 1 )
    if drop_single:
        unique_cols = X.columns[X.sum( axis = 0 ) > 1]
        X = X[unique_cols]
    
    # 5. Drop constant features
    X = X.loc[:, ( X != X.iloc[0] ).any()]
    
    # 6. Implement Drop_Perfect for perfect collinearity
    if drop_perfect:
        X = Drop_Perfect( X )
    
    # 7. Add intercept term
    X = pd.concat( [pd.DataFrame( np.ones( ( X.shape[0], 1 ) ), columns = ['Intercept'] ), X], axis = 1 )
    
    # 8. Obtain column names for categorical features
    categorical_feature_names = [_ for _ in X.columns if _ not in num_col and _ != 'Intercept']
    
    # 9. Obtain the feature index for the group sparsity
    group_sparsity = []
    for cat_ in categorical_features.columns.tolist():
        temp_ = []
        for col_ in categorical_feature_names:
            if col_.startswith( cat_ ):
                temp_.append( X.columns.get_loc( col_ ) )
        
        if len( temp_ ) < 2: continue
        group_sparsity.append( temp_ )
    
    return X, categorical_feature_names, group_sparsity

#* Function for dropping perfect collinearity
def Drop_Perfect( data: pd.DataFrame ) -> pd.DataFrame:
    """
    Drop one of the columns when their correlation coefficient is 1 or -1.
    
    Parameters
    ----------
    data : np.array
        Feature matrix.
    
    Returns
    ----------
    data : np.array
        Clean data.
    """
    
    # Compute the correlation coefficient matrix
    corr_coef = data.corr().round( 4 )
    
    # Receive the upper triangle elements of the correlation coefficient matrix (not including the diagonal)
    corr_coef = corr_coef.where( np.triu( np.ones_like( corr_coef ), 1 ) == 1 ).to_numpy()
    
    # Receive the index where the correlation coefficient is 1 or -1
    drop_index = np.where( np.abs( corr_coef ) == 1 )
    
    data = data.drop( columns = data.columns[drop_index[0]], axis = 1 )
    
    return data

#* Function for data splitting
def Split_Normalize( X: np.ndarray, y: np.ndarray, valid_size: float = None, seed: int = 901208 ):
    X_train_valid, X_test, y_train_valid, y_test = train_test_split( X, y, test_size = 0.1, random_state = seed )
    X_train_valid, X_test, y_train_valid, y_test = Normalize( X_train_valid, y_train_valid, X_test, y_test )
    if valid_size is None:
        return X_train_valid, X_test, y_train_valid, y_test
    else:
        X_train, X_valid, y_train, y_valid = train_test_split( X_train_valid, y_train_valid, test_size = valid_size, random_state = seed )
        return X_train, X_valid, X_test, y_train, y_valid, y_test

#* Function for data normalization and centering
def Normalize( X_train: np.ndarray, y_train: np.ndarray, \
                X_test: np.ndarray, y_test: np.ndarray, \
                X_valid: np.ndarray = None, y_valid: np.ndarray = None ):
    X_centered = X_train - np.mean( X_train, axis = 0 )
    col_norms  = np.linalg.norm( X_centered, axis = 0 )
    col_norms[ col_norms == 0 ] = 1   # constant column: numerator is also 0, result stays 0
    normalized_train = X_centered / col_norms
    if X_valid is not None:
        normalized_valid = ( X_valid - np.mean( X_train, axis = 0 ) ) / col_norms
    normalized_test = ( X_test - np.mean( X_train, axis = 0 ) ) / col_norms
    
    # Compute the mean for centering data
    y_mean = np.mean( y_train )
    y_train = y_train - y_mean
    if y_valid is not None:
        y_valid = y_valid - y_mean
    y_test = y_test - y_mean
    
    if X_valid is not None:
        return normalized_train, normalized_valid, normalized_test, y_train, y_valid, y_test
    else:
        return normalized_train, normalized_test, y_train, y_test

#* Function for computing performance indicators
def Metric( train_true: np.ndarray, train_pred: np.ndarray, \
            test_true: np.ndarray, test_pred: np.ndarray, \
            A: np.ndarray, b: np.ndarray, beta: np.ndarray, \
            valid_true: np.ndarray = None, valid_pred: np.ndarray = None ) -> dict:
    #! Predictive performance
    train_r2 = r2_score( train_true, train_pred )
    train_mse = mean_squared_error( train_true, train_pred )
    train_mape = mean_absolute_percentage_error( train_true, train_pred )
    test_r2 = r2_score( test_true, test_pred )
    test_mse = mean_squared_error( test_true, test_pred )
    test_mape = mean_absolute_percentage_error( test_true, test_pred )
    if valid_true is not None:
        valid_r2 = r2_score( valid_true, valid_pred )
        valid_mse = mean_squared_error( valid_true, valid_pred )
        valid_mape = mean_absolute_percentage_error( valid_true, valid_pred )
    
    #! Explainability
    # Compute the percentage of constraints satisfied
    percentage = np.count_nonzero( np.round( np.dot( A, beta[:6] ), 5 ) >= b ) / A.shape[0]
    
    #! Interpretability
    nonzero = np.count_nonzero( np.round( beta, 6 ) )
    
    if valid_true is not None:
        metric = {
            # 'train MSE': train_mse,
            # 'valid MSE': valid_mse,
            # 'test MSE': test_mse,
            'train MAPE': train_mape,
            'valid MAPE': valid_mape,
            'test MAPE': test_mape,
            'train R2': train_r2,
            'valid R2': valid_r2,
            'test R2': test_r2,
            'Constraint Satisfaction': percentage,
            'Sparsity': nonzero
        }
    else:
        metric = {
            # 'train MSE': train_mse,
            # 'test MSE': test_mse,
            'train MAPE': train_mape,
            'test MAPE': test_mape,
            'train R2': train_r2,
            'test R2': test_r2,
            'Constraint Satisfaction': percentage,
            'Sparsity': nonzero
        }
    return metric

