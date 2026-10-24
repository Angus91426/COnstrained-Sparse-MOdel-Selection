"""
Synthetic WAT data generator (Section 5.1.1 of the paper).

Base datasets (p = 6): correlated normal draws with the target correlation matrix of
Chien & Chen (2021), rescaled to the observed WAT ranges, with a discretized response,
accepted only if the signs and 1% significance of all correlations and the n-type/p-type
PLS dominance match the reference, and selected by the weighted discrepancy D(w).

High-dimensional datasets: 494 or 4,994 CoV-scaled, permuted distractors per base dataset
(Data/Irrelevant/), followed by the 30 nonlinear transformations of the WAT parameters
(Data/Irrelevant Nonlinear/), giving p = 530 or p = 5030. All random draws are seeded.
"""
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score, f1_score
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool, cpu_count
import os, json, time
from rich import print
from rich.text import Text

plt.rcParams['font.size'] = 10

# Correlation Matrix (from the figure)
desired_corr_matrix = np.array([
    [1.00, 0.74, 0.50, 0.74, -0.079, -0.15, -0.00038],  # Y
    [0.74, 1.00, 0.45, 0.80, -0.38, -0.24, -0.18],  # WAT.1/n
    [0.50, 0.45, 1.00, 0.59, -0.074, 0.088, -0.015],  # WAT.2/n
    [0.74, 0.80, 0.59, 1.00, -0.21, -0.07, 0.013],  # WAT.3/n
    [-0.079, -0.38, -0.074, -0.21, 1.00, 0.40, 0.40],  # WAT.1/p
    [-0.15, -0.24, 0.088, -0.07, 0.40, 1.00, 0.55],  # WAT.2/p
    [-0.00038, -0.18, -0.015, 0.013, 0.40, 0.55, 1.00]   # WAT.3/p
])

desired_p_value = np.array([
    [0,     0.001, 0.001, 0.001, 0.57,  0.29,  1.00],
    [0.001, 0,     0.001, 0.001, 0.001, 0.072, 0.19],
    [0.001, 0.001, 0,     0.001, 0.59,  0.52,  0.92],
    [0.001, 0.001, 0.001, 0,     0.12,  0.61,  0.92],
    [0.57,  0.001, 0.59,  0.12,  0,     0.001, 0.001],
    [0.29,  0.072, 0.52,  0.61,  0.001, 0,     0.001],
    [1.00,  0.19,  0.92,  0.92,  0.001, 0.001, 0]
])

# Variable Ranges (estimated from the figure - ADJUST THESE)
ranges = {
    'Y': (13.4, 14.6),      # Example: Adjust these based on the figure
    'WAT.1/n': (6.0, 7.4),   # Example: Adjust these based on the figure
    'WAT.2/n': (4.3, 5.5),   # Example: Adjust these based on the figure
    'WAT.3/n': (2.6, 3.8),   # Example: Adjust these based on the figure
    'WAT.1/p': (5.7, 6.9),   # Example: Adjust these based on the figure
    'WAT.2/p': (4.2, 5.6),     # Example: Adjust these based on the figure
    'WAT.3/p': (2.6, 3.6)      # Example: Adjust these based on the figure
}

ticks = {
    'Y': np.arange( 13.6, 14.6, 0.2 ),
    'WAT.1/n': np.arange( 6.0, 7.5, 0.2 ),
    'WAT.2/n': np.arange( 4.4, 5.5, 0.2 ),
    'WAT.3/n': np.arange( 2.6, 3.9, 0.2 ),
    'WAT.1/p': np.arange( 5.8, 6.9, 0.2 ),
    'WAT.2/p': np.arange( 4.2, 5.7, 0.2 ),
    'WAT.3/p': np.arange( 2.6, 3.7, 0.2 )
}

WAT_Coef = [0.399, 0.128, 0.272, 0.214, -0.131, 0.075]

# Generate initial synthetic dataset
mean = [0] * len( desired_corr_matrix )

simulations = 100
tuning_sets = 100
rng = np.random.default_rng( seed = 11220426 )
# cpus = round( cpu_count() / 2 )
cpus = cpu_count()

def tune_weights( num_important, important_rate, num_unimportant, unimportant_rate, rand_seeds ):
    weights = np.empty( ( len( rand_seeds ), num_important + num_unimportant ) )
    for seed_id, seed in enumerate( rand_seeds ):
        random.seed( int(seed) )
        #! Create important weights
        # Generate random weights between 0 and 1
        important_weight = [random.random() for _ in range( num_important )]
        # Calculate current sum
        current_sum = sum( important_weight )
        # Normalize to target sum
        important_weight = [w * important_rate / current_sum for w in important_weight]
        
        #! Create unimportant weights
        # Generate random weights between 0 and 1
        unimportant_weight = [random.random() for _ in range( num_unimportant )]
        # Calculate current sum
        current_sum = sum( unimportant_weight )
        # Normalize to target sum
        unimportant_weight = [w * unimportant_rate / current_sum for w in unimportant_weight]
        
        #! Combine weights
        weights[seed_id] = np.concatenate( [important_weight, unimportant_weight] )
    return weights

def _pls_fit( X, y, ncomp ):
    """PLS fit used in the acceptance check (Step 4 of Section 5.1.1): unscaled
    PLSRegression, returning the coefficient vector and the training predictions."""
    pls = PLSRegression( n_components = ncomp, scale = False )
    pls.fit( X, y )
    return np.ravel( pls.coef_ ), np.ravel( pls.predict( X ) )

def Data_Performance( df, weights ):
    #! Correlation coefficient matrix
    df_corr = df.corr().values
    
    # MSE
    mse_corr = mean_squared_error( desired_corr_matrix, df_corr )
    
    # F1
    transfer_df_corr = np.where( df_corr > 0, 1, 0 )
    transfer_desired_corr = np.where( desired_corr_matrix > 0, 1, 0 )
    F1_corr = f1_score( transfer_df_corr, transfer_desired_corr, average = 'micro' )
    
    #! p-value matrix
    df_p = np.zeros( ( 7, 7 ) )
    for i in range( 7 ):
        for j in range( 7 ):
            if i == j:
                df_p[i, j] = 0
            else:
                df_p[i, j] = pearsonr( df.iloc[:, i], df.iloc[:, j] )[1]
    
    # MSE
    mse_p = mean_squared_error( desired_p_value, df_p )
    
    # F1
    transferred_df_p = np.where( df_p < 0.01, 0, 1 )
    transferred_desired_p = np.where( desired_p_value < 0.01, 0, 1 )
    F1_p = f1_score( transferred_df_p, transferred_desired_p, average = 'micro' )
    
    #! PLS result
    X_normalized = StandardScaler().fit_transform( df.drop( columns=["Y"] ).to_numpy() )
    y_normalized = StandardScaler().fit_transform( df["Y"].to_numpy().reshape( -1, 1 ) ).ravel()
    best_PLS = {
        "ncomp": 0,
        "R2": 0,
        "Coef": None
    }
    for ncomp in range( 1, 7 ):
        PLS_Coef, PLS_Pred = _pls_fit( X_normalized, y_normalized, ncomp = ncomp )
        PLS_r2 = r2_score( y_normalized, PLS_Pred )
        
        if PLS_r2 > best_PLS["R2"]:
            best_PLS["ncomp"] = ncomp
            best_PLS["R2"] = PLS_r2
            best_PLS["Coef"] = PLS_Coef
    
    # MSE
    mse_coef = mean_squared_error( WAT_Coef, best_PLS["Coef"] )
    mse_R2 = np.sqrt( ( 0.671 - best_PLS["R2"] )**2 )
    
    # F1
    coef_order = np.zeros( 3 )
    for coef_id in range( 3 ):
        if best_PLS["Coef"][coef_id] > best_PLS["Coef"][coef_id + 3]:
            coef_order[coef_id] = 1
        else:
            coef_order[coef_id] = 0
    F1_coef = f1_score( [1, 1, 1], coef_order, average='micro' )
    
    if F1_corr == 1 and F1_p == 1 and F1_coef == 1:
        weighted_MSE = weights[0] * mse_corr + weights[1] * mse_R2 + weights[2] * mse_coef + weights[3] * mse_p
    else:
        weighted_MSE = np.inf
    return weighted_MSE, mse_corr, mse_R2, mse_coef, mse_p

def Unit_Generation( num_samples, seed ):
    np.random.seed( seed )
    
    # Cholesky decomposition for correlation coefficient matrix
    L = np.linalg.cholesky( desired_corr_matrix )
    uncorrelated_data = np.random.normal( size = ( num_samples, len( desired_corr_matrix ) ) )
    correlated_data = np.dot( L, uncorrelated_data.T ).T
    
    # Scale and shift to match ranges
    df_initial = pd.DataFrame( correlated_data, columns = ranges.keys() )
    for col, ( min_val, max_val ) in ranges.items():
        df_initial[col] = ( df_initial[col] - df_initial[col].min() ) / ( df_initial[col].max() - df_initial[col].min() )
        df_initial[col] = df_initial[col] * ( max_val - min_val ) + min_val
    
    # Make Y Discrete
    sort_df = df_initial.sort_values( by = 'Y' ).reset_index(drop=True)
    y_values = [13.5, 13.75, 13.875, 14.0, 14.25, 14.51]  # Corrected Y values
    y_values_proportion = [0.09, 0.36, 0.02, 0.36, 0.15, 0.02]
    
    for i in range( len( y_values ) ):
        sort_df.loc[round(np.sum( y_values_proportion[:i] ) * num_samples):round(np.sum( y_values_proportion[:i+1] ) * num_samples), 'Y'] = y_values[i]
    
    return sort_df

# ------------------------------------------------------------------------------------------
# What to generate. The repository already contains the base datasets (Data/Base Data), so
# by default only the high-dimensional datasets of Section 5.1.1 are built from them:
#
#     python -m Python.create_data 500 --overwrite     # p = 530  (494 distractors + 30 nonlinear terms)
#     python -m Python.create_data 5000 --overwrite    # p = 5030 (4994 distractors + 30 nonlinear terms)
#
# Note: the high-dimensional datasets used in the paper (Data/Irrelevant Nonlinear/) were
# generated in November 2025 with this procedure, but the random seed used for the distractors
# was not recorded. Running this script reproduces the response, the six WAT parameters and the
# 30 nonlinear terms to numerical precision, and draws a different set of distractors with the
# same design (Section 5.1.1). Use the shipped files to reproduce the paper's numbers.
#
# Set base_data = True to regenerate the base datasets (p = 6) from scratch; this runs the
# full weight search of Steps 1-5 and takes several hours. Delete Data/Base Data/55/
# Best_Result.json first, otherwise the search resumes from the stored optimum.
# The "Random" designs are not used in the paper.
# ------------------------------------------------------------------------------------------
base_data = False
irrelevant_data = True
nonlinear_irrelevant = True
random_data = False
nonlinear_random = False
_args = [a for a in sys.argv[1:] if not a.startswith( '--' )]
num_features = int( _args[0] ) if _args else 5000
# The datasets used in the paper are shipped in Data/Irrelevant Nonlinear/. They are not
# overwritten unless --overwrite is given (see the README on how the two relate).
overwrite = '--overwrite' in sys.argv

if __name__ == '__main__':
    if base_data:
        for multiplier in range( 1, 7 ):
            num_samples = 55 * multiplier  # Match the original dataset size
            if not os.path.exists( f'Data/Base Data/{num_samples}' ):
                os.makedirs( f'Data/Base Data/{num_samples}' )
            
            if multiplier == 1:
                # Dictionary for the optimal dataset.
                if os.path.exists( f'Data/Base Data/{num_samples}/Best_Result.json' ):
                    with open( f'Data/Base Data/{num_samples}/Best_Result.json', 'r' ) as f:
                        opt_ = json.load( f )
                else:
                    opt_ = {
                        'Weights': None,
                        'Weighted_MSE': np.inf,
                        'Corr MSE': None,
                        'R2 MSE': None,
                        'Coef MSE': None,
                        'P MSE': None,
                        'Data': None
                    }
                
                # Generate weights when num_samples = 55
                for important_rate, unimportant_rate in [( 0.5, 0.5 ), ( 0.6, 0.4 ), ( 0.7, 0.3 ), ( 0.8, 0.2 ), ( 0.9, 0.1 )]:
                    weights_seeds = rng.integers( 0, 10000, size = tuning_sets )
                    weights = tune_weights( 2, important_rate, 2, unimportant_rate, weights_seeds )
                    
                    for weight_id, weight in enumerate( weights ):
                        print( Text( f'n = {num_samples} || Important rate: {important_rate} || Unimportant rate: {unimportant_rate} || Generate weights {weight_id + 1}/{len( weights )}', style = 'blue' ) )
                        unit_seeds = rng.integers( 0, 10000, size = simulations )
                        with Pool( cpus ) as pool:
                            results = pool.starmap( Unit_Generation, [( num_samples, seed ) for seed in unit_seeds] )
                        
                        best_result = min( results, key = lambda x: Data_Performance( x, weight )[0] )
                        
                        if Data_Performance( best_result, weight )[0] < opt_['Weighted_MSE']:
                            print( Text( f'n = {num_samples} || Update optimal result with MSE: {Data_Performance( best_result, weight )[0]:.4f}', style = 'green' ) )
                            opt_['Weights'] = weight.tolist()
                            opt_['Weighted_MSE'], opt_['Corr MSE'], opt_['R2 MSE'], \
                                opt_['Coef MSE'], opt_['P MSE'] = Data_Performance( best_result, weight )
                            opt_['Data'] = best_result
                            
                            opt_['Data'].to_csv( f'Data/Base Data/{num_samples}/Synthetic.csv', index = False )
                            generated_dataset = opt_['Data']
                            plt.figure( figsize = ( 20, 8 ) )
                            for i, col1 in enumerate( ranges.keys() ):
                                for j, col2 in enumerate( ranges.keys() ):
                                    plt.subplot( len( ranges ), len( ranges ), i * len( ranges ) + j + 1 )
                                    if i == j:
                                        plt.text( 0.5, 0.5, col1, ha = 'center' )
                                        plt.xticks( [] )
                                        plt.yticks( [] )
                                    elif i > j:
                                        plt.scatter( generated_dataset[col2], generated_dataset[col1], s = 10 )
                                        # Only show the ticks in the first column and the last row
                                        if j != 0:
                                            plt.yticks( [] )
                                        else:
                                            plt.yticks( ticks[col1] )
                                        
                                        if i != len( ranges ) - 1:
                                            plt.xticks( [] )
                                        else:
                                            plt.xticks( ticks[col2] )
                                    else:
                                        corr = generated_dataset[col1].corr( generated_dataset[col2] )
                                        p_val = pearsonr( generated_dataset[col1], generated_dataset[col2] )[1]
                                        plt.text( 0.5, 0.7, f'r={corr:.2f}', ha = 'center', color = 'blue' )
                                        plt.text( 0.5, 0.3, f'p={p_val:.4f}', ha = 'center', color = 'red' )
                                        plt.xticks( [] )
                                        plt.yticks( [] )
                            plt.tight_layout()
                            plt.savefig( f"Data/Base Data/{num_samples}/Synthetic.png", dpi = 1000 )
                            
                            # Save the final result of final as JSON
                            opt_['Data'] = f'Data/Base Data/{num_samples}/Synthetic.csv'
                            with open( f"Data/Base Data/{num_samples}/Best_Result.json", "w" ) as f:
                                json.dump( opt_, f )
                
                # Save the weights of the 55 samples
                weight = opt_['Weights']
            else:
                _55_opt_ = json.load( open( f'Data/Base Data/55/Best_Result.json', 'r' ) )
                opt_ = {
                    'Weights': _55_opt_['Weights'],
                    'Weighted_MSE': np.inf,
                    'Corr MSE': None,
                    'R2 MSE': None,
                    'Coef MSE': None,
                    'P MSE': None,
                    'Data': None
                }
                weight = opt_['Weights']
                
                unit_seeds = rng.integers( 0, 10000, size = simulations )
                with Pool( cpus ) as pool:
                    results = pool.starmap( Unit_Generation, [( num_samples, seed ) for seed in unit_seeds] )
                
                best_result = min( results, key = lambda x: Data_Performance( x, weight )[0] )
                
                if Data_Performance( best_result, weight )[0] < opt_['Weighted_MSE']:
                    print( Text( f'n = {num_samples} || Update optimal result with MSE: {Data_Performance( best_result, weight )[0]:.4f}', style = 'green' ) )
                    opt_['Weighted_MSE'], opt_['Corr MSE'], opt_['R2 MSE'], \
                        opt_['Coef MSE'], opt_['P MSE'] = Data_Performance( best_result, weight )
                    opt_['Data'] = best_result
                    
                    opt_['Data'].to_csv( f'Data/Base Data/{num_samples}/Synthetic.csv', index = False )
                    generated_dataset = opt_['Data']
                    plt.figure( figsize = ( 20, 8 ) )
                    for i, col1 in enumerate( ranges.keys() ):
                        for j, col2 in enumerate( ranges.keys() ):
                            plt.subplot( len( ranges ), len( ranges ), i * len( ranges ) + j + 1 )
                            if i == j:
                                plt.text( 0.5, 0.5, col1, ha = 'center' )
                                plt.xticks( [] )
                                plt.yticks( [] )
                            elif i > j:
                                plt.scatter( generated_dataset[col2], generated_dataset[col1], s = 10 )
                                # Only show the ticks in the first column and the last row
                                if j != 0:
                                    plt.yticks( [] )
                                else:
                                    plt.yticks( ticks[col1] )
                                
                                if i != len( ranges ) - 1:
                                    plt.xticks( [] )
                                else:
                                    plt.xticks( ticks[col2] )
                            else:
                                corr = generated_dataset[col1].corr( generated_dataset[col2] )
                                p_val = pearsonr( generated_dataset[col1], generated_dataset[col2] )[1]
                                plt.text( 0.5, 0.7, f'r={corr:.2f}', ha = 'center', color = 'blue' )
                                plt.text( 0.5, 0.3, f'p={p_val:.4f}', ha = 'center', color = 'red' )
                                plt.xticks( [] )
                                plt.yticks( [] )
                    plt.tight_layout()
                    plt.savefig( f"Data/Base Data/{num_samples}/Synthetic.png", dpi = 1000 )
                
                # Save the final result of final as JSON
                opt_['Data'] = f'Data/Base Data/{num_samples}/Synthetic.csv'
                with open( f"Data/Base Data/{num_samples}/Best_Result.json", "w" ) as f:
                    json.dump( opt_, f )
    
    if irrelevant_data:
        if not os.path.exists( f'Data/Irrelevant/{num_features}_p' ):
            os.makedirs( f'Data/Irrelevant/{num_features}_p' )
        
        # Determine the irrelevant features
        rng = np.random.default_rng( seed = 11220426 )
        p = 6
        
        irrelevant_generate = {
            'noise_loc': [],
            'feature_idx': []
        }
        for _ in range( num_features - p ):
            irrelevant_generate['noise_loc'].append( int( rng.integers( 0, 5 ) ) )
            irrelevant_generate['feature_idx'].append( int( rng.integers( 0, p ) ) )
        
        for multiplier in range( 1, 7 ):
            print( Text( f'Sample size: {multiplier * 55}', style = 'red' ) )
            
            rng = np.random.default_rng( seed = 11220426 )
            data_ = pd.read_csv( f'Data/Base Data/{multiplier * 55}/Synthetic.csv' )
            X = data_.drop( columns = ["Y"] )
            feature_col = X.columns.tolist()
            y = data_["Y"]
            
            # Create irrelevant features
            n, _ = X.shape
            irrelevant_features = np.zeros( ( n, num_features - p ) )
            irrelevant_col = []
            for _ in range( num_features - p ):
                noise_loc = irrelevant_generate['noise_loc'][_]
                feature_idx = irrelevant_generate['feature_idx'][_]
                rand_noise = rng.normal( loc = noise_loc, scale = 1, size = n )
                mean_ = np.mean( X.iloc[:, feature_idx] )
                std_ = np.std( X.iloc[:, feature_idx] )
                mean_factor = 1
                irrelevant_ = ( X.iloc[:, feature_idx] * ( std_ / ( mean_factor * mean_ ) ) + rand_noise ).to_numpy()
                irrelevant_features[:, _] = rng.permuted( irrelevant_ )
                irrelevant_col.append( f'{feature_col[feature_idx]}_loc_{noise_loc}_feature_{_}' )
            
            # Combining irrelevant features into the original data
            irrelevant_df = pd.DataFrame( irrelevant_features, columns = irrelevant_col )
            feature_df = pd.concat( [X, irrelevant_df], axis = 1 )
            data = pd.concat( [y, feature_df], axis = 1 )
            data.to_csv( f'Data/Irrelevant/{num_features}_p/{multiplier * 55}.csv', index = False )
    
    if nonlinear_irrelevant:
        if not os.path.exists( f'Data/Irrelevant Nonlinear/{num_features}_p' ):
            os.makedirs( f'Data/Irrelevant Nonlinear/{num_features}_p' )
        
        for multiplier in range( 1, 7 ):
            print( Text( f'Sample size: {multiplier * 55}', style = 'red' ) )
            
            data_ = pd.read_csv( f'Data/Irrelevant/{num_features}_p/{multiplier * 55}.csv' )
            X = data_.drop( columns = ["Y"] )
            feature_col = X.columns.tolist()
            WAT_col = feature_col[:6]
            irrelevant_col = feature_col[6:]
            y = data_["Y"]
            
            new_col = []
            for col in WAT_col:
                X[col + '_square'] = X[col] ** 2
                new_col.append( col + '_square' )
                
                X[col + '_cube'] = X[col] ** 3
                new_col.append( col + '_cube' )
                
                X[col + '_log'] = np.log( X[col] )
                new_col.append( col + '_log' )
                
                X[col + '_sqrt'] = np.sqrt( X[col] )
                new_col.append( col + '_sqrt' )
                
                X[col + '_sin'] = np.sin( X[col] )
                new_col.append( col + '_sin' )
            
            col_order = WAT_col + new_col + irrelevant_col
            X = X[col_order]
            
            data = pd.concat( [y, X], axis = 1 )
            out_path = f'Data/Irrelevant Nonlinear/{num_features}_p/{multiplier * 55}.csv'
            if os.path.exists( out_path ) and not overwrite:
                print( f'{out_path} exists (dataset used in the paper); skipped. Use --overwrite to replace it.' )
            else:
                data.to_csv( out_path, index = False )
    
    if random_data:
        if not os.path.exists( f'Data/Random/{num_features}_p' ):
            os.makedirs( f'Data/Random/{num_features}_p' )
        
        # Determine the irrelevant features
        rng = np.random.default_rng( seed = 11220426 )
        p = 6
        
        constant_irrelevant = {
            'mean': []
        }
        correlated_irrelevant = {
            'feature_idx': [],
            'correlation': []
        }
        
        for _ in range( num_features - p ):
            if _ % 3 == 1: # Near constant features
                constant_irrelevant['mean'].append( rng.integers( 1, 7 ) )
            elif _ % 3 == 2: # Features correlated to original WAT features
                correlated_irrelevant['feature_idx'].append( int( rng.integers( 0, p ) ) )
                values = np.arange( 0.1, 0.7, 0.1 )
                correlated_irrelevant['correlation'].append( round( float( rng.choice( values ) ), 1 ) )
        
        for multiplier in range( 1, 7 ):
            print( Text( f'Sample size: {multiplier * 55}', style = 'red' ) )
            
            rng = np.random.default_rng( seed = 11220426 )
            data_ = pd.read_csv( f'Data/Base Data/{multiplier * 55}/Synthetic.csv' )
            X = data_.drop( columns = ["Y"] )
            feature_col = X.columns.tolist()
            y = data_["Y"]
            
            # Create irrelevant features
            n, _ = X.shape
            irrelevant_features = np.zeros( ( n, num_features - p ) )
            irrelevant_col = []
            for _ in range( num_features - p ):
                if _ % 3 == 0: # Features from normal distribution
                    irrelevant_features[:, _] = rng.normal( loc = 0, scale = 1, size = n )
                    irrelevant_col.append( f'{_}_Normal' )
                elif _ % 3 == 1: # Near constant features
                    mean_ = constant_irrelevant['mean'][_ // 3]
                    irrelevant_features[:, _] = rng.normal( loc = mean_, scale = 0.01, size = n )
                    irrelevant_col.append( f'{_}_Constant_{mean_}' )
                elif _ % 3 == 2: # Features correlated to original WAT features
                    feature_idx = correlated_irrelevant['feature_idx'][_ // 3]
                    correlation = correlated_irrelevant['correlation'][_ // 3]
                    Spec_WAT = X.iloc[:, feature_idx]
                    mean_ = np.mean( Spec_WAT )
                    std_ = np.std( Spec_WAT )
                    Normalize_Spec_WAT = ( Spec_WAT - mean_ ) / std_
                    noise = rng.normal( loc = 0, scale = 1, size = n )
                    New_normalized = ( correlation * Normalize_Spec_WAT ) + ( np.sqrt( 1 - correlation**2 ) * noise )
                    New_Rescaled = ( New_normalized * std_ ) + mean_
                    irrelevant_features[:, _] = New_Rescaled
                    irrelevant_col.append( f'{_}_Correlated_{feature_idx}_{correlation}' )
            
            irrelevant_df = pd.DataFrame( irrelevant_features, columns = irrelevant_col )
            feature_df = pd.concat( [X, irrelevant_df], axis = 1 )
            data = pd.concat( [y, feature_df], axis = 1 )
            data.to_csv( f'Data/Random/{num_features}_p/{multiplier * 55}.csv', index = False )
    
    if nonlinear_random:
        if not os.path.exists( f'Data/Random Nonlinear/{num_features}_p' ):
            os.makedirs( f'Data/Random Nonlinear/{num_features}_p' )
        
        for multiplier in range( 1, 7 ):
            print( Text( f'Sample size: {multiplier * 55}', style = 'red' ) )
            
            data_ = pd.read_csv( f'Data/Random/{num_features}_p/{multiplier * 55}.csv' )
            X = data_.drop( columns = ["Y"] )
            feature_col = X.columns.tolist()
            WAT_col = feature_col[:6]
            irrelevant_col = feature_col[6:]
            y = data_["Y"]
            
            new_col = []
            for col in WAT_col:
                X[col + '_square'] = X[col] ** 2
                new_col.append( col + '_square' )
                
                X[col + '_cube'] = X[col] ** 3
                new_col.append( col + '_cube' )
                
                X[col + '_log'] = np.log( X[col] )
                new_col.append( col + '_log' )
                
                X[col + '_sqrt'] = np.sqrt( X[col] )
                new_col.append( col + '_sqrt' )
                
                X[col + '_sin'] = np.sin( X[col] )
                new_col.append( col + '_sin' )
            
            col_order = WAT_col + new_col + irrelevant_col
            X = X[col_order]
            
            data = pd.concat( [y, X], axis = 1 )
            data.to_csv( f'Data/Random Nonlinear/{num_features}_p/{multiplier * 55}.csv', index = False )