import pandas as pd
import numpy as np
import matplotlib.pyplot as plt, seaborn as sns
import os

from itertools import product

# Set font size for matplotlib
plt.rcParams.update({"font.size": 7})

dict_ = {
    'baseline': {
        'csv_id': '20260927-220803',
        'sim_name': 'Baseline',
        'num_of_features': 6,
        'compute_average': True,
        'compute_std': True,
        'visualize': False,
        'title': 'Baseline'
    },
    
    # '500_Irrelevant':{
    #     'csv_id': '20260915-134126',
    #     'sim_name': 'Irrelevant',
    #     'num_of_features': 500,
    #     'compute_average': True,
    #     'compute_std': True,
    #     'visualize': False,
    #     'title': 'CoV'
    # },
    
    '500_Irrelevant_Nonlinear':{
        'csv_id': '20260915-151925',
        'sim_name': 'Irrelevant Nonlinear',
        'num_of_features': 500,
        'compute_average': True,
        'compute_std': True,
        'visualize': False,
        'title': 'CoV + Nonlinear'
    },
    
    # '5000_Irrelevant':{
    #     'csv_id': '20260915-141536',
    #     'sim_name': 'Irrelevant',
    #     'num_of_features': 5000,
    #     'compute_average': True,
    #     'compute_std': True,
    #     'visualize': False,
    #     'title': 'CoV'
    # },
    
    '5000_Irrelevant_Nonlinear':{
        'csv_id': '20260915-153244',
        'sim_name': 'Irrelevant Nonlinear',
        'num_of_features': 5000,
        'compute_average': True,
        'compute_std': True,
        'visualize': False,
        'title': 'CoV + Nonlinear'
    },
    
    # '500_Random':{
    #     'csv_id': '20260915-184630',
    #     'sim_name': 'Random',
    #     'num_of_features': 500,
    #     'compute_average': True,
    #     'compute_std': True,
    #     'visualize': False,
    #     'title': 'Random'
    # },
    
    # '500_Random_Nonlinear':{
    #     'csv_id': '20260915-214406',
    #     'sim_name': 'Random Nonlinear',
    #     'num_of_features': 500,
    #     'compute_average': True,
    #     'compute_std': True,
    #     'visualize': False,
    #     'title': 'Random + Nonlinear'
    # },
    
    # '5000_Random':{
    #     'csv_id': '20260915-191054',
    #     'sim_name': 'Random',
    #     'num_of_features': 5000,
    #     'compute_average': True,
    #     'compute_std': True,
    #     'visualize': False,
    #     'title': 'Random'
    # },
    
    # '5000_Random_Nonlinear':{
    #     'csv_id': '20260915-215559',
    #     'sim_name': 'Random Nonlinear',
    #     'num_of_features': 5000,
    #     'compute_average': True,
    #     'compute_std': True,
    #     'visualize': False,
    #     'title': 'Random + Nonlinear'
    # },
}

for name, value in dict_.items():
    model_order = [
        "COSMOS",
        "COSMOS_Relax",
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
    
    csv_path = f'Output/{value["sim_name"]}/_{value["num_of_features"]}/Optimal_{value["csv_id"]}.csv'
    
    optimal_df = pd.read_csv( csv_path )
    summary_df = optimal_df.iloc[:, :14].copy()
    nonlinear_col_names = [
        '(WAT.1/n)^2', '(WAT.1/n)^3', 'log(WAT.1/n)', 'sqrt(WAT.1/n)', 'sin(WAT.1/n)',
        '(WAT.2/n)^2', '(WAT.2/n)^3', 'log(WAT.2/n)', 'sqrt(WAT.2/n)', 'sin(WAT.2/n)',
        '(WAT.3/n)^2', '(WAT.3/n)^3', 'log(WAT.3/n)', 'sqrt(WAT.3/n)', 'sin(WAT.3/n)',
        '(WAT.1/p)^2', '(WAT.1/p)^3', 'log(WAT.1/p)', 'sqrt(WAT.1/p)', 'sin(WAT.1/p)',
        '(WAT.2/p)^2', '(WAT.2/p)^3', 'log(WAT.2/p)', 'sqrt(WAT.2/p)', 'sin(WAT.2/p)',
        '(WAT.3/p)^2', '(WAT.3/p)^3', 'log(WAT.3/p)', 'sqrt(WAT.3/p)', 'sin(WAT.3/p)'
    ]
    
    if value["sim_name"] != '_baseline':
        if "Nonlinear" in value['sim_name']:
            # WAT: X1 to X6 (first 6 features)
            wat_cols = [f'X{i}' for i in range( 1, 7 )]
            summary_df['WAT'] = ( optimal_df[wat_cols] != 0 ).sum( axis = 1 )
            
            # Nonlinear: X7 to X36 (30 features)
            nonlinear_cols = [f'X{i}' for i in range( 7, 37 )]
            summary_df['Nonlinear'] = ( optimal_df[nonlinear_cols] != 0 ).sum( axis = 1 )
            
            # Irrelevant: remaining features
            irrelevant_cols = [f'X{i}' for i in range( 37, value["num_of_features"] + 31 )]
            summary_df['Irrelevant'] = ( optimal_df[irrelevant_cols] != 0 ).sum( axis = 1 )
            
            Nonlinear_Coef = optimal_df[nonlinear_cols].copy()
            Nonlinear_Coef[Nonlinear_Coef != 0] = 1
            Nonlinear_Coef.columns = nonlinear_col_names
            summary_df = pd.concat( [summary_df, Nonlinear_Coef], axis = 1 )
        else:
            # WAT: X1 to X6 (first 6 features)
            wat_cols = [f'X{i}' for i in range( 1, 7 )]
            summary_df['WAT'] = ( optimal_df[wat_cols] != 0 ).sum( axis = 1 )
            
            summary_df['Nonlinear'] = [0]*len( optimal_df )
            
            # Irrelevant: remaining features
            irrelevant_cols = [f'X{i}' for i in range( 7, value["num_of_features"] + 1 )]
            summary_df['Irrelevant'] = ( optimal_df[irrelevant_cols] != 0 ).sum( axis = 1 )
    
    summary_df['Constraint Satisfaction'] = ( summary_df['Constraint Satisfaction'] * 100 ).round( 2 )
    summary_df.to_excel( f'{csv_save_path}/{value["sim_name"]}_{value["num_of_features"]}_Summary.xlsx', index = False )
    
    df_ = summary_df.drop( columns = ['Param1', 'Param2', 'Param3', 'CV'] )
    # Compute average and std grouped by Model and Sample size
    final_df = df_.groupby(['Model', 'Sample size']).agg({
        'Time': 'mean',
        'test R2': ['mean', 'std'],
        'Constraint Satisfaction': 'mean',
        'Sparsity': 'mean',
        **({col: 'mean' for col in ['WAT', 'Nonlinear', 'Irrelevant']} if value["sim_name"] != '_baseline' else {}),
        **({col: 'sum' for col in nonlinear_col_names} if value["sim_name"] != '_baseline' and "Nonlinear" in value['sim_name'] else {})
    }).reset_index()
    
    # Flatten column names
    final_df.columns = ['Model', 'Sample size', 'Time', 'test R2', 'test R2 std', 'Constraint Satisfaction', 'Sparsity'] + \
                    (['WAT', 'Nonlinear', 'Irrelevant'] if value["sim_name"] != '_baseline' else []) + \
                    (nonlinear_col_names if value["sim_name"] != '_baseline' and "Nonlinear" in value['sim_name'] else [])
    
    for (model_, sample_) in product( final_df['Model'].unique(), final_df['Sample size'].unique() ):
        log_path = f'Log/{value["sim_name"]}/_{value["num_of_features"]}/{sample_}_{value["csv_id"]}_{model_}.parquet'
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
    final_df.to_excel( f'{csv_save_path}/{value["sim_name"]}_{value["num_of_features"]}_Final.xlsx', index = False )
    
    print( f'Finished processing {value["sim_name"]} with {value["num_of_features"]} features.' )