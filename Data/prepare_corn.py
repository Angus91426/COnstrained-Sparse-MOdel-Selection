"""
Convert the corn benchmark of Eigenvector Research into the CSV files used by the code.

1. Download corn.mat from
   https://eigenvector.com/resources/data-sets/nir-of-corn-samples-for-standardization-benchmarking/
   and save it as Data/NIR Corn/corn.mat.
2. From the repository root, run:  python Data/prepare_corn.py

Writes Data/NIR Corn/corn_m5spec.csv (80 x 700 spectra of the m5 instrument, 1100-2498 nm)
and Data/NIR Corn/corn_propvals.csv (moisture, oil, protein, starch), both keyed by `sample`.
"""
import os
import numpy as np
import pandas as pd
import scipy.io as sio

SRC = os.path.join( 'Data', 'NIR Corn', 'corn.mat' )
OUT = os.path.join( 'Data', 'NIR Corn' )

mat = sio.loadmat( SRC )
spec = mat['m5spec'][0, 0]          # Eigenvector DataSet object
prop = mat['propvals'][0, 0]

X = np.asarray( spec['data'], dtype = float )                       # 80 x 700
wavelengths = np.asarray( spec['axisscale'][1, 0] ).ravel().astype( int )
Y = np.asarray( prop['data'], dtype = float )                       # 80 x 4
names = [ str( s ).strip() for s in prop['label'][1, 0] ]           # Moisture, Oil, Protein, Starch

samples = np.arange( 1, X.shape[0] + 1 )
X_df = pd.DataFrame( X, columns = [ str( w ) for w in wavelengths ] )
X_df.insert( 0, 'sample', samples )
Y_df = pd.DataFrame( Y, columns = names )
Y_df.insert( 0, 'sample', samples )

X_df.to_csv( os.path.join( OUT, 'corn_m5spec.csv' ), index = False )
Y_df.to_csv( os.path.join( OUT, 'corn_propvals.csv' ), index = False )
print( f"Wrote corn_m5spec.csv {X_df.shape} and corn_propvals.csv {Y_df.shape} to {OUT}" )
