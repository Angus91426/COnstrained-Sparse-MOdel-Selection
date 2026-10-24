"""
Check that the prepared real-world data match the files used in the paper.

From the repository root, after running Data/prepare_corn.py and Data/prepare_riboflavin.py:

    python Data/check_real_data.py

The check compares the shape, the column names and order, the sample ids and order (the
cross-validation folds depend on the row order), and the column sums of every predictor and
response against Data/real_data_fingerprint.json, with a relative tolerance of 1e-9 so that
harmless differences in how numbers are written to text do not matter.
"""
import argparse, hashlib, json, os, sys
import numpy as np
import pandas as pd

FILES = {
    'corn_m5spec':   os.path.join( 'Data', 'NIR Corn', 'corn_m5spec.csv' ),
    'corn_propvals': os.path.join( 'Data', 'NIR Corn', 'corn_propvals.csv' ),
    'riboflavin_x':  os.path.join( 'Data', 'Riboflavin', 'riboflavin_x.csv' ),
    'riboflavin_y':  os.path.join( 'Data', 'Riboflavin', 'riboflavin_y.csv' ),
}
FINGERPRINT = os.path.join( 'Data', 'real_data_fingerprint.json' )
ID_COLUMN = 'sample'


def _sha( items ) -> str:
    return hashlib.sha256( '\n'.join( map( str, items ) ).encode( 'utf-8' ) ).hexdigest()


def fingerprint( path: str ) -> dict:
    df = pd.read_csv( path )
    values = df.drop( columns = [ID_COLUMN] ).to_numpy( dtype = float )
    return {
        'shape': list( df.shape ),
        'columns_sha256': _sha( df.columns ),
        'sample_ids_sha256': _sha( df[ID_COLUMN] ),
        'column_sums': values.sum( axis = 0 ).tolist(),
    }


def main():
    parser = argparse.ArgumentParser( description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter )
    parser.add_argument( '--record', action = 'store_true', help = 'write the fingerprint file from the current data (maintainers only)' )
    parser.add_argument( '--root', default = '.', help = 'directory that contains Data/ (default: repository root)' )
    args = parser.parse_args()

    if args.record:
        record = { name: fingerprint( os.path.join( args.root, path ) ) for name, path in FILES.items() }
        with open( FINGERPRINT, 'w' ) as f:
            json.dump( record, f, indent = 1 )
        print( f'Wrote {FINGERPRINT}' )
        return

    with open( FINGERPRINT ) as f:
        expected = json.load( f )

    ok = True
    for name, path in FILES.items():
        full = os.path.join( args.root, path )
        if not os.path.exists( full ):
            print( f'MISSING  {path}  (run the preparation script first)' )
            ok = False
            continue
        got, exp = fingerprint( full ), expected[name]
        problems = []
        if got['shape'] != exp['shape']:
            problems.append( f"shape {got['shape']} != {exp['shape']}" )
        if got['columns_sha256'] != exp['columns_sha256']:
            problems.append( 'column names or order differ' )
        if got['sample_ids_sha256'] != exp['sample_ids_sha256']:
            problems.append( 'sample ids or row order differ' )
        if not problems and not np.allclose( got['column_sums'], exp['column_sums'], rtol = 1e-9, atol = 1e-12 ):
            problems.append( 'values differ' )
        print( f"{'OK      ' if not problems else 'MISMATCH'} {path}" + ( '' if not problems else ': ' + '; '.join( problems ) ) )
        ok &= not problems

    print( 'All real-world data match the paper.' if ok else 'Some files do not match; results may differ from the paper.' )
    sys.exit( 0 if ok else 1 )


if __name__ == '__main__':
    main()
