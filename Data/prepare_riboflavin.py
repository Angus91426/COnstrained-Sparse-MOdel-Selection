"""
Export the riboflavin data of Buhlmann, Kalisch & Meier (2014) into the CSV files used by
the code. From the repository root, run:

    python Data/prepare_riboflavin.py

The data were distributed with the R package hdi, which was archived on CRAN on 2026-07-10.
The script therefore does not install hdi: it downloads the archived source package
hdi_0.1-9.tar.gz from the CRAN archive, extracts the riboflavin data file, and reads it with
R through rpy2 (already required for L0Learn). If you have a copy of hdi_0.1-9.tar.gz, pass
its path instead of downloading:  python Data/prepare_riboflavin.py path/to/hdi_0.1-9.tar.gz

Writes Data/Riboflavin/riboflavin_x.csv (71 x 4088 log expression levels) and
Data/Riboflavin/riboflavin_y.csv (log riboflavin production rate), both keyed by `sample`.
Check the result with:  python Data/check_real_data.py
"""
import os
import subprocess
import sys

# On Windows with a non-UTF-8 system code page (e.g. cp950 for Traditional Chinese), rpy2
# fails while reading R's settings, because R reports them in UTF-8. Re-run this script in
# Python's UTF-8 mode in that case.
if os.name == 'nt' and not sys.flags.utf8_mode:
    sys.exit( subprocess.call( [sys.executable, '-X', 'utf8'] + sys.argv ) )

import tarfile
import tempfile
import urllib.request

import numpy as np
import pandas as pd

HDI_URL = 'https://cran.r-project.org/src/contrib/Archive/hdi/hdi_0.1-9.tar.gz'
OUT = os.path.join( 'Data', 'Riboflavin' )


def _prepare_r_on_windows():
    """On Windows, point rpy2 to R and put R's DLL folder on PATH (as Python/models_.py does)."""
    if os.name != 'nt':
        return
    if not os.environ.get( 'R_HOME' ):
        try:
            import winreg
            for hive in ( winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER ):
                try:
                    with winreg.OpenKey( hive, r'SOFTWARE\R-core\R' ) as key:
                        os.environ['R_HOME'] = winreg.QueryValueEx( key, 'InstallPath' )[0]
                        break
                except OSError:
                    continue
        except ImportError:
            pass
    r_bin = os.path.join( os.environ.get( 'R_HOME', '' ), 'bin', 'x64' )
    if os.path.isdir( r_bin ):
        os.environ['PATH'] = r_bin + os.pathsep + os.environ.get( 'PATH', '' )


def _extract_data_file( tarball: str, workdir: str ) -> str:
    with tarfile.open( tarball, 'r:gz' ) as tar:
        members = [ m for m in tar.getmembers()
                    if os.path.basename( m.name ).lower().split( '.' )[0] == 'riboflavin'
                    and '/data/' in m.name ]
        if not members:
            raise FileNotFoundError( f'No riboflavin data file found in {tarball}' )
        member = members[0]
        member.name = os.path.basename( member.name )          # extract flat, without folders
        tar.extract( member, workdir )
        return os.path.join( workdir, member.name )


def main():
    with tempfile.TemporaryDirectory() as tmp:
        if len( sys.argv ) > 1:
            tarball = sys.argv[1]
        else:
            tarball = os.path.join( tmp, 'hdi_0.1-9.tar.gz' )
            print( f'Downloading {HDI_URL} ...' )
            urllib.request.urlretrieve( HDI_URL, tarball )
        data_file = _extract_data_file( tarball, tmp )

        _prepare_r_on_windows()
        import rpy2.robjects as ro

        ro.r( f'load("{data_file.replace( os.sep, "/" )}")' )
        n_rows = int( ro.r( 'nrow(riboflavin$x)' )[0] )
        n_cols = int( ro.r( 'ncol(riboflavin$x)' )[0] )
        # R stores matrices column by column, hence order = 'F'.
        x = np.array( ro.r( 'as.vector(unclass(riboflavin$x))' ), dtype = float ).reshape( ( n_rows, n_cols ), order = 'F' )
        genes = [ str( g ) for g in ro.r( 'colnames(riboflavin$x)' ) ]
        samples = [ str( s ) for s in ro.r( 'rownames(riboflavin)' ) ]
        y = np.array( ro.r( 'as.numeric(riboflavin$y)' ), dtype = float )

    os.makedirs( OUT, exist_ok = True )
    X_df = pd.DataFrame( x, columns = genes )
    X_df.insert( 0, 'sample', samples )
    y_df = pd.DataFrame( { 'sample': samples, 'y': y } )
    X_df.to_csv( os.path.join( OUT, 'riboflavin_x.csv' ), index = False )
    y_df.to_csv( os.path.join( OUT, 'riboflavin_y.csv' ), index = False )
    print( f'Wrote riboflavin_x.csv {X_df.shape} and riboflavin_y.csv {y_df.shape} to {OUT}' )


if __name__ == '__main__':
    main()
