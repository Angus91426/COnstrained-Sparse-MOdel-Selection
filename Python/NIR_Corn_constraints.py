"""
Literature-supported coefficient constraints for the COSMOS NIR-corn application.

Builds the matrix form  A*beta >= b  (with A = [A_C  0], as in Python/simulation_CV.py's
WAT example) for each of the four corn responses -- moisture, oil, protein, starch.
One COSMOS model is fitted per response (see Docs/Word/COSMOS_constraints_corn_riboflavin.docx,
Sections 2.2-2.3), so each response gets its own (A, b) pair.

Predictors are the 700 NIR wavelength channels in Data/NIR Corn/corn_m5spec.csv
(1100-2498 nm, 2 nm grid); the leading "sample" column is the sample ID, not a feature,
so X has p = 700 columns and column j of X corresponds to feature index j here.

Constraint IDs follow the docx (N1-N9): every row is beta_j >= epsilon (or -beta_j >= epsilon
for N2, which has the opposite sign), so A is a signed one-hot row per constrained wavelength
and b is epsilon (or -epsilon collected the same way, i.e. b stays +epsilon since the sign
lives in A) at every row.
"""

import numpy as np, pandas as pd

epsilon = 1e-4

# ------------------------------------------------------------------
# Load just the header of the corn spectra to recover the wavelength -> column-index map.
# Column 0 ("sample") is the sample ID and is not a predictor, so it is dropped before indexing.
# ------------------------------------------------------------------
CORN_SPEC_PATH = 'Data/NIR Corn/corn_m5spec.csv'
header = pd.read_csv( CORN_SPEC_PATH, nrows = 0 ).columns.tolist()
assert header[0] == 'sample', f"Expected first column to be 'sample', got {header[0]!r}"
wavelengths = header[1:]                      # p = 700 predictor names, in column order
p = len( wavelengths )                        # 700
wl_index = { wl: j for j, wl in enumerate( wavelengths ) }   # '1940' -> 420, etc.

def _col( wavelength_nm: int ) -> int:
    """Column index (0-based, within the p predictors) of a wavelength on the 2 nm grid."""
    key = str( wavelength_nm )
    if key not in wl_index:
        raise KeyError( f"Wavelength {wavelength_nm} nm not found in {CORN_SPEC_PATH} header" )
    return wl_index[key]

def _mandatory_row( wavelength_nm: int, sign: float = 1.0 ) -> np.ndarray:
    """One constraint row of length p: +-1 at the given wavelength's column, 0 elsewhere."""
    row = np.zeros( p )
    row[_col( wavelength_nm )] = sign
    return row

# ------------------------------------------------------------------
# Response: moisture  (Table 3a)  -- N1, N2 (both recommended)
#   N1:  beta_1940         >= epsilon
#   N2: -beta_2080          >= epsilon
# ------------------------------------------------------------------
A_moisture = np.array([
    _mandatory_row( 1940,  1.0 ),   # N1
    _mandatory_row( 2080, -1.0 ),   # N2
])
b_moisture = np.array( [epsilon, epsilon] )

# ------------------------------------------------------------------
# Response: oil  (Table 3b)  -- N3, N4, N5 recommended; N6 optional
#   N3: beta_2308 >= epsilon
#   N4: beta_2348 >= epsilon
#   N5: beta_1762 >= epsilon
#   N6: beta_1722 >= epsilon   (optional; drop this row to use recommended-only)
# ------------------------------------------------------------------
A_oil = np.array([
    _mandatory_row( 2308, 1.0 ),   # N3
    _mandatory_row( 2348, 1.0 ),   # N4
    _mandatory_row( 1762, 1.0 ),   # N5
    _mandatory_row( 1722, 1.0 ),   # N6 (optional)
])
b_oil = np.array( [epsilon, epsilon, epsilon, epsilon] )

# Recommended-only variant of the oil constraints (N6 dropped), per the docx's note in 2.3.
A_oil_recommended = A_oil[:3, :]
b_oil_recommended = b_oil[:3]

# ------------------------------------------------------------------
# Response: protein  (Table 5)  -- N7, N8 (both recommended)
#   N7: beta_2174 >= epsilon
#   N8: beta_2058 >= epsilon
# ------------------------------------------------------------------
A_protein = np.array([
    _mandatory_row( 2174, 1.0 ),   # N7
    _mandatory_row( 2058, 1.0 ),   # N8
])
b_protein = np.array( [epsilon, epsilon] )

# ------------------------------------------------------------------
# Response: starch (optional case)  (Table 6)  -- N9 (optional)
#   N9: beta_2100 >= epsilon
# ------------------------------------------------------------------
A_starch = np.array([
    _mandatory_row( 2100, 1.0 ),   # N9
])
b_starch = np.array( [epsilon] )

# ------------------------------------------------------------------
# Convenience lookup, one (A, b) pair per response -- mirrors how simulation_CV.py's
# main() consumes a single (A, b) pair per sim_name.
# ------------------------------------------------------------------
CORN_CONSTRAINTS = {
    'moisture': ( A_moisture, b_moisture ),
    'oil':      ( A_oil_recommended, b_oil_recommended ),      # used in the paper (N3-N5)
    'oil_optional': ( A_oil, b_oil ),                          # adds the optional N6 row (not used in the paper)
    'protein':  ( A_protein, b_protein ),
    'starch':   ( A_starch, b_starch ),
}

if __name__ == '__main__':
    for name, ( A, b ) in CORN_CONSTRAINTS.items():
        l_mandatory = int( np.sum( np.any( A != 0, axis = 0 ) ) )
        print( f"{name:>16s} | A: {A.shape}  b: {b.shape}  l_mandatory = {l_mandatory}" )
        constraint_df = pd.DataFrame( A, columns = wavelengths )
        constraint_df['b'] = b
        constraint_df.to_csv( f"Data/NIR Corn/Constraints/{name}.csv", index = False )