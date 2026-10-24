"""
Literature-supported coefficient constraints for the COSMOS riboflavin application.

Builds the matrix form  A*beta >= b  (with A = [A_C  0], as in Python/simulation_CV.py's
WAT example) for the riboflavin log-production-rate model (see
Docs/Word/COSMOS_constraints_corn_riboflavin.docx, Sections 3.2-3.3, Table 5).
Only one response (log riboflavin production rate) is modeled, so there is a single (A, b).

Predictors are the 4088 gene-expression probes in Data/Riboflavin/riboflavin_x.csv;
the leading "sample" column is the sample ID, not a feature, so X has p = 4088 columns
and column j of X corresponds to feature index j here.

Constraint IDs follow the docx (R1-R8):
  R1-R4 (recommended): beta_j  >= epsilon                          for j in {RIBG, RIBB, RIBA, RIBH}
  R5-R8 (optional):    -beta_j >= -u_j  (i.e. beta_j <= u_j)        for j in {RIBG, RIBB, RIBA, RIBH}
where u_j = 1 on the raw log-log scale (the docx's alternative, u_j = ||x~_j|| / ||y~|| after
column normalization, would have to be recomputed per fold since the pipeline normalizes X, y
inside the CV loop -- see Normalize() in simulation_CV.py -- so the fixed-b convention already
used for the WAT/epsilon rows is kept here with u_j = 1).
"""

import numpy as np, os, pandas as pd

epsilon = 1e-4

# ------------------------------------------------------------------
# Load just the header of the riboflavin predictors to recover the gene -> column-index map.
# Column 0 ("sample") is the sample ID and is not a predictor, so it is dropped before indexing.
# ------------------------------------------------------------------
RIBOFLAVIN_X_PATH = 'Data/Riboflavin/riboflavin_x.csv'
header = pd.read_csv( RIBOFLAVIN_X_PATH, nrows = 0 ).columns.tolist()
assert header[0] == 'sample', f"Expected first column to be 'sample', got {header[0]!r}"
genes = header[1:]                             # p = 4088 predictor names, in column order
p = len( genes )                               # 4088
gene_index = { g: j for j, g in enumerate( genes ) }   # 'RIBG_at' -> 944, etc.

def _col( gene: str ) -> int:
    """Column index (0-based, within the p predictors) of a gene probe."""
    if gene not in gene_index:
        raise KeyError( f"Gene {gene!r} not found in {RIBOFLAVIN_X_PATH} header" )
    return gene_index[gene]

def _mandatory_row( gene: str, sign: float = 1.0 ) -> np.ndarray:
    """One constraint row of length p: +-1 at the given gene's column, 0 elsewhere."""
    row = np.zeros( p )
    row[_col( gene )] = sign
    return row

RIB_GENES = ['RIBG_at', 'RIBB_at', 'RIBA_at', 'RIBH_at']   # order matches Table 5 (C = (RIBG, RIBB, RIBA, RIBH))
u = { gene: 1.0 for gene in RIB_GENES }                     # u_j on the raw log-log scale (see docstring)

# ------------------------------------------------------------------
# R1-R4 (recommended): sign constraints on the four rib-operon enzymes
#   R1: beta_RIBG >= epsilon
#   R2: beta_RIBB >= epsilon
#   R3: beta_RIBA >= epsilon
#   R4: beta_RIBH >= epsilon
# ------------------------------------------------------------------
A_recommended = np.array( [_mandatory_row( gene, 1.0 ) for gene in RIB_GENES] )
b_recommended = np.array( [epsilon] * len( RIB_GENES ) )

# ------------------------------------------------------------------
# R5-R8 (optional): MCA upper bounds, -beta_j >= -u_j
#   R5: -beta_RIBG >= -u_RIBG
#   R6: -beta_RIBB >= -u_RIBB
#   R7: -beta_RIBA >= -u_RIBA
#   R8: -beta_RIBH >= -u_RIBH
# ------------------------------------------------------------------
A_optional = np.array( [_mandatory_row( gene, -1.0 ) for gene in RIB_GENES] )
b_optional = np.array( [-u[gene] for gene in RIB_GENES] )

# Full matrix (R1-R8), matching Table 5 exactly.
A = np.vstack( [A_recommended, A_optional] )
b = np.concatenate( [b_recommended, b_optional] )

# Recommended-only variant (R1-R4), for a run that skips the MCA upper bounds.
A_recommended_only = A_recommended
b_recommended_only = b_recommended

# ------------------------------------------------------------------
# Convenience lookup, one (A, b) pair per constraint set -- mirrors
# Python/NIR_Corn_constraints.py's CORN_CONSTRAINTS so that Python/real_CV.py
# can consume both datasets through the same Data/<dataset>/Constraints/<name>.csv layout.
# ------------------------------------------------------------------
RIBOFLAVIN_CONSTRAINTS = {
    'riboflavin':             ( A_recommended_only, b_recommended_only ),     # R1-R4, used in the paper
    'riboflavin_optional':    ( A, b ),                                       # R1-R8, adds optional upper bounds (not used in the paper)
}

if __name__ == '__main__':
    os.makedirs( 'Data/Riboflavin/Constraints', exist_ok = True )
    for name, ( A_, b_ ) in RIBOFLAVIN_CONSTRAINTS.items():
        # Mandatory features are those touched by a row with b_i > 0 (a sign/upper
        # bound row with b_i <= 0 is satisfied by beta_j = 0, so it mandates nothing);
        # this matches Constrained_Sparse_Model_Selection's own C_mandatory derivation.
        l_mandatory = int( np.sum( np.any( ( A_ != 0 ) & ( b_[:, np.newaxis] > 0 ), axis = 0 ) ) )
        print( f"{name:>22s} | A: {A_.shape}  b: {b_.shape}  l_mandatory = {l_mandatory}" )
        constraint_df = pd.DataFrame( A_, columns = genes )
        constraint_df['b'] = b_
        constraint_df.to_csv( f"Data/Riboflavin/Constraints/{name}.csv", index = False )