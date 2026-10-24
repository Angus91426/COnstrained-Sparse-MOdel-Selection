"""
Verify_Inner_Solver.py

Checks that the exact inner solver of COSMOS (inner_QP_solver + _solve_lcp)
returns the global minimizer of the fixed-support QP

    min_b  b' H_SS b - c_S' b     s.t.  A_S b >= b_vec,
    H_SS = (1 - alpha) X_S' X_S + lambda I,   c_S = (2 - alpha) X_S' y.

Part A: random fixed supports, COSMOS vs. Clarabel (interior-point, via cvxpy).
Part B: full COSMOS fits; KKT residuals of every inner solve on the real path.

Outputs (in OUT_DIR):
    PartA_Clarabel_comparison.csv, PartB_KKT_on_fits.csv, Summary.txt
"""
import os, time, warnings
import numpy as np, pandas as pd
import cvxpy as cp
from sklearn.model_selection import KFold

from Python.global_ import Normalize
from Python.wat_constants import a1, a2
from Python.models_ import Constrained_Sparse_Model_Selection

warnings.filterwarnings( "ignore" )

# ──────────────────────────────────────────────────────────────────────────────
# Configuration (edit paths if your folder layout differs)
# ──────────────────────────────────────────────────────────────────────────────
SEED        = 11220426
N_SUPPORTS  = 10                 # random supports per (dataset, sample size) in Part A
N_FITS      = 3                  # random hyperparameter draws per dataset in Part B
SAMPLE_SIZES = [55, 110, 165, 220, 275, 330]
OUT_DIR     = "Output/Verify_Inner_Solver"

WAT_BASE    = "Data/Base Data/{n}/Synthetic.csv"                     # p = 6
WAT_HD      = "Data/Irrelevant Nonlinear/{p}_p/{n}.csv"              # p = 530 / 5030
CORN_X      = "Data/NIR Corn/corn_m5spec.csv"
CORN_Y      = "Data/NIR Corn/corn_propvals.csv"
CORN_CONS   = "Data/NIR Corn/Constraints/{resp}.csv"                 # rows of A, last column 'b'
RIBO_X      = "Data/Riboflavin/riboflavin_x.csv"
RIBO_Y      = "Data/Riboflavin/riboflavin_y.csv"

EPS = 1e-4
WAT_A = np.array([
    [0,  0,     1, 0,  0,  0],
    [0,  0,     0, 0,  0,  1],
    [1, -1,     0, 0,  0,  0],
    [1,  0, -a1(), 0,  0,  0],
    [0,  1, -a2(), 0,  0,  0],
    [0,  0,     0, 1, -1,  0],
    [0,  0,     0, 0,  1, -1]
], dtype = float)
WAT_b = np.full( 7, EPS )
WAT_GROUPS = [ list( range( 6 + 5 * g, 11 + 5 * g ) ) for g in range( 6 ) ]   # nonlinear terms

# ──────────────────────────────────────────────────────────────────────────────
# Data loaders: each returns a list of dicts with X, y, A, b, groups, name
# ──────────────────────────────────────────────────────────────────────────────
def first_fold( X, y ):
    """Training part of the first CV fold, normalized exactly as in the experiments."""
    kf = KFold( n_splits = 5, shuffle = True, random_state = SEED )
    tr, te = next( iter( kf.split( X ) ) )
    X_tr, _, y_tr, _ = Normalize( X_train = X[tr], y_train = y[tr], X_test = X[te], y_test = y[te] )
    return np.asarray( X_tr, float ), np.asarray( y_tr, float ).ravel()

def load_wat():
    cases = []
    for n in SAMPLE_SIZES:
        d = pd.read_csv( WAT_BASE.format( n = n ) )
        X, y = first_fold( d.iloc[:, 1:].values, d.iloc[:, 0].values )
        cases.append( dict( name = f"WAT p=6 n={n}", X = X, y = y, A = WAT_A, b = WAT_b, groups = None ) )
        for p in ( 500, 5000 ):
            path = WAT_HD.format( p = p, n = n )
            if not os.path.exists( path ):
                continue
            d = pd.read_csv( path )
            X, y = first_fold( d.iloc[:, 1:].values, d.iloc[:, 0].values )
            cases.append( dict( name = f"WAT p={X.shape[1]} n={n}", X = X, y = y,
                                A = WAT_A, b = WAT_b, groups = WAT_GROUPS ) )
            # Sign-constrained variant: add beta_j >= 0 (b_j = 0) on 5 nonlinear terms,
            # which makes them optional but sign-constrained when selected.
            if p == 500 and n == 110:
                rng = np.random.default_rng( SEED )
                cols = rng.choice( np.arange( 6, 36 ), size = 5, replace = False )
                A_ext = np.zeros( ( 7 + len( cols ), 36 ) )
                A_ext[:7, :6] = WAT_A
                for r, j in enumerate( cols ):
                    A_ext[7 + r, j] = 1.0
                b_ext = np.concatenate( [WAT_b, np.zeros( len( cols ) )] )
                cases.append( dict( name = f"WAT p={X.shape[1]} n={n} +sign", X = X, y = y,
                                    A = A_ext, b = b_ext, groups = WAT_GROUPS ) )
    return cases

def load_corn():
    cases = []
    if not os.path.exists( CORN_X ):
        return cases
    Xd = pd.read_csv( CORN_X ).iloc[:, 1:]                   # drop 'sample'
    Yd = pd.read_csv( CORN_Y ).iloc[:, 1:]
    for resp in ( "moisture", "oil", "protein", "starch" ):
        cons = pd.read_csv( CORN_CONS.format( resp = resp ) )
        b = cons["b"].values.astype( float )
        A = cons.drop( columns = ["b"] ).values.astype( float )
        X, y = first_fold( Xd.values, Yd[resp.capitalize()].values )
        cases.append( dict( name = f"Corn {resp}", X = X, y = y, A = A, b = b, groups = None ) )
    return cases

def load_riboflavin():
    if not os.path.exists( RIBO_X ):
        return []
    # Both files have a 'sample' column (array names); use it as the index so that
    # X and y are aligned by sample name rather than by row position.
    Xd = pd.read_csv( RIBO_X, index_col = "sample" )                 # 71 x 4088 genes
    yd = pd.read_csv( RIBO_Y, index_col = "sample" )                 # 71 x 1, column 'y'
    if set( Xd.index ) != set( yd.index ):
        raise ValueError( "riboflavin_x.csv and riboflavin_y.csv contain different samples." )
    y_raw = yd.loc[Xd.index, "y"].values.astype( float )

    genes = [ "RIBG_at", "RIBB_at", "RIBA_at", "RIBH_at" ]           # RIBT_at stays unconstrained
    A = np.zeros( ( len( genes ), Xd.shape[1] ) )
    for r, g in enumerate( genes ):
        A[r, Xd.columns.get_loc( g )] = 1.0
    X, y = first_fold( Xd.values.astype( float ), y_raw )
    return [ dict( name = "Riboflavin", X = X, y = y, A = A, b = np.full( len( genes ), EPS ),
                   groups = None ) ]

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def objective( X, y, alpha, lam, beta ):
    """f(beta) = beta' H beta - c' beta without forming H."""
    r = X @ beta
    return ( 1 - alpha ) * r @ r + lam * beta @ beta - ( 2 - alpha ) * ( X.T @ y ) @ beta

def A_on_support( A, S, p ):
    """Columns of A indexed by S (A may have fewer columns than p)."""
    A_full = np.zeros( ( A.shape[0], p ) )
    A_full[:, :A.shape[1]] = A
    return A_full[:, S]

def violation( A_S, b, beta_S ):
    return max( 0.0, float( np.max( b - A_S @ beta_S ) ) ) if len( b ) else 0.0

def kkt_residuals( model ):
    """KKT residuals of the current inner solution stored in a COSMOS model."""
    S       = np.asarray( model.S, dtype = int )
    X_S     = model.X_[:, S]
    beta_S  = model.beta_[S]
    A_S     = A_on_support( model._A_dense, S, model.X_.shape[1] )
    rows    = np.any( A_S != 0, axis = 1 )                     # rows that touch S
    A_r, b_r, mu = A_S[rows], model.b[rows], model.mu_[rows]
    Hb      = ( 1 - model.alpha_ ) * ( X_S.T @ ( X_S @ beta_S ) ) + model.lambda_ * beta_S
    grad    = 2 * Hb - model.c[S] - A_r.T @ mu
    s       = A_r @ beta_S - b_r
    scale   = 1.0 + np.max( np.abs( model.c[S] ) )
    return dict(
        stationarity  = float( np.max( np.abs( grad ) ) / scale ),
        primal_infeas = float( max( 0.0, -np.min( s ) ) ) if len( s ) else 0.0,
        dual_infeas   = float( max( 0.0, -np.min( mu ) ) ) if len( mu ) else 0.0,
        complementarity = float( abs( mu @ s ) ) if len( s ) else 0.0,
        m_active_rows = int( rows.sum() ),
        rank_A_S      = int( np.linalg.matrix_rank( A_r ) ) if rows.any() else 0,
    )

def cosmos_inner( X, y, A, b, groups, S, alpha, lam ):
    """Run COSMOS's inner_QP_solver on a fixed support S (mirrors fit_model's setup)."""
    model = Constrained_Sparse_Model_Selection( k = len( S ), alpha_ = alpha, lambda_ = lam,
                                                A = A, b = b, group_sparsity = groups )
    p = X.shape[1]
    model.X_    = X
    model.c     = ( 2 - alpha ) * ( X.T @ y )
    model.Hjj   = ( 1 - alpha ) + lam
    model.S     = np.asarray( S, dtype = int )
    model.beta_ = np.zeros( p )
    t0 = time.perf_counter()
    model.inner_QP_solver()
    return model, time.perf_counter() - t0

def clarabel_inner( X, y, A, b, S, alpha, lam ):
    X_S  = X[:, S]
    c_S  = ( 2 - alpha ) * ( X_S.T @ y )
    A_S  = A_on_support( A, S, X.shape[1] )
    rows = np.any( A_S != 0, axis = 1 )
    bet  = cp.Variable( len( S ) )
    obj  = ( 1 - alpha ) * cp.sum_squares( X_S @ bet ) + lam * cp.sum_squares( bet ) - c_S @ bet
    cons = [ A_S[rows] @ bet >= b[rows] ] if rows.any() else []
    prob = cp.Problem( cp.Minimize( obj ), cons )
    t0 = time.perf_counter()
    prob.solve( solver = cp.CLARABEL, tol_gap_abs = 1e-10, tol_gap_rel = 1e-10, tol_feas = 1e-10,
                max_iter = 500 )
    return np.asarray( bet.value ).ravel(), prob.status, time.perf_counter() - t0

def random_support( model, p, k, rng ):
    """C plus k - l optional features drawn at random, respecting the groups."""
    optional = np.array( [ j for j in range( p ) if j not in model._C_mandatory_set ] )
    picks = model._select_top_optional( rng.permutation( optional ).tolist(), k - model.l )
    return sorted( list( model.C_mandatory ) + picks )

# ──────────────────────────────────────────────────────────────────────────────
# Part A: COSMOS vs. Clarabel on random fixed supports
# ──────────────────────────────────────────────────────────────────────────────
def part_a( cases, rng ):
    rows = []
    for case in cases:
        X, y, A, b, groups = case["X"], case["y"], case["A"], case["b"], case["groups"]
        n, p = X.shape
        probe = Constrained_Sparse_Model_Selection( k = 6, alpha_ = 0.5, lambda_ = 0.1, A = A, b = b,
                                                    group_sparsity = groups )
        l = probe.l
        n_rep = 1 if p == l else N_SUPPORTS
        for rep in range( n_rep ):
            k_max = min( p, l + 300, l + ( 6 if groups is not None else 10**9 ) if False else p )
            k = l if p == l else int( rng.integers( l + 1, min( p, max( l + 2, 2 * n ) ) + 1 ) )
            alpha = float( rng.uniform( 0, 1 ) )
            lam   = float( 10 ** rng.uniform( -8, 0 ) )
            try:
                S = random_support( probe, p, k, rng )
            except ValueError:
                continue
            model, t_cos = cosmos_inner( X, y, A, b, groups, S, alpha, lam )
            beta_cos = model.beta_[S]
            beta_cla, status, t_cla = clarabel_inner( X, y, A, b, S, alpha, lam )
            full_cos = np.zeros( p ); full_cos[S] = beta_cos
            full_cla = np.zeros( p ); full_cla[S] = beta_cla
            f_cos = objective( X, y, alpha, lam, full_cos )
            f_cla = objective( X, y, alpha, lam, full_cla )
            A_S = A_on_support( A, S, p )
            kkt = kkt_residuals( model )
            rows.append( dict(
                dataset = case["name"], n_train = n, p = p, k = len( S ), alpha = alpha, lam = lam,
                clarabel_status = status,
                f_cosmos = f_cos, f_clarabel = f_cla,
                f_diff = f_cos - f_cla,                                   # < 0: COSMOS better
                f_rel_diff = ( f_cos - f_cla ) / ( 1 + abs( f_cla ) ),
                beta_max_abs_diff = float( np.max( np.abs( beta_cos - beta_cla ) ) ),
                viol_cosmos = violation( A_S, b, beta_cos ),
                viol_clarabel = violation( A_S, b, beta_cla ),
                n_active_mu = int( np.sum( model.mu_ > 1e-12 ) ),
                time_cosmos = t_cos, time_clarabel = t_cla, **kkt ) )
        print( f"[Part A] {case['name']:<28s} done" )
    return pd.DataFrame( rows )

# ──────────────────────────────────────────────────────────────────────────────
# Part B: KKT residuals of every inner solve inside full COSMOS fits
# ──────────────────────────────────────────────────────────────────────────────
class LoggedCOSMOS( Constrained_Sparse_Model_Selection ):
    def inner_QP_solver( self ):
        super().inner_QP_solver()
        if not hasattr( self, "kkt_log" ):
            self.kkt_log = []
        self.kkt_log.append( kkt_residuals( self ) )

def part_b( cases, rng ):
    rows = []
    for case in cases:
        X, y, A, b, groups = case["X"], case["y"], case["A"], case["b"], case["groups"]
        n, p = X.shape
        probe = Constrained_Sparse_Model_Selection( k = 6, alpha_ = 0.5, lambda_ = 0.1, A = A, b = b,
                                                    group_sparsity = groups )
        l = probe.l
        for rep in range( 1 if p == l else N_FITS ):
            k = l if p == l else int( rng.integers( l + 1, min( p, max( l + 2, n ) ) + 1 ) )
            alpha = float( rng.uniform( 0, 1 ) )
            lam   = float( 10 ** rng.uniform( -8, 0 ) )
            model = LoggedCOSMOS( k = k, alpha_ = alpha, lambda_ = lam, A = A, b = b, group_sparsity = groups )
            try:
                init = model.Initial_Sol( X, y )
                t0 = time.perf_counter()
                model.fit_model( X = X, y = y, initial_sol = init )
                t_fit = time.perf_counter() - t0
            except ValueError as e:                      # e.g. group constraints make k infeasible
                print( f"[Part B] {case['name']}: skipped k={k} ({e})" )
                continue
            log = pd.DataFrame( model.kkt_log )
            rows.append( dict(
                dataset = case["name"], n_train = n, p = p, k = k, alpha = alpha, lam = lam,
                outer_iters = model.iter_, inner_solves = len( log ), time_fit = t_fit,
                max_stationarity = log["stationarity"].max(),
                max_primal_infeas = log["primal_infeas"].max(),
                max_dual_infeas = log["dual_infeas"].max(),
                max_complementarity = log["complementarity"].max(),
                rank_deficient_solves = int( ( log["rank_A_S"] < log["m_active_rows"] ).sum() ) ) )
        print( f"[Part B] {case['name']:<28s} done" )
    return pd.DataFrame( rows )

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs( OUT_DIR, exist_ok = True )
    rng = np.random.default_rng( SEED )
    cases = load_wat() + load_corn() + load_riboflavin()
    print( f"{len( cases )} dataset cases loaded." )

    df_a = part_a( cases, rng )
    df_a.to_csv( f"{OUT_DIR}/PartA_Clarabel_comparison.csv", index = False )
    df_b = part_b( cases, rng )
    df_b.to_csv( f"{OUT_DIR}/PartB_KKT_on_fits.csv", index = False )

    df_a["family"] = df_a["dataset"].str.extract( r"^(WAT p=\d+(?: n=\d+ \+sign)?|Corn|Riboflavin)" )[0] \
                         .str.replace( r" n=\d+", "", regex = True )
    summ_a = df_a.groupby( "family" ).agg(
        supports = ( "f_diff", "size" ),
        max_abs_f_diff = ( "f_diff", lambda v: np.max( np.abs( v ) ) ),
        max_rel_f_diff = ( "f_rel_diff", lambda v: np.max( np.abs( v ) ) ),
        cosmos_worse_by_gt_1e8 = ( "f_rel_diff", lambda v: int( ( v > 1e-8 ).sum() ) ),
        max_beta_diff = ( "beta_max_abs_diff", "max" ),
        max_viol_cosmos = ( "viol_cosmos", "max" ),
        max_viol_clarabel = ( "viol_clarabel", "max" ),
        max_stationarity = ( "stationarity", "max" ),
        max_complementarity = ( "complementarity", "max" ),
        singular_M_cases = ( "rank_A_S", lambda v: int( ( v < df_a.loc[v.index, "m_active_rows"] ).sum() ) ),
        non_optimal_clarabel = ( "clarabel_status", lambda v: int( ( v != "optimal" ).sum() ) ) )

    pd.set_option( "display.width", 250 ); pd.set_option( "display.max_columns", 30 )
    report = [ "=== Part A: COSMOS inner solver vs. Clarabel (random fixed supports) ===",
               summ_a.to_string( float_format = lambda x: f"{x:.2e}" ), "",
               "=== Part B: KKT residuals over all inner solves in full COSMOS fits ===",
               df_b.to_string( float_format = lambda x: f"{x:.2e}" ), "",
               f"Overall Part A: {len( df_a )} supports, max |f_COSMOS - f_Clarabel| / (1 + |f|) = "
               f"{df_a['f_rel_diff'].abs().max():.2e}, max COSMOS violation = {df_a['viol_cosmos'].max():.2e}",
               f"Overall Part B: {int( df_b['inner_solves'].sum() )} inner solves, max stationarity = "
               f"{df_b['max_stationarity'].max():.2e}, max complementarity = {df_b['max_complementarity'].max():.2e}" ]
    text = "\n".join( report )
    print( text )
    with open( f"{OUT_DIR}/Summary.txt", "w" ) as f:
        f.write( text )