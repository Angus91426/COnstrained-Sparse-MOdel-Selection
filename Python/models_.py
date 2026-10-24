import os
import platform


def _resolve_r_home() -> str:
    """
    Locate R_HOME without hardcoding a specific R version, so that upgrading R
    (as happened before, breaking rpy2) doesn't silently break this again.
    """
    # Respect an R_HOME that is already set (e.g. by the user or a conda environment).
    if os.environ.get('R_HOME'):
        return os.environ['R_HOME']

    if platform.system() == 'Windows':
        # The R installer always writes this key regardless of version; it's
        # the same lookup RStudio/reticulate use, so it survives R upgrades.
        try:
            import winreg
            for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
                try:
                    with winreg.OpenKey(hive, r"SOFTWARE\R-core\R") as key:
                        install_path, _ = winreg.QueryValueEx(key, "InstallPath")
                        if os.path.isdir(install_path):
                            return install_path
                except OSError:
                    continue
        except ImportError:
            pass

        # Registry lookup failed (e.g. portable R install) - fall back to the
        # newest R-x.y.z folder under the default install location.
        base = r"C:\Program Files\R"
        if os.path.isdir(base):
            candidates = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
            if candidates:
                candidates.sort(key=lambda d: [int(p) for p in d.split('R-')[-1].split('.') if p.isdigit()])
                return os.path.join(base, candidates[-1])

        return r"C:\Program Files\R\R-4.5.3"  # last-resort default

    # macOS: `Resources` is a symlink R itself keeps pointed at whichever
    # version is currently active (Versions/Current), so this needs no
    # version number and survives R upgrades on its own.
    if platform.system() == 'Darwin':
        return "/Library/Frameworks/R.framework/Resources"

    # Linux and other systems: ask R itself where it lives.
    import subprocess
    try:
        return subprocess.check_output( ['R', 'RHOME'], text = True ).strip()
    except ( OSError, subprocess.CalledProcessError ):
        return '/usr/lib/R'

os.environ['R_HOME'] = _resolve_r_home()

if platform.system() == 'Windows':
    # R's DLLs (R.dll, Rblas.dll, Rlapack.dll, ...) live under bin/x64 and are
    # not found by LoadLibrary unless this directory is on PATH - without it,
    # loading base packages like 'stats' fails with a cryptic LoadLibrary error.
    _r_bin = os.path.join(os.environ['R_HOME'], 'bin', 'x64')
    if os.path.isdir(_r_bin):
        os.environ['PATH'] = _r_bin + os.pathsep + os.environ.get('PATH', '')

import numpy as np, gurobipy as gp
import scipy.sparse as sp

import time, warnings

from gurobipy import GRB
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from rich import print
from rich.text import Text
from scipy.optimize import nnls
from scipy.linalg import cho_factor, cho_solve, LinAlgError, solve_triangular

import rpy2.robjects as ro
from rpy2.robjects.packages import importr

_L0Learn_R = importr( "L0Learn" )

class Holistic_Regression:
    def __init__( self, k: int, Gamma: float, constant_1: float, constant_2: float, min_coef_n: float = 0.001, min_coef_p: float = 0.001, M: float = 500, \
                    time_limit: int = None, relax: bool = False, OutputFlag: bool = False, log_file: str = None, \
                    group_sparsity: list = None ):
        
        self.k = k                                  # Model parameter 1: maximum number of features to select
        self.Gamma = Gamma                          # Model parameter 2: regularization parameter
        self.constant_1 = constant_1                # Constraint parameter 1: constant for the proportionality constraint
        self.constant_2 = constant_2                # Constraint parameter 2: constant for the proportionality constraint
        self.min_coef_n = min_coef_n                # Minimum coefficient value for n-features
        self.min_coef_p = min_coef_p                # Minimum coefficient value for p-features
        self.M = M                                  # Big M constant
        self.time_limit = time_limit                # Time limit for Gurobi optimization
        self.relax = relax                          # Whether to relax integer constraints
        self.OutputFlag = OutputFlag                # Whether to output Gurobi logs
        self.log_file = log_file                    # Log file path
        self.group_sparsity = group_sparsity        # Group sparsity constraints
    
    def fit_model( self, X: np.array, y: np.array ):
        """
        Fit Holistic regression model with special constraints.
        
        Parameters
        ----------
        X : np.array
            Feature matrix.
        y : np.array
            Target vector.
        
        Source:
        ----------
        Bertsimas, D., & Li, M. L. (2020). Scalable holistic linear regression. Operations Research Letters, 48(3), 203-208.
        https://doi.org/10.1016/j.orl.2020.02.008
        Bertsimas, D., & King, A. (2016). OR forum—An algorithmic approach to linear regression. Operations Research, 64(1), 2-16.
        http://dx.doi.org/10.1287/opre.2015.1436
        """
        with gp.Env( empty = True ) as env:
            env.setParam( 'OutputFlag', 1 )  # always 1 so log file receives full detail
            env.setParam( 'LogToConsole', 1 if self.OutputFlag else 0 )
            env.start()
            with gp.Model( "Holistic_Regression", env = env ) as m:
                # Setup model parameters
                m.Params.IntFeasTol = 1e-9
                m.Params.LogToConsole = 1 if self.OutputFlag else 0  # controls terminal only
                if self.log_file is not None:
                    m.Params.LogFile = self.log_file
                if self.time_limit is not None:
                    m.Params.TimeLimit = self.time_limit
                
                # Decision variables
                beta = m.addVars( X.shape[1], lb = -self.M, ub = self.M, vtype = GRB.CONTINUOUS, name = "beta" ) # Special constraints no.1 => The coefficient should be non-negative
                beta_abs = m.addVars( X.shape[1], lb = 0.0, vtype = GRB.CONTINUOUS, name = "beta_abs" )
                z = m.addVars( X.shape[1], vtype = GRB.BINARY, name = "z" )
                error = m.addVars( X.shape[0], lb = -self.M, ub = self.M, vtype = GRB.CONTINUOUS, name = "error" )
                
                # Objective function
                obj = gp.QuadExpr()
                for i in range( X.shape[0] ):
                    obj += ( error[i] * error[i] )
                obj /= 2
                obj += ( self.Gamma * beta_abs.sum() )
                m.setObjective( obj, GRB.MINIMIZE )
                
                # Absolute value constraints
                for j in range( X.shape[1] ):
                    m.addGenConstrAbs( beta_abs[j], beta[j] )
                
                epsilon = 10 ** ( -4 )
                
                # """ Monotonicity
                m.addConstr( beta[0] >= beta[1] + epsilon )
                m.addConstr( beta[2] >= epsilon ) # Non-negativity
                
                if self.relax: # Allow WAT.1/p, WAT.2/p, and WAT.3/p to be 0 for model sparsity
                    m.addConstr( beta[3] >= 0 )
                    m.addConstr( beta[4] >= 0 )
                    m.addConstr( beta[5] >= 0 ) # Non-negativity
                else:
                    m.addConstr( beta[3] >= beta[4] + epsilon )
                    m.addConstr( beta[4] >= beta[5] + epsilon )
                    m.addConstr( beta[5] >= epsilon ) # Non-negativity
                # """
                
                # """ Proportionality
                m.addConstr( beta[0] >= self.constant_1 * beta[2] + epsilon )
                m.addConstr( beta[1] >= self.constant_2 * beta[2] + epsilon )
                # """
                
                # error = ( y - X*beta )
                for i in range( X.shape[0] ):
                    m.addConstr( error[i] == y[i] - gp.quicksum( X[i, j] * beta[j] for j in range( X.shape[1] ) ) )
                
                # Big-M constraint
                for j in range( X.shape[1] ):
                    m.addConstr( beta[j] <= self.M * z[j] )
                    m.addConstr( - self.M * z[j] <= beta[j] )
                
                # Total number of selected features constraint
                m.addConstr( z.sum() <= self.k )

                # Group sparsity: at most one feature per group
                if self.group_sparsity is not None:
                    for g_idx, group in enumerate( self.group_sparsity ):
                        for j in group:
                            if j < 0 or j >= X.shape[1]:
                                raise ValueError( f"group_sparsity[{g_idx}] contains index {j}, which is out of range [0, {X.shape[1]})." )
                        m.addConstr( gp.quicksum( z[j] for j in group ) <= 1 )

                # Solve the model
                start = time.time()
                m.optimize()
                end = time.time()
                
                if m.Status == GRB.INFEASIBLE:
                    m.computeIIS()
                    m.write("IIS.ilp")
                    print("Error: Model infeasible.")
                
                self.time_ = end - start
                self.MIPGap_ = m.MIPGap
                
                # Get the solution of beta
                self.beta_ = []
                for v in m.getVars():
                    if 'beta' in v.varName and 'beta_abs' not in v.varName:
                        self.beta_.append( v.x )
    
    def predict( self, X: np.array ):
        return np.dot( X, self.beta_ )

class Partial_Least_Squares:
    def __init__( self, ncomp: int ):
        self.ncomp = ncomp
        self.param_ = {
            'ncomp': ncomp
        }
    
    def fit_model( self, X: np.array, y: np.array ):
        """
        Fit Partial Least Squares model.
        
        Parameters
        ----------
        X : np.array
            Feature matrix.
        y : np.array
            Target vector.
        
        """
        self.pls = PLSRegression( n_components = self.ncomp, scale = False )
        start_time = time.time()
        self.pls.fit( X, y )
        end_time = time.time()
        self.time_ = end_time - start_time
        self.beta_ = self.pls.coef_.flatten()
    
    def predict( self, X: np.array ):
        y_pred = self.pls.predict( X )
        return y_pred

class Sparse_Partial_Least_Squares:
    def __init__( self, K: int, eta: float, kappa = 0.5, select = "pls2", fit = "simpls",
                    scale_x = False, scale_y = False, eps = 1e-4, maxstep = 100, trace = False ):
        
        self.K = K
        self.eta = eta
        self.kappa = kappa
        self.select = select
        self.fit = fit
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.eps = eps
        self.maxstep = maxstep
        self.trace = trace
        self.param_ = {
            'K': K,
            'eta': eta
        }
    
    def ust( self, b: np.ndarray, eta: float ) -> np.ndarray:
        """
        Univariate Soft‐Thresholding Estimator
        
        Parameters
        ----------
        b : np.ndarray
            One‐dimensional array of coefficients.
        eta : float
            Threshold parameter. If eta >= 1, returns a zero vector.
        
        Returns
        -------
        np.ndarray
            A new array of the same shape as b, where each entry has been
            soft‐thresholded by (eta * max(abs(b))):
            
                b_ust[i] = sign(b[i]) * max(|b[i]| - eta * max(abs(b)), 0)
        """
        b = np.asarray( b, dtype = float )
        b_ust = np.zeros_like( b, dtype = float )
        
        if eta < 1:
            max_abs_b = np.max( np.abs( b ) )
            valb = np.abs( b ) - eta * max_abs_b
            mask = ( valb >= 0 )
            b_ust[mask] = valb[mask] * np.sign(b[mask])
        
        return b_ust
    
    def spls_dv( self, Z: np.ndarray,
                eta: float,
                kappa: float,
                eps: float,
                maxstep: int) -> np.ndarray:
        """
        Fit a SPLS (Sparse Partial Least Squares) direction vector.
        
        This function solves for the SPLS direction vector 'c' given:
        - Z:   a (p × q) data matrix (features in rows, responses in columns)
        - eta: sparsity‐inducing threshold parameter (0 <= eta < 1)
        - kappa: tuning parameter in (0, 0.5] that controls the PLS weighting
        - eps: convergence tolerance for iterative updates
        - maxstep: maximum number of iterations
        
        Returns
        -------
        c : np.ndarray, shape (p,)
            The estimated direction vector (one-dimensional array of length p).
        
        Notes
        -----
        1. If q == 1, this reduces to simple univariate soft‐thresholding on Z.
        2. If q > 1 and kappa == 0.5, there is a closed‐form SVD step for updating
        the PLS weight `a`. Otherwise, for 0 < kappa < 0.5, we solve a small
        Lagrange multiplier problem via bisection at each iteration.
        3. The input Z is first scaled by median(abs(Z)) to improve numerical stability.
        """
        # Ensure Z is a NumPy array
        Z = np.asarray( Z, dtype = float )
        p, q = Z.shape
        
        # 1) Normalize Z by its median absolute value
        Znorm1 = np.median( np.abs( Z ) )
        if Znorm1 == 0:
            raise ValueError( "All entries of Z are zero. Cannot normalize by median(abs(Z))." )
        Z = Z / Znorm1
        
        # Initialize c
        if q == 1:
            # Univariate response: simply apply soft-thresholding to the (p×1) vector Z[:,0]
            # Note: ust expects a 1D array
            return self.ust( Z.ravel(), eta )
        
        if q > 1:
            # Multivariate response: form M = Z Z^T (p × p matrix)
            M = Z.dot( Z.T )
            dis = 10.0
            i_step = 1
            
            # Case A: kappa == 0.5 --> use SVD‐based updates
            if np.isclose( kappa, 0.5 ):
                # Initialize c outside the unit circle (arbitrary large starting point)
                c = np.full( ( p, ), 10.0, dtype = float )
                c_old = c.copy()
                
                while dis > eps and i_step <= maxstep:
                    # --- Update a for fixed c via the leading SVD direction of M c ---
                    # Compute M @ c as a (p × 1) vector
                    Mc = M.dot( c ).reshape( ( p, 1 ) )
                    # Compute SVD(Mc) with economy‐size so that u is (p×1) and vt is (1×1)
                    u, _, vt = np.linalg.svd( Mc, full_matrices = False )
                    # a = u @ vt  (shape: p×1), then flatten to (p,)
                    a = ( u @ vt ).ravel()
                    
                    # --- Update c for fixed a via soft‐thresholding ---
                    Mca = M.dot( a )  # shape: (p,)
                    c = self.ust( Mca, eta )
                    
                    # --- Compute convergence criterion ---
                    dis = np.max( np.abs( c - c_old ) )
                    c_old = c.copy()
                    i_step += 1
                
                return c
            
            # Case B: 0 < kappa < 0.5 --> solve via Lagrange multiplier at each step
            '''
            if 0 < kappa < 0.5:
                kappa2 = ( 1.0 - kappa ) / ( 1.0 - 2.0 * kappa )
                
                # Initial c (outside unit circle)
                c = np.full( ( p, ), 10.0, dtype = float )
                c_old = c.copy()
                
                # Define helper function h(lambda) = ||alpha||^2 - 1/kappa2^2,
                # where alpha = (M + λ I)^{-1} (M c)
                def h( libra: float ) -> float:
                    # Solve (M + λ I) alpha = M c
                    A = M + lmbda * np.eye( p )
                    rhs = M.dot( c )
                    alpha = np.linalg.solve( A, rhs )  # shape: (p,)
                    return float( alpha.dot( alpha ) - 1.0 / ( kappa2 ** 2 ) )
                
                # Pre‐scale M and c if necessary so that h(eps) and h(1e30) have opposite signs
                # (bisection requires a sign change). If not, scale both by 2 repeatedly until so.
                def ensure_sign_change():
                    """
                    Ensure that h(eps) * h(1e30) < 0 by scaling M and c.
                    If h(eps) * h(1e30) > 0, then repeatedly double M and c
                    until h(eps) is sufficiently large (> 1e5) to force a sign change.
                    """
                    nonlocal M, c
                    # Evaluate h at the endpoints
                    he = h( eps )
                    h_large = h( 1e30 )
                    if he * h_large > 0:
                        # When both have the same sign, enlarge M and c until
                        # h(eps) becomes large enough in magnitude.
                        while h( eps ) <= 1e5:
                            M = 2.0 * M
                            c = 2.0 * c
                
                # First scaling step
                ensure_sign_change()
                
                # Simple bisection to find the root of h(λ) on [eps, 1e30]
                def find_root_bisection( f, left: float, right: float, tol: float ) -> float:
                    fl = f( left )
                    fr = f( right )
                    if fl * fr > 0:
                        raise ValueError( "The function does not have opposite signs at the interval endpoints." )
                    mid = 0.5 * ( left + right )
                    while ( right - left ) / 2.0 > tol:
                        fm = f( mid )
                        if abs( fm ) < tol:
                            return mid
                        # Decide which subinterval to keep
                        if fl * fm <= 0:
                            right = mid
                            fr = fm
                        else:
                            left = mid
                            fl = fm
                        mid = 0.5 * ( left + right )
                    return mid
                
                while dis > eps and i_step <= maxstep:
                    # Re‐scale if needed on each iteration
                    ensure_sign_change()
                    
                    # --- Update a for fixed c via solving the Lagrange multiplier problem ---
                    # Find λ* in [eps, 1e30] such that h(λ*) = 0
                    try:
                        lambdas = find_root_bisection( h, eps, 1e30, tol = eps )
                    except ValueError as ve:
                        raise RuntimeError( f"Unable to find root of h(λ) in [{eps}, 1e30]: {ve}" )
                    
                    # Compute a = kappa2 * (M + λ* I)^{-1} (M c)
                    A = M + lambdas * np.eye( p )
                    rhs = M.dot( c )
                    a = kappa2 * np.linalg.solve( A, rhs )  # shape: (p,)
                    
                    # --- Update c for fixed a via soft‐thresholding ---
                    Mca = M.dot( a )
                    c = self.ust( Mca, eta )
                    
                    # --- Compute convergence criterion ---
                    dis = np.max( np.abs( c - c_old ) )
                    c_old = c.copy()
                    i_step += 1
            '''
            
        return c
    
    def correctp( self, x, y, eta, K, kappa, select, fit ):
        """
        Correct or validate parameter specifications for an SPLS model.
        
        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            The predictor matrix.
        y : array-like
            The response vector or matrix (not used in this correction step).
        eta : float or array-like
            Sparsity threshold parameter; must satisfy 0 <= eta < 1.
        K : int or array-like of ints
            Number of latent components; must be positive integers and
            cannot exceed the number of features or the sample size.
        kappa : float
            PLS tuning parameter; should lie in [0, 0.5]. Values outside this
            interval are reset to 0.5.
        select : str
            PLS algorithm for variable selection; valid options are "pls2" and "simpls".
            Invalid values are replaced by "pls2".
        fit : str
            PLS algorithm for model fitting; valid options are
            "simpls", "kernelpls", "widekernelpls", "oscorespls".
            Invalid values are replaced by "simpls".
        
        Returns
        -------
        dict
            A dictionary containing potentially corrected parameters:
            {
                "K": K,
                "eta": eta,
                "kappa": kappa,
                "select": select,
                "fit": fit
            }
        
        Raises
        ------
        ValueError
            If eta is not in [0,1), or if K is not a positive integer vector within
            appropriate bounds.
        """
        # -------------------------------------------------------
        # 1) Validate eta: must be between 0 (inclusive) and 1 (exclusive)
        # -------------------------------------------------------
        eta_arr = np.asarray( eta, dtype = float )
        if ( eta_arr.min() < 0 ) or ( eta_arr.max() >= 1 ):
            if eta_arr.max() == 1:
                raise ValueError( "eta should be strictly less than 1!" )
            if eta_arr.size == 1:
                raise ValueError( "eta should be between 0 and 1!" )
            else:
                raise ValueError( "eta should be between 0 and 1! Choose an appropriate range of eta!" )
        
        # -------------------------------------------------------
        # 2) Validate K: positive integers, not exceeding number of features or sample size
        # -------------------------------------------------------
        # Ensure K_arr is at least 1D for consistent checks
        K_arr = np.array( K, ndmin = 1)
        # Extract sample size (n) and number of predictors (p) from x.shape
        n_samples, n_features = np.asarray( x ).shape
        
        if np.max( K_arr ) > n_features:
            raise ValueError("K cannot exceed the number of predictors! Pick a smaller K!")
        if np.max( K_arr ) >= n_samples:
            raise ValueError("K cannot exceed the sample size! Pick a smaller K!")
        # Check that all entries of K_arr are positive integers
        if ( np.min( K_arr ) <= 0 ) or ( not np.all( np.equal( K_arr, K_arr.astype(int) ) ) ):
            if K_arr.size == 1:
                raise ValueError("K should be a positive integer!")
            else:
                raise ValueError("K should be a positive integer! Choose an appropriate range of K!")
        
        # -------------------------------------------------------
        # 3) Validate kappa: should be in [0, 0.5]. If not, reset to 0.5 with a warning.
        # -------------------------------------------------------
        if ( kappa > 0.5 ) or ( kappa < 0 ):
            print( "kappa should be between 0 and 0.5! kappa=0.5 is used.\n" )
            kappa = 0.5
        
        # -------------------------------------------------------
        # 4) Validate select: must be "pls2" or "simpls". Otherwise default to "pls2".
        # -------------------------------------------------------
        if select not in ( "pls2", "simpls" ):
            print( "Invalid PLS algorithm for variable selection." )
            print( "pls2 algorithm is used.\n" )
            select = "pls2"
        
        # -------------------------------------------------------
        # 5) Validate fit: must be one of the allowed PLS-fitting methods.
        #    Otherwise default to "simpls".
        # -------------------------------------------------------
        valid_fits = ["simpls", "kernelpls", "widekernelpls", "oscorespls"]
        if fit not in valid_fits:
            print( "Invalid PLS algorithm for model fitting" )
            print( "simpls algorithm is used.\n" )
            fit = "simpls"
        
        return {
            "K": K,
            "eta": eta,
            "kappa": kappa,
            "select": select,
            "fit": fit
        }
    
    def fit_model( self, X: np.array, y: np.array ):
        """
        Fit Sparse Partial Least Squares model.
        
        Parameters
        ----------
        X : np.array
            Feature matrix.
        y : np.array
            Target vector.
        
        """
        
        # ---------------------------
        # 1) Convert inputs to NumPy arrays and basic dimensions
        # ---------------------------
        X = np.asarray( X, dtype = float )
        n_samples, n_features = X.shape
        Y = np.asarray( y, dtype = float )
        if Y.ndim == 1:
            # Make Y a 2D column vector if univariate
            Y = Y.reshape( -1, 1 )
        n_y_cols = Y.shape[1]
        
        # ---------------------------
        # 2) Center Y and X
        # ---------------------------
        # Compute row vector of Y means: shape (1, n_targets)
        mu = Y.mean( axis = 0, keepdims = True )  # equivalent to (1 × n) * Y / n in R
        # Yc = Y - mu  # centered Y
        Yc = Y.copy()
        
        # Compute 1D array of X means: shape (n_features,)
        meanx = X.mean( axis = 0 )
        # Xc = X - meanx  # centered X
        Xc = X.copy()
        
        # ---------------------------
        # 3) Optionally scale X
        # ---------------------------
        normx = np.ones( n_features )
        
        # ---------------------------
        # 4) Optionally scale Y
        # ---------------------------
        if self.scale_y:
            normy = np.sqrt( ( Yc ** 2 ).sum( axis = 0 ) / ( n_samples - 1 ) )
            eps_mach = np.finfo( float ).eps
            if np.any( normy < eps_mach ):
                raise ValueError( "Some columns of Y have (almost) zero variance." )
            Yc = Yc / normy
        else:
            normy = np.ones(n_y_cols)
        
        # ---------------------------
        # 5) Initialize storage for coefficients and deflated data
        # ---------------------------
        betahat = np.zeros( ( n_features, n_y_cols ), dtype = float )
        betamat = []       # list of βhat at each component
        x1 = Xc.copy()     # deflated X, initially the full centered+scaled X
        y1 = Yc.copy()     # deflated Y, initially the full centered+scaled Y
        
        # ---------------------------
        # 6) Validate or correct parameters via correctp()
        # ---------------------------
        params = self.correctp( Xc, Yc, self.eta, self.K, self.kappa, self.select, self.fit )
        eta = params["eta"]
        K = params["K"]
        kappa = params["kappa"]
        select = params["select"]
        fit = params["fit"]
        
        # ---------------------------
        # 7) Prepare index vector ip = [0,1,...,n_features-1]
        # ---------------------------
        ip = np.arange( n_features )
        
        # ---------------------------
        # 8) Determine variable names for trace printing (if any)
        # ---------------------------
        # If X was originally a pandas DataFrame, retrieve column names; else use indices
        try:
            xnames = list( x1.columns )
        except Exception:
            xnames = list( range( n_features ) )
        
        new2As = []  # to store newly added indices at each step
        
        if self.trace:
            print( "Variables entering the active set at each step:" )
        
        # ---------------------------
        # 9) Main loop over K latent components
        # ---------------------------
        start_time = time.time()
        for k in range( 1, K + 1 ):
            # 9a) Define Z = X₁ᵀ Y₁  (shapes: (n_features × n_samples) @ (n_samples × n_targets))
            Z = x1.T.dot( y1 )  # shape (n_features, n_targets)
            
            # 9b) Fit the SPLS direction vector 'what' using spls_dv()
            #     'what' is a length‐p vector of coefficients (some zero, some nonzero)
            what = self.spls_dv( Z, eta, kappa, self.eps, self.maxstep ).ravel()  # flatten to 1D
            
            # 9c) Construct the active set A
            #     R logic: A = unique( ip[ what != 0 OR betahat[:,0] != 0 ] )
            #   In Python, betahat[:,0] is the coefficients from the previous component on target 0.
            already_in = ( betahat[:, 0] != 0 )
            newly_selected = ( what != 0 ) & ( ~already_in )
            A = np.where( ( what != 0 ) | already_in )[0]        # active indices
            new2A = np.where( newly_selected )[0]               # newly added this step
            new2As.append( new2A.tolist() )
            
            # 9d) Fit a standard PLS model on X[:, A] → Y using scikit‐learn
            #     Number of PLS components = min(k, |A|)
            n_active = A.size
            n_pls_comp = min( k, n_active )
            if n_active == 0:
                # If no active variables, everything remains zero. Skip fitting.
                betahat[:, :] = 0.0
                betamat.append( betahat.copy() )
                pj = np.zeros( ( 0, 0 ) )  # empty projection
            else:
                X_A = Xc[:, A]  # shape (n_samples, n_active)
                # We only support the "simpls" style (NIPALS) PLS. Other methods are not implemented.
                pls_model = PLSRegression( n_components = n_pls_comp, scale = False, max_iter = 500 )
                pls_model.fit( X_A, Yc )
                
                # 9e) Update betahat: place coefficients into correct rows
                # sklearn's coef_ has shape (n_active, n_targets)
                coef_A = pls_model.coef_  # shape (n_active, n_y_cols)
                temp_betahat = np.zeros_like( betahat )
                temp_betahat[A, :] = coef_A.reshape( -1, n_y_cols )
                betahat = temp_betahat
                betamat.append( betahat.copy() )
                
                # 9f) Extract the PLS X‐weights (projection) for deflation if needed
                #     sklearn attribute x_weights_ has shape (n_active, n_pls_comp)
                pj = pls_model.x_weights_
            
            # 9g) Update residuals or deflate X depending on 'select'
            if select == "pls2":
                # Deflate Y:  Y₁ ← Y - X βhat
                y1 = Yc - Xc.dot( betahat )
            else:  # select == "simpls"
                # Deflate X on the active set:  X₁[:, A] ← X[:, A] - [X[:, A] @ P]
                # where P = pj @ (pjᵀ pj)^{-1} @ pjᵀ  (projection matrix onto subspace spanned by columns of pj)
                if pj.size == 0:
                    # Nothing to deflate if no active variables
                    x1 = Xc.copy()
                else:
                    # Compute P = pj (n_active × r) @ inv(r × r) @ pjᵀ (r × n_active) → (n_active × n_active)
                    # pjᵀ pj is (r × r), invertible if pj has full column rank
                    pts_p = pj.T.dot( pj )  # (r, r)
                    inv_pts_p = np.linalg.inv( pts_p )  # (r, r)
                    P = pj.dot( inv_pts_p ).dot( pj.T )    # (n_active, n_active)
                    # Deflate columns A only; other columns of x1 remain as in Xc
                    x1 = Xc.copy()
                    XA = Xc[:, A]  # (n_samples, n_active)
                    XA_deflated = XA - XA.dot( P )  # (n_samples, n_active)
                    x1[:, A] = XA_deflated
            
            # 9h) If tracing, print newly‐entered variable names
            if self.trace:
                if new2A.size == 0:
                    print( f"- Step {k} (K={k}): no new variables entered." )
                else:
                    names_new = [xnames[i] for i in new2A]
                    print( f"- Step {k} (K={k}):" )
                    # Print up to 10 names per line
                    for i_start in range( 0, len( names_new ), 10 ):
                        line_block = names_new[i_start : i_start + 10]
                        print( "  ", "  ".join( map( str, line_block ) ) )
        
        end_time = time.time()
        self.time_ = end_time - start_time
        
        self.beta_ = betahat.flatten()
        self.A = np.array( A.tolist() )
        self.Xc = np.array( Xc )
        self.mu = np.array( mu )
        self.meanx = np.array( meanx )
        self.normx = np.array( normx )
    
    def predict( self, X: np.ndarray ):
        y_pred = X.dot( self.beta_ )
        y_pred = y_pred.flatten()
        return y_pred

class Constrained_Sparse_Model_Selection:
    def __init__( self, k: int, alpha_: float, lambda_: float, A: np.ndarray, b: np.ndarray, \
                    mandatory_init: np.ndarray = None, \
                    converge_tol: float = 1e-6, feasible_tol: float = 1e-4, optimal_tol: float = 1e-4, swap_num: int = None, maxIters: int = 200, \
                    group_sparsity: list = None, lcp_tol: float = 1e-12, \
                    verbose: bool = False, log_file: str = None ):

        self.k = k
        self.alpha_ = alpha_
        self.lambda_ = lambda_
        self.A = A
        self.b = b
        self.converge_tol = converge_tol
        self.feasible_tol = feasible_tol
        self.optimal_tol = optimal_tol
        self.lcp_tol = lcp_tol          # tolerance of the exact multiplier (LCP) solve

        # ── Constraint matrix ─────────────────────────────────────────────────
        # A (m x q) may be dense or scipy-sparse.  The number of constraint rows m
        # is small, and the fixed-support QP only needs the columns of A that
        # belong to the current support, so a dense copy is kept for slicing.
        self._A_is_sparse = sp.issparse( A )
        self._A_dense     = A.toarray() if self._A_is_sparse else np.asarray( A, dtype = float )

        # ── Mandatory vs. sign-constrained features ───────────────────────────
        # A feature is *mandatory* (always in support) only when at least one
        # constraint row with b_i > 0 involves it — that strictly forces a nonzero
        # coefficient.  Sign constraints (b_i = 0) do NOT mandate selection because
        # beta_j = 0 trivially satisfies a_j * beta_j >= 0.
        #
        # A feature is *sign-constrained* (C_sign) if it appears in at least one
        # constraint row but ALL those rows have b_i = 0.  These features are
        # optional in the support but receive bounded CD updates when selected.
        if self._A_is_sparse:
            pos_rows = np.where( b > 0 )[0]
            if len( pos_rows ) > 0:
                A_pos = A[pos_rows]
                mandatory_mask = np.asarray( ( A_pos != 0 ).sum( axis = 0 ) > 0 ).ravel()
            else:
                mandatory_mask = np.zeros( A.shape[1], dtype = bool )
            any_nonzero    = np.asarray( ( A != 0 ).sum( axis = 0 ) > 0 ).ravel()
        else:
            pos_b          = b[:, np.newaxis] > 0                         # broadcast over columns
            mandatory_mask = np.any( ( A != 0 ) & pos_b, axis = 0 )
            any_nonzero    = np.any( A != 0, axis = 0 )

        sign_mask = any_nonzero & ~mandatory_mask

        self.C_mandatory      = sorted( np.where( mandatory_mask )[0].tolist() )
        self.C_sign           = sorted( np.where( sign_mask )[0].tolist() )
        self.l                = len( self.C_mandatory )
        self._C_mandatory_set = set( self.C_mandatory )
        self._C_sign_set      = set( self.C_sign )

        # Group sparsity: at most one feature may be selected from each group.
        # Groups may overlap; a feature belonging to multiple groups blocks all of them once chosen.
        # Mandatory features are not allowed inside any group.
        self.group_sparsity = group_sparsity

        # Logging: mirrors the HR model's log_file / OutputFlag pattern.
        # verbose=True prints progress to the console; log_file (if given) writes the
        # same lines to a text file that can be monitored with  `tail -f <log_file>`.
        self.verbose  = verbose
        self.log_file = log_file
        # Open (or create) the log file eagerly so the header appears immediately.
        if self.log_file is not None:
            with open( self.log_file, 'w', buffering = 1 ) as _f:
                _f.write( "=" * 80 + "\n" )
                _f.write( f"COSMOS Solver Log | Hyperparameter: k = {self.k}, alpha = {self.alpha_:.4f}, lambda = {self.lambda_:.4f} \n" )
                _f.write( "=" * 80 + "\n" )
        if group_sparsity is not None:
            self._feature_to_groups = {}      # feature index -> set of group ids it belongs to
            offending = []
            for g_idx, group in enumerate( group_sparsity ):
                for j in group:
                    if j in self._C_mandatory_set:
                        offending.append( ( j, g_idx ) )
                    self._feature_to_groups.setdefault( j, set() ).add( g_idx )
            if offending:
                raise ValueError(
                    "group_sparsity contains mandatory features, which is not allowed. "
                    f"Offending (feature, group_id) pairs: {offending}"
                )
        else:
            self._feature_to_groups = None

        if swap_num is None:
            self.swap_num = max( 0, min( 5, self.k - self.l ) )
        else:
            self.swap_num = swap_num
        self.maxIters = maxIters
        self.param_ = {
            'k': k,
            'alpha_': alpha_,
            'lambda_': lambda_
        }

        # Feasible initial values for the C-block (mandatory) features.
        # The caller may supply them explicitly; otherwise they are computed via LP.
        if mandatory_init is not None:
            self.mandatory_init = np.asarray( mandatory_init, dtype = float )
        else:
            self.mandatory_init = self._compute_feasible_init()

    def _log( self, msg: str ):
        """
        Write *msg* to the console (if verbose) and/or to the log file (if set).
        The file is opened in line-buffered append mode so every line is flushed
        immediately — allowing  `tail -f <log_file>`  to show live progress.
        """
        if self.verbose:
            print( msg )
        if self.log_file is not None:
            with open( self.log_file, 'a', buffering = 1 ) as _f:
                _f.write( msg + "\n" )

    def _compute_feasible_init( self ) -> np.ndarray:
        """
        Find a strictly-feasible starting point for the mandatory (C-block) features
        by solving the LP:  min  sum(beta_C)  s.t.  A[:, C] @ beta_C >= b.
        Falls back to a least-squares projection if the LP fails.
        Returns an empty array when there are no mandatory features (l == 0).
        """
        if self.l == 0:
            return np.array( [] )           # nothing to initialise

        from scipy.optimize import linprog
        A_C = self.A[:, self.C_mandatory]   # shape (m, l); sparse or dense
        if sp.issparse( A_C ):
            A_C = A_C.toarray()             # linprog 'highs' handles sparse but dense is safer
        result = linprog(
            c       = np.ones( self.l ),    # minimise sum → small but feasible values
            A_ub    = -A_C,
            b_ub    = -self.b,
            bounds  = [ (None, None) ] * self.l,
            method  = 'highs',
        )
        if result.success:
            return result.x + 1e-5          # small margin for strict feasibility
        # Fallback: least-squares pseudoinverse (may be infeasible, but best effort)
        return np.linalg.lstsq( A_C, self.b, rcond = None )[0]

    def _select_top_optional( self, ranked_candidates, n_optional: int, occupied_groups: set = None ):
        """
        Greedy group-aware selection of n_optional features from a pre-ranked list of candidate
        feature indices (best first). Skips any candidate whose groups intersect already-claimed
        groups. Returns the picked indices in selection order.

        Parameters
        ----------
        ranked_candidates : iterable of int
            Candidate optional feature indices ordered from best to worst by some score.
            Mandatory features must already be excluded by the caller.
        n_optional : int
            Number of features to pick.
        occupied_groups : set, optional
            Groups already occupied by features outside this selection (e.g. members of U
            we are not removing during a swap). Defaults to empty.

        Raises
        ------
        ValueError
            If group_sparsity makes it impossible to pick n_optional features from the
            given candidate pool.
        """
        if n_optional <= 0:
            return []

        if self.group_sparsity is None:
            picks = list( ranked_candidates )[:n_optional]
            if len( picks ) < n_optional:
                raise ValueError(
                    f"Only {len( picks )} candidates available, but n_optional={n_optional} requested."
                )
            return picks

        claimed = set( occupied_groups ) if occupied_groups is not None else set()
        picks = []
        for j in ranked_candidates:
            groups_of_j = self._feature_to_groups.get( j, set() )
            if groups_of_j & claimed:
                continue                       # would violate at-most-one-per-group
            picks.append( j )
            claimed |= groups_of_j
            if len( picks ) == n_optional:
                break

        if len( picks ) < n_optional:
            raise ValueError(
                f"group_sparsity makes k infeasible: could only fill {len( picks )} of "
                f"{n_optional} optional slots from the candidate pool. Reduce k, relax groups, "
                f"or expand the feature set."
            )
        return picks

    def _occupied_groups_from_features( self, features ) -> set:
        """Return the union of group ids occupied by the given feature indices."""
        if self.group_sparsity is None:
            return set()
        occupied = set()
        for j in features:
            occupied |= self._feature_to_groups.get( j, set() )
        return occupied

    def Initial_Sol( self, X_train: np.ndarray, y_train: np.ndarray, seed:int = 11220426 ):
        p          = X_train.shape[1]
        n_optional = max( 0, self.k - self.l )
        top_corr   = []
        if n_optional != 0: 
            # Correlation screening over optional (non-mandatory) features only
            optional_indices = np.array( [ j for j in range( p ) if j not in self._C_mandatory_set ] )
            corr_ = np.zeros( p )
            for j in optional_indices:
                corr_[j] = ( 1 / X_train.shape[0] ) * np.dot( X_train[:, j], y_train )

            ranked   = optional_indices[ np.argsort( np.abs( corr_[optional_indices] ) )[::-1] ].tolist()
            top_corr = self._select_top_optional( ranked, n_optional )

        # Build initial beta: mandatory features get the feasible warm-start values
        initial_beta = np.zeros( p )
        for j, idx in enumerate( self.C_mandatory ):
            initial_beta[idx] = self.mandatory_init[j]
        rng = np.random.default_rng( seed = seed )
        if len( top_corr ) > 0:
            initial_beta[top_corr] = rng.normal( loc = 0, scale = 1, size = len( top_corr ) )
        return initial_beta

    def _ridge_initial_sol( self, X_train: np.ndarray, y_train: np.ndarray, seed: int = 11220426 ) -> np.ndarray:
        """
        Start 2: Ridge-regression-based initialization.
        Selects the top (k-l) optional features by ridge coefficient magnitude,
        capturing joint relevance under collinearity.
        Reference: Hoerl & Kennard (1970); Hazimeh & Mazumder (2020).
        """
        ridge = Ridge( alpha = 1.0, fit_intercept = False )
        ridge.fit( X_train, y_train )
        ridge_coef = np.abs( ridge.coef_ )

        # Zero out mandatory features so they don't compete for optional slots
        for idx in self.C_mandatory:
            ridge_coef[idx] = 0.0

        n_optional       = max( 0, self.k - self.l )
        optional_indices = np.array( [ j for j in range( X_train.shape[1] ) if j not in self._C_mandatory_set ] )
        ranked           = optional_indices[ np.argsort( ridge_coef[optional_indices] )[::-1] ].tolist()
        top_ridge        = self._select_top_optional( ranked, n_optional )

        initial_beta = np.zeros( X_train.shape[1] )
        for j, idx in enumerate( self.C_mandatory ):
            initial_beta[idx] = self.mandatory_init[j]
        rng = np.random.default_rng( seed = seed )
        if len( top_ridge ) > 0:
            initial_beta[top_ridge] = rng.normal( loc = 0, scale = 1, size = len( top_ridge ) )
        return initial_beta

    def _gradient_initial_sol( self, X_train: np.ndarray, y_train: np.ndarray, seed: int = 11220426 ) -> np.ndarray:
        """
        Start 3: Add-score (gradient)-based initialization.
        Computes the COSMOS add score g_j^2 / H_jj for each optional feature
        evaluated at beta = 0, then selects the top (k-l) by score.
        This is internally consistent with COSMOS's own outer combinatorial criterion.
        Reference: Hazimeh & Mazumder (2020).
        """
        # X columns are unit-norm => H_jj = (1-alpha)||x_j||^2 + lambda = (1-alpha) + lambda
        # Use scalar Hjj instead of forming the full O(np^2) Hessian matrix.
        Hjj  = ( 1 - self.alpha_ ) + self.lambda_
        c    = ( 2 - self.alpha_ ) * ( X_train.T @ y_train )
        grad = -c   # gradient at beta = 0

        add_score = np.zeros( X_train.shape[1] )
        for j in range( X_train.shape[1] ):
            if j not in self._C_mandatory_set:
                add_score[j] = ( grad[j] ** 2 ) / ( 4 * Hjj )

        n_optional       = max( 0, self.k - self.l )
        optional_indices = np.array( [ j for j in range( X_train.shape[1] ) if j not in self._C_mandatory_set ] )
        ranked           = optional_indices[ np.argsort( add_score[optional_indices] )[::-1] ].tolist()
        top_grad         = self._select_top_optional( ranked, n_optional )

        initial_beta = np.zeros( X_train.shape[1] )
        for j, idx in enumerate( self.C_mandatory ):
            initial_beta[idx] = self.mandatory_init[j]
        rng = np.random.default_rng( seed = seed )
        if len( top_grad ) > 0:
            initial_beta[top_grad] = rng.normal( loc = 0, scale = 1, size = len( top_grad ) )
        return initial_beta

    def _random_initial_sol( self, X_train: np.ndarray, y_train: np.ndarray, seed: int = 11220426 ) -> np.ndarray:
        """
        Start 4: Randomized correlation screening
        """
        p              = X_train.shape[1]
        n_optional     = max( 0, self.k - self.l )
        optional_idx   = np.array( [ j for j in range( p ) if j not in self._C_mandatory_set ] )
        rng            = np.random.default_rng( seed = seed )
        # Random ordering of optional features, then group-aware greedy pick
        permuted         = rng.permutation( optional_idx ).tolist()
        sampled_optional = self._select_top_optional( permuted, n_optional )

        initial_beta = np.zeros( p )
        for j, idx in enumerate( self.C_mandatory ):
            initial_beta[idx] = self.mandatory_init[j]
        if len( sampled_optional ) > 0:
            initial_beta[sampled_optional] = rng.normal( loc = 0, scale = 1, size = len( sampled_optional ) )
        return initial_beta

    def _sync_residual( self ):
        """
        Re-synchronise the residual  r_ = X_S beta_S  with the current support S.
        Called after every support change (proposal or rollback).  O(n|S|).
        """
        S_arr   = np.asarray( self.S, dtype = int )
        self.r_ = self.X_[:, S_arr] @ self.beta_[S_arr]

    def _compute_gradient( self ):
        # Implicit Hessian-vector product: Hβ = (1-α)X^T(Xβ) + λβ = (1-α)X^T r + λβ
        # Cost: O(np) instead of O(p²) for explicit H @ β.
        self.gradients_ = ( 2 * ( 1 - self.alpha_ ) * ( self.X_.T @ self.r_ )
                          + 2 * self.lambda_ * self.beta_
                          - self.c )
    
    def _solve_lcp( self, M: np.ndarray, s0: np.ndarray ) -> np.ndarray:
        """
        Solve  s = s0 + M mu >= 0,  mu >= 0,  mu^T s = 0   (M symmetric PSD),
        equivalently  min_{mu >= 0}  0.5 mu^T M mu + s0^T mu,
        with a Lawson-Hanson style active-set method.  The passive set P holds
        the constraints treated as active (mu_P free, s_P = 0); constraints are
        added one at a time (most violated first) and dropped when their
        multiplier would turn negative.  Singular M is handled by lstsq.
        """
        m  = len( s0 )
        mu = np.zeros( m )
        if np.all( s0 >= -self.lcp_tol ):
            return mu                                    # unconstrained optimum is feasible
        tol = min( self.lcp_tol * max( 1.0, np.max( np.abs( s0 ) ) ), 1e-9 )
        P   = np.zeros( m, dtype = bool )
        delta = 1e-10 * max( 1.0, float( np.max( np.diag( M ) ) ) )
        M     = M + delta * np.eye( m )
        for _ in range( 10 * m + 10 ):                   # outer loop: add a constraint
            s    = s0 + M @ mu                           # current primal slack
            cand = np.where( ~P & ( s < -tol ) )[0]
            if len( cand ) == 0:
                break                                    # all KKT conditions hold
            P[ cand[ np.argmin( s[cand] ) ] ] = True     # most violated constraint enters
            for _ in range( 10 * m + 10 ):               # inner loop: keep mu >= 0
                idx    = np.where( P )[0]
                z      = np.zeros( m )
                z[idx] = np.linalg.lstsq( M[np.ix_( idx, idx )], -s0[idx], rcond = None )[0]
                if np.all( z[idx] > 0 ):
                    mu = z                               # equality-constrained step accepted
                    break
                neg    = idx[ ( z[idx] <= 0 ) & ( mu[idx] - z[idx] > 0 ) ]
                step   = np.min( mu[neg] / ( mu[neg] - z[neg] ) ) if len( neg ) > 0 else 0.0
                mu     = mu + step * ( z - mu )
                P     &= mu > 1e-15                      # drop constraints whose mu hit 0
                mu[~P] = 0.0
                if not P.any():
                    break
        s = s0 + ( M - delta * np.eye( m ) ) @ mu          # slack for the original M
        if np.min( s ) < -1e-8:
            warnings.warn( f"_solve_lcp: constraint violation {-np.min( s ):.2e} after active-set solve." )
        return mu

    def inner_QP_solver( self ):
        """
        Exact fixed-support QP:  min_b  b'H_SS b - c_S'b   s.t.  A_S b >= b_vec.
        With H_SS = L L', u = L'b - d and L d = c_S / 2, the QP becomes the least-distance
        problem  min ||u||  s.t.  G u >= h,  G = A_S L^{-T},  h = b_vec - A_S beta0,
        which is solved exactly through one NNLS problem in the m multipliers
        (Lawson & Hanson, 1974, Ch. 23).  Robust to linearly dependent constraint rows.
        """
        self.S = np.asarray( self.S, dtype = int )
        k      = len( self.S )
        X_S    = self.X_[:, self.S]
        H      = ( 1 - self.alpha_ ) * ( X_S.T @ X_S )
        H[ np.diag_indices( k ) ] += self.lambda_
        c_S    = self.c[self.S]

        # Step 1: Cholesky factor H = L L' (tiny diagonal jitter only if round-off breaks it)
        try:
            L = np.linalg.cholesky( H )
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky( H + 1e-12 * np.trace( H ) / k * np.eye( k ) )

        # Step 2: unconstrained minimizer beta0 = H^{-1} c_S / 2 = L^{-T} d,  L d = c_S / 2
        d      = solve_triangular( L, 0.5 * c_S, lower = True )
        beta_S = solve_triangular( L.T, d, lower = False )

        # Step 3: constraint rows that touch the support
        q        = self._A_dense.shape[1]
        in_A     = self.S < q
        self.mu_ = np.zeros( self._A_dense.shape[0] )
        if self._A_dense.shape[0] > 0 and in_A.any():
            A          = np.zeros( ( self._A_dense.shape[0], k ) )
            A[:, in_A] = self._A_dense[:, self.S[in_A]]
            rows       = np.any( A != 0, axis = 1 )
            if rows.any():
                A_r, b_r = A[rows], self.b[rows]
                s0       = A_r @ beta_S - b_r
                if np.min( s0 ) < 0:                                 # unconstrained optimum infeasible
                    # Step 4: least-distance problem  min ||u||  s.t.  G u >= h
                    G = solve_triangular( L, A_r.T, lower = True ).T  # G = A_r L^{-T}  (m x k)
                    h = -s0
                    # Step 5: LDP via NNLS:  min_{w>=0} || E w - f ||,  E = [G'; h'],  f = e_{k+1}
                    E    = np.vstack( [ G.T, h[None, :] ] )
                    f    = np.zeros( k + 1 ); f[-1] = 1.0
                    w, _ = nnls( E, f, maxiter = 50 * E.shape[1] )
                    res  = E @ w - f
                    if abs( res[-1] ) < 1e-14:
                        raise ValueError( "Fixed-support QP is infeasible (check Assumption on A, b)." )
                    u      = -res[:k] / res[-1]
                    beta_S = beta_S + solve_triangular( L.T, u, lower = False )
                    self.mu_[rows] = 2.0 * w / ( -res[-1] )           # KKT multipliers of the QP

        self.beta_[self.S] = beta_S
        self.r_            = X_S @ beta_S

    def _Addition_score( self, feature_idx ):
        # Works for a scalar index or an index array.
        return ( self.gradients_[feature_idx] ) ** 2 / ( 4 * self.Hjj )

    def _Removal_score( self, feature_idx ):
        # Works for a scalar index or an index array.
        return ( self.beta_[feature_idx] ) ** 2 * self.Hjj

    def outer_support_update( self ):
        """
        Propose a swap of up to swap_num optional features (vectorized scores).
          add score    (j not in S):  g_j^2 / (4 H_jj)   -- predicted decrease if j enters
          removal score (j in U)   :  beta_j^2 H_jj      -- predicted increase if j leaves
        """
        self._compute_gradient()                         # one O(np) product per outer iteration
        p = len( self.beta_ )
        add_score    = np.full( p, -np.inf )
        remove_score = np.full( p,  np.inf )

        cand = np.ones( p, dtype = bool )
        cand[ self.C_mandatory ] = False
        cand[ self.U ]           = False
        add_score[cand] = self._Addition_score( cand )
        add_sequence    = np.argsort( add_score )[::-1].tolist()

        if len( self.U ) > 0:
            U_arr = np.asarray( self.U, dtype = int )
            remove_score[U_arr] = self._Removal_score( U_arr )
        remove_sequence = np.argsort( remove_score ).tolist()

        # Determine which swap to perform.
        # Without group constraints: swap_num-for-swap_num swap.
        # With group constraints: removes are committed best-first, occupied groups are
        # recomputed from the remaining U, then adds are picked greedily so each new feature
        # comes from a still-unoccupied group. The swap size m falls back from swap_num down to 0
        # until a feasible (m removes, m adds) pair exists.
        if self.group_sparsity is None:
            adds_chosen    = add_sequence[:self.swap_num]
            removes_chosen = remove_sequence[:self.swap_num]
        else:
            add_candidates = [ j for j in add_sequence if np.isfinite( add_score[j] ) ]
            adds_chosen, removes_chosen = [], []
            for m in range( self.swap_num, -1, -1 ):
                removes_trial = remove_sequence[:m]
                rm_set        = set( removes_trial )
                remaining_U   = [ j for j in self.U if j not in rm_set ]
                occupied      = self._occupied_groups_from_features( remaining_U )
                try:
                    adds_trial = self._select_top_optional(
                        add_candidates, n_optional = m, occupied_groups = occupied,
                    )
                except ValueError:
                    continue   # try smaller m
                adds_chosen, removes_chosen = adds_trial, removes_trial
                break

        # Accept the swap iff it strictly improves the score sum (apples-to-apples length).
        if len( adds_chosen ) > 0 and np.sum( add_score[adds_chosen] ) > np.sum( remove_score[removes_chosen] ):
            for remove_idx in removes_chosen:
                self.U.remove( remove_idx )
                self.beta_[remove_idx] = 0
            for add_idx in adds_chosen:
                self.U.append( add_idx )
        self.U.sort()

    def _compute_obj( self ):
        # β^T H β = (1-α)||Xβ||² + λ||β||² = (1-α)||r_||² + λ||β||²
        # Uses maintained residual r_ instead of the O(p²) matrix product.
        obj = ( ( 1 - self.alpha_ ) * np.dot( self.r_, self.r_ )
              + self.lambda_ * np.dot( self.beta_, self.beta_ )
              - np.dot( self.c, self.beta_ ) )
        return obj
    
    def fit_model(self, X: np.ndarray, y: np.ndarray, initial_sol: np.ndarray):
        # Validate group_sparsity indices against the actual feature count.
        if self.group_sparsity is not None:
            p = X.shape[1]
            for g_idx, group in enumerate( self.group_sparsity ):
                for j in group:
                    if j < 0 or j >= p:
                        raise ValueError(
                            f"group_sparsity[{g_idx}] contains index {j}, which is out of range [0, {p})."
                        )
        
        # Initialization
        self.iter_ = 0
        self.beta_ = initial_sol.copy()
        self.S = np.where(np.array(self.beta_) != 0)[0].tolist()
        self.C = list(self.C_mandatory)                                          # mandatory features
        self.U = [j for j in self.S if j not in self._C_mandatory_set]          # optional features in support
        self.beta_[self.U] = 0

        if len(self.S) != self.k:
            raise ValueError(f"Initial beta must have exactly k={self.k} nonzeros; got {len(self.S)}")

        # Verify the initial support already obeys group_sparsity.
        if self.group_sparsity is not None:
            seen_groups = set()
            for j in self.U:
                groups_of_j = self._feature_to_groups.get( j, set() )
                if groups_of_j & seen_groups:
                    raise ValueError(
                        f"Initial support violates group_sparsity at feature {j}: "
                        f"groups {sorted( groups_of_j & seen_groups )} are already represented."
                    )
                seen_groups |= groups_of_j

        start_time = time.time()

        # Implicit Hessian: never form the O(p²) matrix H = (1-α)X^TX + λI.
        # Instead store X itself and use Hessian-vector products on the fly:
        #   ∇f(β) = 2(1-α)X^T(Xβ) + 2λβ - c   [O(np) per call]
        #   β^T H β = (1-α)||r||² + λ||β||²     [O(n+p) using residual r=Xβ]
        # Because X columns are unit-norm, H_jj = (1-α) + λ  (scalar constant).
        start_ = time.time()
        self.X_  = X                                                  # store reference (no copy)
        self.c   = ( 2 - self.alpha_ ) * ( X.T @ y )                 # O(np)
        self.Hjj = ( 1 - self.alpha_ ) + self.lambda_                 # scalar, valid for all j
        self.S    = np.array( self.S )
        self.r_  = self.X_[:, self.S] @ self.beta_[self.S]             # residual r = Xβ, O(nk)
        end_ = time.time()

        self._log( f'Implicit Hessian (X_, c, Hjj, r_) computed! (Computational time: {(end_ - start_):.4f}s)' )
        self._log( "-" * 80 )

        # Logging header
        self._log( f"{'Iter':>6}  {'Obj':>18}  {'Delta Obj':>14}" )
        self._log( "-" * 80 )

        # Solve fixed-support QP once for the initial support
        self.inner_QP_solver()
        prev_obj = self._compute_obj()
        self._log( f"{'Init':>6}  {prev_obj:>18.8f}  {'—':>14}" )
        
        # Outer support update loop
        while True:
            self.iter_ += 1
            
            # Save current state (for possible rollback)
            old_beta = self.beta_.copy()
            old_S = self.S.copy()
            old_U = self.U.copy()
            old_obj = prev_obj
            
            # --- Propose a support update (swap) ---
            self.outer_support_update()
            self.S = self.C + self.U
            self._sync_residual()         # r_ = X_S beta_S for the proposed support

            # --- Re-solve inner QP for the proposed support ---
            self.inner_QP_solver()
            new_obj = self._compute_obj()
            
            delta_obj = new_obj - old_obj
            self._log( f"{self.iter_:>6}  {new_obj:>18.8f}  {delta_obj:>+14.2e}" )

            # (A) Reject non-improving (or numerically equal) updates to prevent cycling
            # You can tune this tolerance if needed.
            improvement_tol = 1e-12
            if new_obj >= old_obj - improvement_tol:
                # Roll back and terminate
                self.beta_ = old_beta
                self.S = old_S
                self.U = old_U
                self._sync_residual()        # restore r_ to match rolled-back support
                prev_obj = old_obj
                self._log( f"{'':>6}  Terminated: no improvement (delta >= -{improvement_tol:.0e}). Rolled back." )
                break

            # (B) Maximum iteration safeguard
            if self.iter_ >= self.maxIters:
                # Optional: keep the best found so far (already in beta_)
                self._log( f"{'':>6}  Terminated: reached maxIters={self.maxIters}." )
                break

            # (C) Objective convergence check (same as your original logic)
            if abs(new_obj - old_obj) / (1 + abs(old_obj)) < self.converge_tol:
                prev_obj = new_obj
                self._log( f"{'':>6}  Terminated: objective converged (rel. change < {self.converge_tol:.0e})." )
                break

            # Accept update and continue
            prev_obj = new_obj
        
        end_time = time.time()
        self.time_ = end_time - start_time

        self._log( "=" * 80 )
        self._log( f"COSMOS Solved  |  Iters: {self.iter_}  |  Obj: {self._compute_obj():.8f}  |  Time: {self.time_:.4f}s" )
        self._log( f"Final support  : {sorted( self.S )}" )
        self._log( "=" * 80 )

    def predict( self, X: np.ndarray ):
        return np.dot( X, self.beta_ )

class L0Learn_Model:
    """
    Wrapper around the R package L0Learn (via rpy2) that conforms to the
    fit_model / predict interface used throughout this codebase.

    Fits at a SINGLE (lambda, gamma) point specified by the caller.
    The exact lambda is passed via lambdaGrid; for L0L1/L0L2 penalties,
    gamma is pinned by setting nGamma=1 with gammaMax=gammaMin=gamma_val.
    No intercept is fitted (intercept=False).

    Parameters
    ----------
    penalty    : "L0", "L0L1", or "L0L2".
    algorithm  : "CD" or "CDPSI".
    lambda_val : exact lambda regularization value.
    gamma_val  : exact gamma regularization value (ignored for "L0").
    """

    def __init__( self, penalty: str = "L0", algorithm: str = "CD",
                  lambda_val: float = 0.01, gamma_val: float = 0.0 ):
        self.penalty    = penalty
        self.algorithm  = algorithm
        self.lambda_val = lambda_val
        self.gamma_val  = gamma_val
        self.param_ = {
            'penalty':   penalty,
            'algorithm': algorithm,
            'lambda':    lambda_val,
            'gamma':     gamma_val
        }

    def fit_model( self, X: np.ndarray, y: np.ndarray ):
        """
        Fit L0Learn at the single (lambda, gamma) point and store coefficients.

        lambdaGrid pins the exact lambda value.  For L0L1/L0L2, nGamma=1 with
        gammaMax=gammaMin=gamma_val pins the exact gamma.  The fitted coefficient
        vector is stored in self.beta_ and the actual support size (number of
        non-zeros selected by the model) in self.supp_size_.
        """
        X = np.asarray( X, dtype = np.float64 )
        y = np.asarray( y, dtype = np.float64 ).ravel()
        n, p = X.shape

        r_X           = ro.r.matrix( ro.FloatVector( X.flatten( order = 'F' ) ), nrow = n, ncol = p )  # R fills matrices column-major
        r_y           = ro.FloatVector( y )
        r_lambda_grid = ro.r['list']( ro.FloatVector( [self.lambda_val] ) )

        fit_kwargs = dict(
            penalty    = self.penalty,
            algorithm  = self.algorithm,
            lambdaGrid = r_lambda_grid,
            loss       = "SquaredError",
            intercept  = False
        )
        if self.penalty in ( "L0L1", "L0L2" ):
            fit_kwargs['nGamma']   = 1
            fit_kwargs['gammaMax'] = self.gamma_val
            fit_kwargs['gammaMin'] = self.gamma_val

        fit = _L0Learn_R.L0Learn_fit( r_X, r_y, **fit_kwargs )

        beta_mat     = np.array( ro.r["as.matrix"]( fit.rx2( "beta" )[0] ) )  # (p, 1)
        suppsize_vec = np.array( fit.rx2( "suppSize" )[0] )                    # (1,)

        self.beta_      = beta_mat[ :, 0 ].copy()
        self.supp_size_ = int( suppsize_vec[0] )

    def predict( self, X: np.ndarray ) -> np.ndarray:
        """Predict using the fitted coefficient vector."""
        return np.dot( X, self.beta_ )