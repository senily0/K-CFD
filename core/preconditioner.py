"""
전처리기(Preconditioner) 모듈.

Jacobi, ILU(0), ILU(k), AMG 전처리기를 scipy LinearOperator로 제공.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, spilu

try:
    import pyamg
    HAS_PYAMG = True
except ImportError:
    HAS_PYAMG = False


def create_preconditioner(A_csr, method='none', **kwargs):
    """
    전처리기 생성.

    Parameters
    ----------
    A_csr : scipy.sparse.csr_matrix
    method : 'none', 'jacobi', 'ilu0', 'iluk', 'amg'
    kwargs : fill_factor (for iluk), max_levels/max_coarse (for amg)

    Returns
    -------
    M : LinearOperator or None
        None if method='none'
    info : dict
        {'method': str, 'setup_time': float}
    """
    import time
    t0 = time.time()

    n = A_csr.shape[0]

    if method == 'none':
        return None, {'method': 'none', 'setup_time': 0.0}

    if method == 'jacobi':
        diag = A_csr.diagonal()
        diag_inv = np.where(np.abs(diag) > 1e-30, 1.0 / diag, 0.0)
        M = LinearOperator((n, n), matvec=lambda x: diag_inv * x)
        return M, {'method': 'jacobi', 'setup_time': time.time() - t0}

    if method == 'ilu0':
        # ILU(0): fill_factor=1 means no additional fill
        A_csc = A_csr.tocsc()
        ilu = spilu(A_csc, fill_factor=1, drop_tol=0)
        M = LinearOperator((n, n), matvec=ilu.solve)
        return M, {'method': 'ilu0', 'setup_time': time.time() - t0}

    if method == 'iluk':
        fill_factor = kwargs.get('fill_factor', 3)
        A_csc = A_csr.tocsc()
        ilu = spilu(A_csc, fill_factor=fill_factor)
        M = LinearOperator((n, n), matvec=ilu.solve)
        return M, {'method': f'iluk(fill={fill_factor})', 'setup_time': time.time() - t0}

    if method == 'amg':
        if not HAS_PYAMG:
            print("  [Preconditioner] pyamg 미설치. ILU(0)으로 폴백합니다.")
            return create_preconditioner(A_csr, method='ilu0')
        max_levels = kwargs.get('max_levels', 10)
        max_coarse = kwargs.get('max_coarse', 500)
        ml = pyamg.smoothed_aggregation_solver(
            A_csr, max_levels=max_levels, max_coarse=max_coarse
        )
        M = ml.aspreconditioner()
        return M, {'method': 'amg', 'setup_time': time.time() - t0,
                   'levels': len(ml.levels), 'complexity': ml.operator_complexity()}

    raise ValueError(f"Unknown preconditioner: {method}")
