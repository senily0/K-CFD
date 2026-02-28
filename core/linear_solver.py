"""
선형 시스템 솔버 래퍼.

scipy.sparse.linalg를 사용한 반복 솔버 인터페이스.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, bicgstab, gmres
from core.fvm_operators import FVMSystem
from core.preconditioner import create_preconditioner


def solve_linear_system(system: FVMSystem, x0: np.ndarray = None,
                        method: str = 'direct',
                        tol: float = 1e-8, maxiter: int = 1000,
                        backend: str = 'auto',
                        preconditioner: str = 'none',
                        **precond_kwargs) -> np.ndarray:
    """
    선형 시스템 Ax = b 풀기.

    Parameters
    ----------
    system : FVMSystem (행렬 + 우변)
    x0 : 초기 추정값 (반복법 사용 시)
    method : 'direct', 'bicgstab', 'gmres'
    tol : 수렴 허용오차
    maxiter : 최대 반복 횟수
    backend : 'auto', 'cpu', 'gpu' (gpu 사용 시 CuPy BiCGSTAB)
    preconditioner : 'none', 'jacobi', 'ilu0', 'iluk', 'amg'
    precond_kwargs : fill_factor (iluk), max_levels/max_coarse (amg)

    Returns
    -------
    x : 해 벡터
    """
    A = system.to_sparse()
    b = system.rhs

    # GPU 백엔드 시도 (전처리기 미적용)
    if backend == 'gpu':
        try:
            from core.gpu_solver import gpu_bicgstab, detect_gpu_backend
            if detect_gpu_backend() == 'cupy':
                return gpu_bicgstab(A, b, x0, tol, maxiter)
        except (ImportError, Exception):
            pass

    if method == 'direct':
        return spsolve(A, b)

    if x0 is None:
        x0 = np.zeros(system.n)

    M, _ = create_preconditioner(A, preconditioner, **precond_kwargs)

    if method == 'bicgstab':
        x, info = bicgstab(A, b, x0=x0, atol=tol, maxiter=maxiter, M=M)
        if info != 0 and info > 0:
            print(f"[bicgstab] 수렴 실패: {info} 반복 후 중단")
        return x

    if method == 'gmres':
        x, info = gmres(A, b, x0=x0, atol=tol, maxiter=maxiter, M=M)
        if info != 0 and info > 0:
            print(f"[gmres] 수렴 실패: {info} 반복 후 중단")
        return x

    raise ValueError(f"Unknown solver method: {method}")


def solve_pressure_correction(system: FVMSystem, method: str = 'direct',
                              tol: float = 1e-6, maxiter: int = 2000,
                              preconditioner: str = 'none',
                              **precond_kwargs) -> np.ndarray:
    """
    압력 보정 방정식 전용 솔버.

    압력 보정 방정식은 대칭이므로 직접법 또는 CG 사용 가능.
    """
    A = system.to_sparse()
    b = system.rhs

    if method == 'direct':
        return spsolve(A, b)

    M, _ = create_preconditioner(A, preconditioner, **precond_kwargs)

    from scipy.sparse.linalg import cg
    x0 = np.zeros(system.n)
    x, info = cg(A, b, x0=x0, atol=tol, maxiter=maxiter, M=M)
    return x
