"""
GPU 가속 선형 솔버.

CuPy 기반 BiCGSTAB 및 SpMV.
CuPy 미설치 시 CPU 폴백.
"""

import numpy as np
from scipy import sparse
import time

try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

try:
    import pyopencl as cl
    HAS_OPENCL = True
except ImportError:
    HAS_OPENCL = False


def detect_gpu_backend() -> str:
    """
    사용 가능한 GPU 백엔드 감지.

    Returns
    -------
    backend : 'cupy', 'opencl', 'cpu'
    """
    if HAS_CUPY:
        try:
            cp.cuda.Device(0).compute_capability
            return 'cupy'
        except Exception:
            pass

    if HAS_OPENCL:
        try:
            platforms = cl.get_platforms()
            if platforms:
                return 'opencl'
        except Exception:
            pass

    print("  [GPU] CuPy/PyOpenCL 미설치. GPU 가속을 사용하려면:")
    print("        pip install cupy-cuda12x  (NVIDIA CUDA 12.x)")
    print("        또는 pip install cupy-cuda11x  (NVIDIA CUDA 11.x)")
    return 'cpu'


def gpu_spmv(A_csr: sparse.csr_matrix, x: np.ndarray) -> np.ndarray:
    """
    GPU SpMV (희소 행렬-벡터 곱).

    CuPy 사용 가능하면 GPU, 아니면 CPU.

    Parameters
    ----------
    A_csr : CSR 희소 행렬
    x : 벡터

    Returns
    -------
    y : A @ x
    """
    if HAS_CUPY:
        try:
            A_gpu = cp_sparse.csr_matrix(A_csr)
            x_gpu = cp.array(x)
            y_gpu = A_gpu @ x_gpu
            return cp.asnumpy(y_gpu)
        except Exception:
            pass

    return A_csr @ x


def gpu_bicgstab(A_csr: sparse.csr_matrix, b: np.ndarray,
                  x0: np.ndarray = None,
                  tol: float = 1e-6, max_iter: int = 1000) -> np.ndarray:
    """
    GPU BiCGSTAB 솔버.

    CuPy 사용 가능하면 GPU에서, 아니면 CPU에서 실행.

    Parameters
    ----------
    A_csr : CSR 희소 행렬
    b : 우변 벡터
    x0 : 초기 추정 (None이면 영벡터)
    tol : 수렴 허용오차
    max_iter : 최대 반복

    Returns
    -------
    x : 해 벡터
    """
    if HAS_CUPY:
        try:
            return _gpu_bicgstab_cupy(A_csr, b, x0, tol, max_iter)
        except Exception:
            pass

    return _cpu_bicgstab(A_csr, b, x0, tol, max_iter)


def _gpu_bicgstab_cupy(A_csr, b, x0, tol, max_iter):
    """CuPy GPU BiCGSTAB."""
    A_gpu = cp_sparse.csr_matrix(A_csr)
    b_gpu = cp.array(b)
    x_gpu = cp.array(x0) if x0 is not None else cp.zeros_like(b_gpu)

    r = b_gpu - A_gpu @ x_gpu
    r_hat = r.copy()
    rho = alpha = omega = 1.0
    v = cp.zeros_like(b_gpu)
    p = cp.zeros_like(b_gpu)

    b_norm = float(cp.linalg.norm(b_gpu))
    if b_norm < 1e-30:
        return cp.asnumpy(x_gpu)

    for it in range(max_iter):
        rho_new = float(cp.dot(r_hat, r))
        if abs(rho_new) < 1e-30:
            break

        beta = (rho_new / rho) * (alpha / omega) if abs(rho * omega) > 1e-30 else 0.0
        p = r + beta * (p - omega * v)
        v = A_gpu @ p

        denom = float(cp.dot(r_hat, v))
        alpha = rho_new / denom if abs(denom) > 1e-30 else 0.0
        s = r - alpha * v

        if float(cp.linalg.norm(s)) / b_norm < tol:
            x_gpu += alpha * p
            break

        t = A_gpu @ s
        t_dot_t = float(cp.dot(t, t))
        omega = float(cp.dot(t, s)) / t_dot_t if t_dot_t > 1e-30 else 0.0

        x_gpu += alpha * p + omega * s
        r = s - omega * t
        rho = rho_new

        if float(cp.linalg.norm(r)) / b_norm < tol:
            break

    return cp.asnumpy(x_gpu)


def _cpu_bicgstab(A_csr, b, x0, tol, max_iter):
    """CPU BiCGSTAB (폴백)."""
    x = x0.copy() if x0 is not None else np.zeros_like(b)
    r = b - A_csr @ x
    r_hat = r.copy()
    rho = alpha = omega = 1.0
    v = np.zeros_like(b)
    p = np.zeros_like(b)

    b_norm = np.linalg.norm(b)
    if b_norm < 1e-30:
        return x

    for it in range(max_iter):
        rho_new = np.dot(r_hat, r)
        if abs(rho_new) < 1e-30:
            break

        beta = (rho_new / rho) * (alpha / omega) if abs(rho * omega) > 1e-30 else 0.0
        p = r + beta * (p - omega * v)
        v = A_csr @ p

        denom = np.dot(r_hat, v)
        alpha = rho_new / denom if abs(denom) > 1e-30 else 0.0
        s = r - alpha * v

        if np.linalg.norm(s) / b_norm < tol:
            x += alpha * p
            break

        t = A_csr @ s
        t_dot_t = np.dot(t, t)
        omega = np.dot(t, s) / t_dot_t if t_dot_t > 1e-30 else 0.0

        x += alpha * p + omega * s
        r = s - omega * t
        rho = rho_new

        if np.linalg.norm(r) / b_norm < tol:
            break

    return x


def benchmark_solvers(A_csr: sparse.csr_matrix, b: np.ndarray,
                       n_runs: int = 3) -> dict:
    """
    CPU direct vs CPU BiCGSTAB vs GPU BiCGSTAB 벤치마크.

    Returns
    -------
    results : {'cpu_direct': float, 'cpu_bicgstab': float,
               'gpu_bicgstab': float, 'backend': str}
    """
    from scipy.sparse.linalg import spsolve

    results = {'backend': detect_gpu_backend()}

    # CPU direct
    times = []
    for _ in range(n_runs):
        t0 = time.time()
        x_direct = spsolve(A_csr, b)
        times.append(time.time() - t0)
    results['cpu_direct'] = np.median(times)
    results['x_direct'] = x_direct

    # CPU BiCGSTAB
    times = []
    for _ in range(n_runs):
        t0 = time.time()
        x_bicg = _cpu_bicgstab(A_csr, b, None, 1e-6, 1000)
        times.append(time.time() - t0)
    results['cpu_bicgstab'] = np.median(times)

    # GPU BiCGSTAB
    if HAS_CUPY:
        try:
            # 워밍업
            _gpu_bicgstab_cupy(A_csr, b, None, 1e-6, 1000)

            times = []
            for _ in range(n_runs):
                t0 = time.time()
                x_gpu = _gpu_bicgstab_cupy(A_csr, b, None, 1e-6, 1000)
                times.append(time.time() - t0)
            results['gpu_bicgstab'] = np.median(times)
        except Exception:
            results['gpu_bicgstab'] = float('nan')
    else:
        results['gpu_bicgstab'] = float('nan')

    return results
