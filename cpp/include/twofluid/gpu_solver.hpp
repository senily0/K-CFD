#pragma once

#include <string>
#include <vector>

namespace twofluid {

struct GPUSolveResult {
    bool converged = false;
    int iterations = 0;
    double residual = 1.0;
    double gpu_time_ms = 0.0;
    std::vector<double> x;
};

/// CUDA BiCGSTAB using cuSPARSE (CSR format input)
/// @param n       number of rows/columns
/// @param row_ptr CSR row pointers (size n+1)
/// @param col_idx CSR column indices (size nnz)
/// @param vals    CSR values (size nnz)
/// @param rhs     right-hand side vector (size n)
/// @param x0      initial guess (size n)
/// @param nnz     number of non-zeros
/// @param tol     convergence tolerance
/// @param maxiter maximum iterations
/// @return GPUSolveResult with solution and diagnostics
GPUSolveResult gpu_bicgstab(
    const int n,
    const int* row_ptr, const int* col_idx, const double* vals,
    const double* rhs, const double* x0,
    int nnz, double tol = 1e-6, int maxiter = 1000);

/// Check if a CUDA GPU is available
bool detect_gpu();

/// Get GPU device name
std::string gpu_name();

} // namespace twofluid
