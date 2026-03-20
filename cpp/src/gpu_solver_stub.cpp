#ifndef USE_CUDA

#include "twofluid/gpu_solver.hpp"

namespace twofluid {

GPUSolveResult gpu_bicgstab(
    const int /*n*/,
    const int* /*row_ptr*/, const int* /*col_idx*/, const double* /*vals*/,
    const double* /*rhs*/, const double* /*x0*/,
    int /*nnz*/, double /*tol*/, int /*maxiter*/) {
    GPUSolveResult result;
    result.converged = false;
    result.iterations = 0;
    result.residual = 1.0;
    result.gpu_time_ms = 0.0;
    return result;
}

bool detect_gpu() { return false; }

std::string gpu_name() { return "none (CUDA not compiled)"; }

} // namespace twofluid

#endif
