#pragma once

#include <string>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "twofluid/fvm_operators.hpp"

namespace twofluid {

/// Solve the linear system Ax = b stored in an FVMSystem.
/// @param system  assembled FVM system (COO data + RHS)
/// @param x0      initial guess (ignored for direct solver)
/// @param method  "direct" (SparseLU) or "bicgstab"
/// @param tol     convergence tolerance for iterative solver
/// @param maxiter maximum iterations for iterative solver
/// @return solution vector x
Eigen::VectorXd solve_linear_system(
    const FVMSystem& system,
    const Eigen::VectorXd& x0 = Eigen::VectorXd(),
    const std::string& method = "direct",
    double tol = 1e-6,
    int maxiter = 1000
);

} // namespace twofluid
