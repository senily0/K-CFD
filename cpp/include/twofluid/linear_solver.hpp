#pragma once

#include <functional>
#include <string>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "twofluid/fvm_operators.hpp"

namespace twofluid {

/// Solve the linear system Ax = b stored in an FVMSystem.
/// @param system  assembled FVM system (COO data + RHS)
/// @param x0      initial guess (ignored for direct solver)
/// @param method  "direct" (SparseLU), "bicgstab", or "cg"
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

/// Solve with an explicit preconditioner function.
/// @param system   assembled FVM system
/// @param x0       initial guess
/// @param method   "bicgstab" or "cg"
/// @param precond  M^{-1} apply function; nullptr means no preconditioning
/// @param tol      convergence tolerance
/// @param maxiter  maximum iterations
/// @return solution vector x
Eigen::VectorXd solve_preconditioned(
    const FVMSystem& system,
    const Eigen::VectorXd& x0,
    const std::string& method,
    const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& precond,
    double tol = 1e-6,
    int maxiter = 1000
);

} // namespace twofluid
