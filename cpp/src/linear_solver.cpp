#include "twofluid/linear_solver.hpp"

#include <stdexcept>
#include <Eigen/SparseLU>
#include <Eigen/IterativeLinearSolvers>

namespace twofluid {

Eigen::VectorXd solve_linear_system(
    const FVMSystem& system,
    const Eigen::VectorXd& x0,
    const std::string& method,
    double tol,
    int maxiter) {

    Eigen::SparseMatrix<double> A = system.to_sparse();
    A.makeCompressed();
    const Eigen::VectorXd& b = system.rhs;

    if (method == "direct") {
        Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
        solver.analyzePattern(A);
        solver.factorize(A);
        if (solver.info() != Eigen::Success) {
            // Fallback: return zero vector on factorization failure
            return Eigen::VectorXd::Zero(system.n);
        }
        return solver.solve(b);

    } else if (method == "bicgstab") {
        Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> solver;
        solver.setTolerance(tol);
        solver.setMaxIterations(maxiter);
        solver.compute(A);
        if (solver.info() != Eigen::Success) {
            return Eigen::VectorXd::Zero(system.n);
        }
        if (x0.size() == system.n) {
            return solver.solveWithGuess(b, x0);
        }
        return solver.solve(b);

    } else {
        throw std::invalid_argument(
            "Unknown solver method: " + method +
            ". Available: direct, bicgstab");
    }
}

} // namespace twofluid
