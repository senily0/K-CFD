#include "twofluid/linear_solver.hpp"

#include <cmath>
#include <functional>
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

    } else if (method == "cg") {
        Eigen::ConjugateGradient<Eigen::SparseMatrix<double>,
                                 Eigen::Lower | Eigen::Upper> solver;
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
            ". Available: direct, bicgstab, cg");
    }
}

// ---------------------------------------------------------------------------
// Preconditioned BiCGSTAB (manual implementation)
// ---------------------------------------------------------------------------
static Eigen::VectorXd precond_bicgstab(
    const Eigen::SparseMatrix<double>& A,
    const Eigen::VectorXd& b,
    const Eigen::VectorXd& x0,
    const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& M_inv,
    double tol, int maxiter)
{
    const int n = static_cast<int>(b.size());
    Eigen::VectorXd x = (x0.size() == n) ? x0 : Eigen::VectorXd::Zero(n);

    Eigen::VectorXd r = b - A * x;
    Eigen::VectorXd r_hat = r;  // shadow residual (fixed)

    double rho_old = 1.0, alpha = 1.0, omega = 1.0;
    Eigen::VectorXd v = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd p = Eigen::VectorXd::Zero(n);

    double b_norm = b.norm();
    if (b_norm < 1e-30) return x;

    for (int iter = 0; iter < maxiter; ++iter) {
        double rho_new = r_hat.dot(r);
        if (std::abs(rho_new) < 1e-300) break;

        if (iter == 0) {
            p = r;
        } else {
            double beta = (rho_new / rho_old) * (alpha / omega);
            p = r + beta * (p - omega * v);
        }

        Eigen::VectorXd p_hat = M_inv ? M_inv(p) : p;
        v = A * p_hat;

        double denom = r_hat.dot(v);
        if (std::abs(denom) < 1e-300) break;
        alpha = rho_new / denom;

        Eigen::VectorXd s = r - alpha * v;
        if (s.norm() < tol * b_norm) {
            x += alpha * p_hat;
            break;
        }

        Eigen::VectorXd s_hat = M_inv ? M_inv(s) : s;
        Eigen::VectorXd t = A * s_hat;

        double t_dot_t = t.dot(t);
        omega = (t_dot_t > 1e-300) ? t.dot(s) / t_dot_t : 0.0;

        x += alpha * p_hat + omega * s_hat;
        r = s - omega * t;

        if (r.norm() < tol * b_norm) break;

        rho_old = rho_new;
    }
    return x;
}

// Preconditioned CG
static Eigen::VectorXd precond_cg(
    const Eigen::SparseMatrix<double>& A,
    const Eigen::VectorXd& b,
    const Eigen::VectorXd& x0,
    const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& M_inv,
    double tol, int maxiter)
{
    const int n = static_cast<int>(b.size());
    Eigen::VectorXd x = (x0.size() == n) ? x0 : Eigen::VectorXd::Zero(n);

    Eigen::VectorXd r = b - A * x;
    Eigen::VectorXd z = M_inv ? M_inv(r) : r;
    Eigen::VectorXd p = z;
    double rz_old = r.dot(z);

    double b_norm = b.norm();
    if (b_norm < 1e-30) return x;

    for (int iter = 0; iter < maxiter; ++iter) {
        Eigen::VectorXd Ap = A * p;
        double pAp = p.dot(Ap);
        if (std::abs(pAp) < 1e-300) break;
        double alpha = rz_old / pAp;

        x += alpha * p;
        r -= alpha * Ap;

        if (r.norm() < tol * b_norm) break;

        z = M_inv ? M_inv(r) : r;
        double rz_new = r.dot(z);
        double beta = rz_new / rz_old;
        p = z + beta * p;
        rz_old = rz_new;
    }
    return x;
}

Eigen::VectorXd solve_preconditioned(
    const FVMSystem& system,
    const Eigen::VectorXd& x0,
    const std::string& method,
    const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& precond,
    double tol,
    int maxiter)
{
    Eigen::SparseMatrix<double> A = system.to_sparse();
    A.makeCompressed();
    const Eigen::VectorXd& b = system.rhs;

    if (method == "bicgstab") {
        return precond_bicgstab(A, b, x0, precond, tol, maxiter);
    } else if (method == "cg") {
        return precond_cg(A, b, x0, precond, tol, maxiter);
    } else {
        throw std::invalid_argument(
            "solve_preconditioned: unknown method '" + method +
            "'. Available: bicgstab, cg");
    }
}

} // namespace twofluid
