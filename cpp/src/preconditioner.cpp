#include "twofluid/preconditioner.hpp"
#include <chrono>
#include <cmath>
#include <memory>
#include <Eigen/IterativeLinearSolvers>

namespace twofluid {

std::pair<std::function<Eigen::VectorXd(const Eigen::VectorXd&)>, PreconditionerInfo>
create_preconditioner(const Eigen::SparseMatrix<double>& A,
                      const std::string& method) {
    auto t0 = std::chrono::high_resolution_clock::now();

    if (method == "none") {
        return {nullptr, {"none", 0.0}};
    }

    if (method == "jacobi") {
        int n = static_cast<int>(A.rows());
        Eigen::VectorXd diag_inv(n);
        for (int i = 0; i < n; ++i) {
            double d = A.coeff(i, i);
            diag_inv(i) = (std::abs(d) > 1e-30) ? 1.0 / d : 0.0;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();

        auto apply = [diag_inv](const Eigen::VectorXd& x) -> Eigen::VectorXd {
            return diag_inv.cwiseProduct(x);
        };
        return {apply, {"jacobi", elapsed}};
    }

    if (method == "ilu0") {
        auto ilu = std::make_shared<
            Eigen::IncompleteLUT<double>>();
        ilu->setFillfactor(1);
        ilu->setDroptol(0.0);
        ilu->compute(A);

        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();

        if (ilu->info() != Eigen::Success) {
            return {nullptr, {"ilu0_failed", elapsed}};
        }

        auto apply = [ilu](const Eigen::VectorXd& x) -> Eigen::VectorXd {
            return ilu->solve(x);
        };
        return {apply, {"ilu0", elapsed}};
    }

    if (method == "amg") {
        auto amg = std::make_shared<AMGPreconditioner>(A);
        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        auto apply = [amg](const Eigen::VectorXd& x) -> Eigen::VectorXd {
            return amg->apply(x);
        };
        return {apply, {"amg", elapsed}};
    }

    return {nullptr, {"unknown", 0.0}};
}

// ---------------------------------------------------------------------------
// AMGPreconditioner implementation
// ---------------------------------------------------------------------------

AMGPreconditioner::AMGPreconditioner(const Eigen::SparseMatrix<double>& A,
                                     int max_levels, int max_coarse,
                                     int pre_smooth, int post_smooth)
    : pre_smooth_(pre_smooth), post_smooth_(post_smooth)
{
    build_hierarchy(A, max_levels, max_coarse);
}

void AMGPreconditioner::build_hierarchy(const Eigen::SparseMatrix<double>& A,
                                        int max_levels, int max_coarse)
{
    Eigen::SparseMatrix<double> Ac = A;
    Ac.makeCompressed();

    while (static_cast<int>(levels_.size()) < max_levels &&
           static_cast<int>(Ac.rows()) > max_coarse)
    {
        Level lvl;
        lvl.A = Ac;
        lvl.n = static_cast<int>(Ac.rows());

        // Build inverse diagonal for Gauss-Seidel
        lvl.diag_inv.resize(lvl.n);
        for (int i = 0; i < lvl.n; ++i) {
            double d = Ac.coeff(i, i);
            lvl.diag_inv(i) = (std::abs(d) > 1e-30) ? 1.0 / d : 0.0;
        }

        // Pairwise aggregation coarsening
        // For each row find the column with the largest |off-diagonal| entry
        std::vector<int> aggregate(lvl.n, -1);
        int n_coarse = 0;
        for (int i = 0; i < lvl.n; ++i) {
            if (aggregate[i] >= 0) continue;

            // Find strongest connection
            int jbest = -1;
            double vbest = 0.0;
            for (Eigen::SparseMatrix<double>::InnerIterator it(Ac, i); it; ++it) {
                int j = static_cast<int>(it.col());
                if (j == i) continue;
                double v = std::abs(it.value());
                if (aggregate[j] < 0 && v > vbest) {
                    vbest = v;
                    jbest = j;
                }
            }

            aggregate[i] = n_coarse;
            if (jbest >= 0) {
                aggregate[jbest] = n_coarse;
            }
            ++n_coarse;
        }

        if (n_coarse >= lvl.n) {
            // No coarsening achieved — stop
            levels_.push_back(std::move(lvl));
            break;
        }

        // Build prolongation P: fine(lvl.n) x coarse(n_coarse)
        std::vector<Eigen::Triplet<double>> Ptrips;
        Ptrips.reserve(lvl.n);
        for (int i = 0; i < lvl.n; ++i) {
            Ptrips.emplace_back(i, aggregate[i], 1.0);
        }
        lvl.P.resize(lvl.n, n_coarse);
        lvl.P.setFromTriplets(Ptrips.begin(), Ptrips.end());

        // Restriction R = P^T
        lvl.R = lvl.P.transpose();

        // Galerkin coarse operator: A_c = R * A * P
        Ac = lvl.R * lvl.A * lvl.P;
        Ac.makeCompressed();

        levels_.push_back(std::move(lvl));
    }

    // Add coarsest level (no P/R needed — solved directly)
    Level coarsest;
    coarsest.A = Ac;
    coarsest.n = static_cast<int>(Ac.rows());
    coarsest.diag_inv.resize(coarsest.n);
    for (int i = 0; i < coarsest.n; ++i) {
        double d = Ac.coeff(i, i);
        coarsest.diag_inv(i) = (std::abs(d) > 1e-30) ? 1.0 / d : 0.0;
    }
    levels_.push_back(std::move(coarsest));
}

void AMGPreconditioner::gauss_seidel(const Level& lvl, Eigen::VectorXd& x,
                                     const Eigen::VectorXd& b, int n_iter) const
{
    for (int iter = 0; iter < n_iter; ++iter) {
        for (int i = 0; i < lvl.n; ++i) {
            double sigma = b(i);
            for (Eigen::SparseMatrix<double>::InnerIterator it(lvl.A, i); it; ++it) {
                int j = static_cast<int>(it.col());
                if (j != i) sigma -= it.value() * x(j);
            }
            x(i) = sigma * lvl.diag_inv(i);
        }
    }
}

void AMGPreconditioner::v_cycle(int level, Eigen::VectorXd& x,
                                const Eigen::VectorXd& b) const
{
    const Level& lvl = levels_[level];

    // Coarsest level: Gauss-Seidel solve
    if (level == static_cast<int>(levels_.size()) - 1) {
        gauss_seidel(lvl, x, b, 50);
        return;
    }

    // Pre-smoothing
    gauss_seidel(lvl, x, b, pre_smooth_);

    // Compute residual: r = b - A*x
    Eigen::VectorXd r = b - lvl.A * x;

    // Restrict residual to coarse level
    Eigen::VectorXd r_c = lvl.R * r;

    // Coarse-grid correction
    Eigen::VectorXd e_c = Eigen::VectorXd::Zero(levels_[level + 1].n);
    v_cycle(level + 1, e_c, r_c);

    // Prolongate and correct
    x += lvl.P * e_c;

    // Post-smoothing
    gauss_seidel(lvl, x, b, post_smooth_);
}

Eigen::VectorXd AMGPreconditioner::apply(const Eigen::VectorXd& r) const
{
    Eigen::VectorXd x = Eigen::VectorXd::Zero(r.size());
    v_cycle(0, x, r);
    return x;
}

} // namespace twofluid
