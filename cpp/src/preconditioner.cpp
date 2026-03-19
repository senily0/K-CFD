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

    return {nullptr, {"unknown", 0.0}};
}

} // namespace twofluid
