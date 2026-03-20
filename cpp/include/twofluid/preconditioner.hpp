#pragma once

#include <functional>
#include <string>
#include <vector>
#include <memory>
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace twofluid {

struct PreconditionerInfo {
    std::string method;
    double setup_time = 0.0;
};

/// Create a preconditioner for the sparse matrix A.
/// Supported methods: "none", "jacobi", "ilu0", "amg"
std::pair<std::function<Eigen::VectorXd(const Eigen::VectorXd&)>, PreconditionerInfo>
create_preconditioner(const Eigen::SparseMatrix<double>& A,
                      const std::string& method = "none");

/// Simple algebraic multigrid V-cycle preconditioner.
/// Uses Gauss-Seidel smoothing and pairwise aggregation coarsening.
class AMGPreconditioner {
public:
    AMGPreconditioner(const Eigen::SparseMatrix<double>& A,
                      int max_levels = 10, int max_coarse = 50,
                      int pre_smooth = 2, int post_smooth = 2);

    Eigen::VectorXd apply(const Eigen::VectorXd& r) const;

    int n_levels() const { return static_cast<int>(levels_.size()); }

private:
    struct Level {
        Eigen::SparseMatrix<double> A;    // operator at this level
        Eigen::SparseMatrix<double> P;    // prolongation (fine <- coarse)
        Eigen::SparseMatrix<double> R;    // restriction (coarse <- fine)
        Eigen::VectorXd diag_inv;         // inverse diagonal for smoothing
        int n = 0;                         // size at this level
    };
    std::vector<Level> levels_;
    int pre_smooth_, post_smooth_;

    void build_hierarchy(const Eigen::SparseMatrix<double>& A,
                         int max_levels, int max_coarse);
    void gauss_seidel(const Level& lvl, Eigen::VectorXd& x,
                      const Eigen::VectorXd& b, int n_iter) const;
    void v_cycle(int level, Eigen::VectorXd& x,
                 const Eigen::VectorXd& b) const;
};

} // namespace twofluid
