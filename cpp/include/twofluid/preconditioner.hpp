#pragma once

#include <functional>
#include <string>
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace twofluid {

struct PreconditionerInfo {
    std::string method;
    double setup_time = 0.0;
};

/// Create a preconditioner for the sparse matrix A.
/// Supported methods: "none", "jacobi", "ilu0"
std::pair<std::function<Eigen::VectorXd(const Eigen::VectorXd&)>, PreconditionerInfo>
create_preconditioner(const Eigen::SparseMatrix<double>& A,
                      const std::string& method = "none");

} // namespace twofluid
