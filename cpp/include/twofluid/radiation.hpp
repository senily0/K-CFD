#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <Eigen/Dense>
#include "twofluid/mesh.hpp"
#include "twofluid/fields.hpp"
#include "twofluid/fvm_operators.hpp"

namespace twofluid {

constexpr double SIGMA_SB = 5.67e-8;

struct RadiationResult {
    bool converged;
    int iterations;
    std::vector<double> residuals;
};

class P1RadiationModel {
public:
    P1RadiationModel(FVMesh& mesh, double kappa = 1.0);

    double kappa;
    ScalarField G;
    double alpha_G = 0.8;
    std::unordered_map<std::string, BoundaryCondition> bc_G;

    RadiationResult solve(const ScalarField& T, int max_iter = 100,
                          double tol = 1e-6);
    Eigen::VectorXd compute_radiative_source(const ScalarField& T) const;
    void set_bc(const std::string& patch, const std::string& bc_type,
                double T_wall = 0.0);

private:
    FVMesh& mesh_;
};

} // namespace twofluid
