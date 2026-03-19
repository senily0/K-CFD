#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <Eigen/Dense>
#include "twofluid/mesh.hpp"
#include "twofluid/fields.hpp"
#include "twofluid/fvm_operators.hpp"

namespace twofluid {

/// First-order irreversible reaction A -> B
class FirstOrderReaction {
public:
    explicit FirstOrderReaction(double k_r = 1.0);

    double k_r;

    Eigen::VectorXd reaction_rate(const Eigen::VectorXd& C_A, double rho) const;
    std::pair<Eigen::VectorXd, Eigen::VectorXd>
        source_linearization(const Eigen::VectorXd& C_A, double rho) const;
};

struct SpeciesSolveResult {
    bool converged;
    int iterations;
    std::vector<double> residuals;
};

/// Species transport solver
class SpeciesTransportSolver {
public:
    SpeciesTransportSolver(FVMesh& mesh, double rho = 1.0,
                            double D = 1e-5,
                            FirstOrderReaction* reaction = nullptr);

    double rho, D;
    double alpha_C = 0.8;
    ScalarField C;
    std::unordered_map<std::string, BoundaryCondition> bc_C;

    void set_bc(const std::string& patch, const std::string& type, double value = 0.0);
    SpeciesSolveResult solve_steady(const VectorField& U,
                                     const Eigen::VectorXd& mass_flux,
                                     int max_iter = 200, double tol = 1e-6);

private:
    FVMesh& mesh_;
    FirstOrderReaction* reaction_;
};

} // namespace twofluid
