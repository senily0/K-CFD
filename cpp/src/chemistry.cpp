#include "twofluid/chemistry.hpp"
#include "twofluid/linear_solver.hpp"
#include <cmath>

namespace twofluid {

FirstOrderReaction::FirstOrderReaction(double k_r) : k_r(k_r) {}

Eigen::VectorXd FirstOrderReaction::reaction_rate(
    const Eigen::VectorXd& C_A, double /*rho*/) const {
    return -k_r * C_A;
}

std::pair<Eigen::VectorXd, Eigen::VectorXd>
FirstOrderReaction::source_linearization(
    const Eigen::VectorXd& C_A, double /*rho*/) const {
    int n = static_cast<int>(C_A.size());
    Eigen::VectorXd Su = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd Sp = Eigen::VectorXd::Constant(n, -k_r);
    return {Su, Sp};
}

SpeciesTransportSolver::SpeciesTransportSolver(
    FVMesh& mesh, double rho, double D, FirstOrderReaction* reaction)
    : rho(rho), D(D), C(mesh, "C_A"), mesh_(mesh), reaction_(reaction) {}

void SpeciesTransportSolver::set_bc(const std::string& patch,
                                      const std::string& type,
                                      double value) {
    if (type == "dirichlet") {
        bc_C[patch] = {"dirichlet", 0.0};
        if (mesh_.boundary_patches.count(patch)) {
            int nf = static_cast<int>(mesh_.boundary_patches.at(patch).size());
            C.set_boundary(patch, Eigen::VectorXd::Constant(nf, value));
        }
    } else {
        bc_C[patch] = {"zero_gradient", 0.0};
    }
}

SpeciesSolveResult SpeciesTransportSolver::solve_steady(
    const VectorField& /*U*/, const Eigen::VectorXd& mass_flux,
    int max_iter, double tol) {

    int n = mesh_.n_cells;
    std::vector<double> residuals;

    ScalarField gamma(mesh_, "D_eff");
    gamma.values.setConstant(rho * D);

    for (int it = 0; it < max_iter; ++it) {
        FVMSystem system(n);

        diffusion_operator(mesh_, gamma, system);
        convection_operator_upwind(mesh_, mass_flux, system);

        if (reaction_) {
            auto [Su, Sp] = reaction_->source_linearization(C.values, rho);
            linearized_source(mesh_, Sp, Su, system);
        }

        apply_boundary_conditions(mesh_, C, gamma, mass_flux, system, bc_C);
        under_relax(system, C, alpha_C);

        Eigen::VectorXd C_new = solve_linear_system(system, C.values, "direct");

        double res = std::sqrt((C_new - C.values).squaredNorm() / n);
        residuals.push_back(res);

        C.values = C_new.cwiseMax(0.0);

        if (it > 0 && res < tol) {
            return {true, it + 1, residuals};
        }
    }
    return {false, max_iter, residuals};
}

} // namespace twofluid
