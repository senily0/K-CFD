#include "twofluid/radiation.hpp"
#include "twofluid/linear_solver.hpp"
#include <cmath>

namespace twofluid {

P1RadiationModel::P1RadiationModel(FVMesh& mesh, double kappa)
    : kappa(kappa), G(mesh, "G"), mesh_(mesh) {}

RadiationResult P1RadiationModel::solve(const ScalarField& T,
                                         int max_iter, double tol) {
    int n = mesh_.n_cells;
    std::vector<double> residuals;

    // Diffusion coefficient: gamma = 1/(3*kappa)
    ScalarField gamma(mesh_, "gamma_rad");
    gamma.values.setConstant(1.0 / (3.0 * kappa));

    // Emission source: 4*kappa*sigma*T^4
    Eigen::VectorXd emission(n);
    for (int i = 0; i < n; ++i) {
        double T4 = T.values(i) * T.values(i) * T.values(i) * T.values(i);
        emission(i) = 4.0 * kappa * SIGMA_SB * T4;
    }

    for (int it = 0; it < max_iter; ++it) {
        FVMSystem system(n);

        // Diffusion
        diffusion_operator(mesh_, gamma, system);

        // Linearized source: Su = 4*kappa*sigma*T^4, Sp = -kappa
        Eigen::VectorXd Sp = Eigen::VectorXd::Constant(n, -kappa);
        linearized_source(mesh_, Sp, emission, system);

        // Boundary conditions
        Eigen::VectorXd zero_flux = Eigen::VectorXd::Zero(mesh_.n_faces);
        apply_boundary_conditions(mesh_, G, gamma, zero_flux, system, bc_G);

        // Solve
        Eigen::VectorXd G_new = solve_linear_system(system, G.values, "direct");

        // Residual
        double res = std::sqrt((G_new - G.values).squaredNorm() / n);
        residuals.push_back(res);

        G.values = G_new;

        if (res < tol) {
            return {true, it + 1, residuals};
        }
    }

    return {false, max_iter, residuals};
}

Eigen::VectorXd P1RadiationModel::compute_radiative_source(
    const ScalarField& T) const {
    int n = mesh_.n_cells;
    Eigen::VectorXd q_r(n);
    for (int i = 0; i < n; ++i) {
        double T4 = T.values(i) * T.values(i) * T.values(i) * T.values(i);
        q_r(i) = kappa * (G.values(i) - 4.0 * SIGMA_SB * T4);
    }
    return q_r;
}

void P1RadiationModel::set_bc(const std::string& patch,
                                const std::string& bc_type,
                                double T_wall) {
    if (bc_type == "marshak" && T_wall > 0.0) {
        bc_G[patch] = {"dirichlet", 0.0};
        double T4 = T_wall * T_wall * T_wall * T_wall;
        double G_wall = 4.0 * SIGMA_SB * T4;
        if (mesh_.boundary_patches.count(patch)) {
            int nf = static_cast<int>(mesh_.boundary_patches.at(patch).size());
            G.set_boundary(patch, Eigen::VectorXd::Constant(nf, G_wall));
        }
    } else if (bc_type == "zero_gradient") {
        bc_G[patch] = {"zero_gradient", 0.0};
    } else if (bc_type == "dirichlet") {
        bc_G[patch] = {"dirichlet", 0.0};
    }
}

} // namespace twofluid
