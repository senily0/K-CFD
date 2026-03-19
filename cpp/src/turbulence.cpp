#include "twofluid/turbulence.hpp"
#include "twofluid/fvm_operators.hpp"
#include "twofluid/gradient.hpp"
#include "twofluid/linear_solver.hpp"

#include <algorithm>
#include <cmath>

namespace twofluid {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

KEpsilonModel::KEpsilonModel(FVMesh& mesh, double rho, double mu)
    : mesh_(mesh), rho_(rho), mu_(mu),
      k_(mesh, "k"), epsilon_(mesh, "epsilon") {
    initialize(1e-4, 1e-5);
}

// ---------------------------------------------------------------------------
// Initialize
// ---------------------------------------------------------------------------

void KEpsilonModel::initialize(double k_init, double eps_init) {
    k_init = std::max(k_init, 1e-10);
    eps_init = std::max(eps_init, 1e-10);
    k_.set_uniform(k_init);
    epsilon_.set_uniform(eps_init);
}

// ---------------------------------------------------------------------------
// Turbulent viscosity
// ---------------------------------------------------------------------------

Eigen::VectorXd KEpsilonModel::get_mu_t() const {
    int n = mesh_.n_cells;
    Eigen::VectorXd mu_t(n);
    for (int ci = 0; ci < n; ++ci) {
        double kv = std::max(k_.values(ci), 1e-10);
        double ev = std::max(epsilon_.values(ci), 1e-10);
        mu_t(ci) = rho_ * C_mu * kv * kv / ev;
    }
    return mu_t;
}

// ---------------------------------------------------------------------------
// Production term: P_k = mu_t * 2 * S_ij * S_ij
// ---------------------------------------------------------------------------

Eigen::VectorXd KEpsilonModel::compute_production(
    const VectorField& U, const Eigen::VectorXd& mu_t) const {

    int n = mesh_.n_cells;
    int ndim = mesh_.ndim;

    // Compute gradient of each velocity component
    std::vector<Eigen::MatrixXd> grads;  // grads[i] = grad(U_i), shape (n, ndim)
    for (int i = 0; i < ndim; ++i) {
        ScalarField comp(mesh_, "u_comp");
        comp.values = U.values.col(i);
        // Copy boundary values
        for (const auto& [bname, bvals] : U.boundary_values) {
            comp.set_boundary(bname, bvals.col(i));
        }
        grads.push_back(green_gauss_gradient(comp));
    }

    // S^2 = 2 * S_ij * S_ij, where S_ij = 0.5 * (dU_i/dx_j + dU_j/dx_i)
    Eigen::VectorXd S_sq = Eigen::VectorXd::Zero(n);
    for (int i = 0; i < ndim; ++i) {
        for (int j = 0; j < ndim; ++j) {
            for (int ci = 0; ci < n; ++ci) {
                double Sij = 0.5 * (grads[i](ci, j) + grads[j](ci, i));
                S_sq(ci) += 2.0 * Sij * Sij;
            }
        }
    }

    // P_k = mu_t * S^2
    Eigen::VectorXd P_k(n);
    for (int ci = 0; ci < n; ++ci) {
        P_k(ci) = mu_t(ci) * S_sq(ci);
    }
    return P_k;
}

// ---------------------------------------------------------------------------
// Solve k and epsilon transport equations
// ---------------------------------------------------------------------------

void KEpsilonModel::solve(const VectorField& U, const Eigen::VectorXd& mass_flux,
                          const std::unordered_map<std::string, std::string>& bc_types,
                          double alpha_k, double alpha_eps) {
    int n = mesh_.n_cells;

    // Current turbulent viscosity
    Eigen::VectorXd mu_t = get_mu_t();

    // Production term
    Eigen::VectorXd P_k = compute_production(U, mu_t);

    // Build BC maps for k and epsilon
    std::unordered_map<std::string, BoundaryCondition> bc_k, bc_eps;
    for (const auto& [patch, type] : bc_types) {
        BoundaryCondition bc;
        bc.type = type;
        bc_k[patch] = bc;
        bc_eps[patch] = bc;
    }

    // ---------------------------------------------------------------
    // k equation:
    //   diffusion(mu + mu_t/sigma_k) + convection + source(P_k - rho*eps)
    // ---------------------------------------------------------------
    {
        FVMSystem system(n);

        // Diffusion coefficient: mu + mu_t / sigma_k
        ScalarField gamma_k(mesh_, "gamma_k");
        for (int ci = 0; ci < n; ++ci) {
            gamma_k.values(ci) = mu_ + mu_t(ci) / sigma_k;
        }

        diffusion_operator(mesh_, gamma_k, system);
        convection_operator_upwind(mesh_, mass_flux, system);

        // Source: P_k - rho*epsilon
        // Linearized: Su = P_k, Sp = -rho*eps/k (negative for stability)
        Eigen::VectorXd Su(n), Sp(n);
        for (int ci = 0; ci < n; ++ci) {
            Su(ci) = P_k(ci);
            double kv = std::max(k_.values(ci), 1e-15);
            Sp(ci) = -rho_ * epsilon_.values(ci) / kv;
        }
        linearized_source(mesh_, Sp, Su, system);

        // Boundary conditions
        apply_boundary_conditions(mesh_, k_, gamma_k, mass_flux, system, bc_k);

        // Under-relaxation
        under_relax(system, k_, alpha_k);

        // Solve
        k_.values = solve_linear_system(system, k_.values, "bicgstab");

        // Clamp
        for (int ci = 0; ci < n; ++ci) {
            k_.values(ci) = std::max(k_.values(ci), 1e-10);
        }
    }

    // ---------------------------------------------------------------
    // epsilon equation:
    //   diffusion(mu + mu_t/sigma_eps) + convection
    //   + C1*eps/k*P_k - C2*rho*eps^2/k
    // ---------------------------------------------------------------
    {
        FVMSystem system(n);

        // Diffusion coefficient: mu + mu_t / sigma_eps
        ScalarField gamma_eps(mesh_, "gamma_eps");
        for (int ci = 0; ci < n; ++ci) {
            gamma_eps.values(ci) = mu_ + mu_t(ci) / sigma_eps;
        }

        diffusion_operator(mesh_, gamma_eps, system);
        convection_operator_upwind(mesh_, mass_flux, system);

        // Source: C1*eps/k*P_k - C2*rho*eps^2/k
        // Linearized: Su = C1*(eps/k)*P_k, Sp = -C2*rho*eps/k
        Eigen::VectorXd Su(n), Sp(n);
        for (int ci = 0; ci < n; ++ci) {
            double kv = std::max(k_.values(ci), 1e-10);
            double ev = std::max(epsilon_.values(ci), 1e-10);
            Su(ci) = C_1 * (ev / kv) * P_k(ci);
            Sp(ci) = -C_2 * rho_ * ev / kv;
        }
        linearized_source(mesh_, Sp, Su, system);

        // Boundary conditions
        apply_boundary_conditions(mesh_, epsilon_, gamma_eps, mass_flux, system, bc_eps);

        // Under-relaxation
        under_relax(system, epsilon_, alpha_eps);

        // Solve
        epsilon_.values = solve_linear_system(system, epsilon_.values, "bicgstab");

        // Clamp
        for (int ci = 0; ci < n; ++ci) {
            epsilon_.values(ci) = std::max(epsilon_.values(ci), 1e-10);
        }
    }
}

// ---------------------------------------------------------------------------
// Wall functions
// ---------------------------------------------------------------------------

void KEpsilonModel::apply_wall_functions(
    const VectorField& U, const std::vector<std::string>& wall_patches) {

    for (const auto& patch_name : wall_patches) {
        auto it = mesh_.boundary_patches.find(patch_name);
        if (it == mesh_.boundary_patches.end()) continue;

        for (int fid : it->second) {
            const Face& face = mesh_.faces[fid];
            int ci = face.owner;
            Eigen::Vector3d cell_center = mesh_.cells[ci].center;
            double y_dist = (cell_center - face.center).norm();

            if (y_dist < 1e-15) continue;

            // Cell velocity magnitude
            int ndim = mesh_.ndim;
            double U_mag = 0.0;
            for (int d = 0; d < ndim; ++d) {
                U_mag += U.values(ci, d) * U.values(ci, d);
            }
            U_mag = std::sqrt(U_mag);
            if (U_mag < 1e-15) continue;

            // Friction velocity estimate: u_tau = C_mu^0.25 * sqrt(k)
            double u_tau = std::max(
                std::pow(C_mu, 0.25) * std::sqrt(std::max(k_.values(ci), 1e-10)),
                1e-10);
            double y_plus = rho_ * u_tau * y_dist / mu_;

            if (y_plus > 11.225) {
                // Log-law region: apply wall functions
                k_.values(ci) = std::max(u_tau * u_tau / std::sqrt(C_mu), 1e-10);
                epsilon_.values(ci) = std::max(
                    u_tau * u_tau * u_tau / (kappa * y_dist), 1e-10);
            }
            // Linear sublayer: no modification needed
        }
    }
}

} // namespace twofluid
