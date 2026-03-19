#include "twofluid/turbulence_sst.hpp"
#include "twofluid/fvm_operators.hpp"
#include "twofluid/gradient.hpp"
#include "twofluid/linear_solver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace twofluid {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

KOmegaSSTModel::KOmegaSSTModel(FVMesh& mesh, double rho, double mu)
    : mesh_(mesh), rho_(rho), mu_(mu),
      k_(mesh, "k"), omega_(mesh, "omega"),
      wall_dist_(Eigen::VectorXd::Constant(mesh.n_cells, 1.0)) {
    initialize(1e-4, 1.0);
}

// ---------------------------------------------------------------------------
// Initialize
// ---------------------------------------------------------------------------

void KOmegaSSTModel::initialize(double k_init, double omega_init) {
    k_init = std::max(k_init, 1e-10);
    omega_init = std::max(omega_init, 1e-10);
    k_.set_uniform(k_init);
    omega_.set_uniform(omega_init);
}

// ---------------------------------------------------------------------------
// Wall distance
// ---------------------------------------------------------------------------

void KOmegaSSTModel::compute_wall_distance(
    const std::vector<std::string>& wall_patches) {

    int n = mesh_.n_cells;
    wall_dist_ = Eigen::VectorXd::Constant(n, std::numeric_limits<double>::max());

    for (const auto& patch_name : wall_patches) {
        auto it = mesh_.boundary_patches.find(patch_name);
        if (it == mesh_.boundary_patches.end()) continue;

        for (int fid : it->second) {
            const Face& face = mesh_.faces[fid];
            const Vec3& fc = face.center;

            for (int ci = 0; ci < n; ++ci) {
                double d = (mesh_.cells[ci].center - fc).norm();
                if (d < wall_dist_(ci)) {
                    wall_dist_(ci) = d;
                }
            }
        }
    }

    // Guard against no wall patches or degenerate meshes
    for (int ci = 0; ci < n; ++ci) {
        if (wall_dist_(ci) >= std::numeric_limits<double>::max()) {
            wall_dist_(ci) = 1.0;
        }
        wall_dist_(ci) = std::max(wall_dist_(ci), 1e-15);
    }

    wall_dist_computed_ = true;
}

// ---------------------------------------------------------------------------
// Cross-diffusion term: CDkw_i = 2*rho*sigma_w2/omega * (grad_k . grad_omega)
// ---------------------------------------------------------------------------

Eigen::VectorXd KOmegaSSTModel::compute_CDkw() const {
    int n = mesh_.n_cells;

    Eigen::MatrixXd grad_k = green_gauss_gradient(k_);
    Eigen::MatrixXd grad_w = green_gauss_gradient(omega_);

    Eigen::VectorXd CDkw(n);
    for (int ci = 0; ci < n; ++ci) {
        double dot = grad_k.row(ci).dot(grad_w.row(ci));
        double omega_v = std::max(omega_.values(ci), 1e-10);
        CDkw(ci) = 2.0 * rho_ * sigma_w2 / omega_v * dot;
    }
    return CDkw;
}

// ---------------------------------------------------------------------------
// Blending function F1
// F1 = tanh(arg1^4)
// arg1 = min(max(sqrt(k)/(beta_star*omega*y), 500*mu/(rho*y^2*omega)),
//            4*rho*sigma_w2*k/(CDkw*y^2))
// ---------------------------------------------------------------------------

Eigen::VectorXd KOmegaSSTModel::compute_F1() const {
    int n = mesh_.n_cells;
    Eigen::VectorXd CDkw = compute_CDkw();
    Eigen::VectorXd F1(n);

    for (int ci = 0; ci < n; ++ci) {
        double kv = std::max(k_.values(ci), 1e-10);
        double wv = std::max(omega_.values(ci), 1e-10);
        double y  = std::max(wall_dist_(ci), 1e-15);

        double term1 = std::sqrt(kv) / (beta_star * wv * y);
        double term2 = 500.0 * mu_ / (rho_ * y * y * wv);
        double CD    = std::max(CDkw(ci), 1e-10);
        double term3 = 4.0 * rho_ * sigma_w2 * kv / (CD * y * y);

        double arg1 = std::min(std::max(term1, term2), term3);
        double a4   = arg1 * arg1 * arg1 * arg1;
        F1(ci)      = std::tanh(a4);
    }
    return F1;
}

// ---------------------------------------------------------------------------
// Blending function F2
// F2 = tanh(arg2^2)
// arg2 = max(2*sqrt(k)/(beta_star*omega*y), 500*mu/(rho*y^2*omega))
// ---------------------------------------------------------------------------

Eigen::VectorXd KOmegaSSTModel::compute_F2() const {
    int n = mesh_.n_cells;
    Eigen::VectorXd F2(n);

    for (int ci = 0; ci < n; ++ci) {
        double kv = std::max(k_.values(ci), 1e-10);
        double wv = std::max(omega_.values(ci), 1e-10);
        double y  = std::max(wall_dist_(ci), 1e-15);

        double term1 = 2.0 * std::sqrt(kv) / (beta_star * wv * y);
        double term2 = 500.0 * mu_ / (rho_ * y * y * wv);
        double arg2  = std::max(term1, term2);
        F2(ci)       = std::tanh(arg2 * arg2);
    }
    return F2;
}

// ---------------------------------------------------------------------------
// Turbulent viscosity: mu_t = rho * a1 * k / max(a1*omega, S*F2)
// ---------------------------------------------------------------------------

Eigen::VectorXd KOmegaSSTModel::get_mu_t() const {
    int n = mesh_.n_cells;
    int ndim = mesh_.ndim;

    // Compute strain rate magnitude S = sqrt(2 * S_ij * S_ij)
    // Need velocity gradients -- approximate from k field gradient as fallback.
    // Full S computation requires U; here we store a minimal approximation
    // using sqrt(k) / (C_mu^0.25 * L) -- but the caller typically provides U
    // through solve(). For stand-alone get_mu_t() we use the simpler form
    // mu_t = rho * k / omega which is safe and consistent with the solve path.
    Eigen::VectorXd F2 = compute_F2();
    Eigen::VectorXd mu_t(n);

    for (int ci = 0; ci < n; ++ci) {
        double kv = std::max(k_.values(ci), 1e-10);
        double wv = std::max(omega_.values(ci), 1e-10);
        // Without U we cannot compute S; fall back to standard k-omega mu_t.
        // When called from solve(), the production-limited S is already applied.
        mu_t(ci) = rho_ * a1 * kv / std::max(a1 * wv, 1e-15);
    }
    return mu_t;
}

// ---------------------------------------------------------------------------
// Production term: P_k = mu_t * 2 * S_ij * S_ij
// Returns also S_mag (strain rate magnitude) via out parameter for reuse.
// ---------------------------------------------------------------------------

Eigen::VectorXd KOmegaSSTModel::compute_production(
    const VectorField& U, const Eigen::VectorXd& mu_t) const {

    int n = mesh_.n_cells;
    int ndim = mesh_.ndim;

    // Gradient of each velocity component
    std::vector<Eigen::MatrixXd> grads;
    for (int i = 0; i < ndim; ++i) {
        ScalarField comp(mesh_, "u_comp");
        comp.values = U.values.col(i);
        for (const auto& [bname, bvals] : U.boundary_values) {
            comp.set_boundary(bname, bvals.col(i));
        }
        grads.push_back(green_gauss_gradient(comp));
    }

    // S^2 = 2 * S_ij * S_ij
    Eigen::VectorXd S_sq = Eigen::VectorXd::Zero(n);
    for (int i = 0; i < ndim; ++i) {
        for (int j = 0; j < ndim; ++j) {
            for (int ci = 0; ci < n; ++ci) {
                double Sij = 0.5 * (grads[i](ci, j) + grads[j](ci, i));
                S_sq(ci) += 2.0 * Sij * Sij;
            }
        }
    }

    // P_k = mu_t * S^2, then apply production limiter
    Eigen::VectorXd P_k(n);
    for (int ci = 0; ci < n; ++ci) {
        double kv = std::max(k_.values(ci), 1e-10);
        double wv = std::max(omega_.values(ci), 1e-10);
        double raw = mu_t(ci) * S_sq(ci);
        // Limiter: P_k_tilde = min(P_k, 10 * beta_star * rho * k * omega)
        P_k(ci) = std::min(raw, 10.0 * beta_star * rho_ * kv * wv);
    }
    return P_k;
}

// ---------------------------------------------------------------------------
// Solve k and omega transport equations
// ---------------------------------------------------------------------------

void KOmegaSSTModel::solve(const VectorField& U, const Eigen::VectorXd& mass_flux,
                            const std::unordered_map<std::string, std::string>& bc_types,
                            double alpha_k, double alpha_w) {
    int n = mesh_.n_cells;

    // Blending functions
    Eigen::VectorXd F1 = compute_F1();
    Eigen::VectorXd F2 = compute_F2();

    // Blended SST coefficients per cell
    // phi_blend = F1 * phi1 + (1 - F1) * phi2
    Eigen::VectorXd sigma_k_b(n), sigma_w_b(n), beta_b(n), gamma_b(n);
    for (int ci = 0; ci < n; ++ci) {
        double f = F1(ci);
        sigma_k_b(ci) = f * sigma_k1 + (1.0 - f) * sigma_k2;
        sigma_w_b(ci) = f * sigma_w1 + (1.0 - f) * sigma_w2;
        beta_b(ci)    = f * beta_1   + (1.0 - f) * beta_2;
        gamma_b(ci)   = f * gamma_1  + (1.0 - f) * gamma_2;
    }

    // Compute mu_t using SST definition with strain rate
    // First pass: use simple estimate to compute production
    Eigen::VectorXd mu_t_simple(n);
    for (int ci = 0; ci < n; ++ci) {
        double kv = std::max(k_.values(ci), 1e-10);
        double wv = std::max(omega_.values(ci), 1e-10);
        mu_t_simple(ci) = rho_ * kv / wv;
    }

    // Compute production with simple mu_t (for strain rate S)
    // We need S^2 separately to apply the SST mu_t formula
    int ndim = mesh_.ndim;
    std::vector<Eigen::MatrixXd> grads;
    for (int i = 0; i < ndim; ++i) {
        ScalarField comp(mesh_, "u_comp");
        comp.values = U.values.col(i);
        for (const auto& [bname, bvals] : U.boundary_values) {
            comp.set_boundary(bname, bvals.col(i));
        }
        grads.push_back(green_gauss_gradient(comp));
    }

    Eigen::VectorXd S_sq = Eigen::VectorXd::Zero(n);
    for (int i = 0; i < ndim; ++i) {
        for (int j = 0; j < ndim; ++j) {
            for (int ci = 0; ci < n; ++ci) {
                double Sij = 0.5 * (grads[i](ci, j) + grads[j](ci, i));
                S_sq(ci) += 2.0 * Sij * Sij;
            }
        }
    }

    // SST mu_t = rho * a1 * k / max(a1*omega, S*F2)
    Eigen::VectorXd mu_t(n);
    for (int ci = 0; ci < n; ++ci) {
        double kv   = std::max(k_.values(ci), 1e-10);
        double wv   = std::max(omega_.values(ci), 1e-10);
        double S    = std::sqrt(std::max(S_sq(ci), 0.0));
        double denom = std::max(a1 * wv, S * F2(ci));
        mu_t(ci) = rho_ * a1 * kv / std::max(denom, 1e-15);
    }

    // Production with SST mu_t and production limiter
    Eigen::VectorXd P_k = compute_production(U, mu_t);

    // Cross-diffusion term for omega equation: (1 - F1) * 2*rho*sigma_w2/omega * grad_k.grad_omega
    Eigen::VectorXd CDkw_raw = compute_CDkw();
    Eigen::VectorXd CD_omega(n);
    for (int ci = 0; ci < n; ++ci) {
        CD_omega(ci) = (1.0 - F1(ci)) * CDkw_raw(ci);
    }

    // Build BC maps
    std::unordered_map<std::string, BoundaryCondition> bc_k, bc_w;
    for (const auto& [patch, type] : bc_types) {
        BoundaryCondition bc;
        bc.type = type;
        bc_k[patch] = bc;
        bc_w[patch] = bc;
    }

    // ---------------------------------------------------------------
    // k equation:
    //   d/dt(rho*k) + div(rho*U*k) = div((mu + mu_t*sigma_k)*grad(k))
    //                                 + P_k_tilde - beta_star*rho*k*omega
    // ---------------------------------------------------------------
    {
        FVMSystem system(n);

        ScalarField gamma_k(mesh_, "gamma_k");
        for (int ci = 0; ci < n; ++ci) {
            gamma_k.values(ci) = mu_ + mu_t(ci) * sigma_k_b(ci);
        }

        diffusion_operator(mesh_, gamma_k, system);
        convection_operator_upwind(mesh_, mass_flux, system);

        // Source: P_k - beta_star*rho*k*omega
        // Linearized: Su = P_k, Sp = -beta_star*rho*omega (negative for stability)
        Eigen::VectorXd Su(n), Sp(n);
        for (int ci = 0; ci < n; ++ci) {
            double wv = std::max(omega_.values(ci), 1e-10);
            Su(ci) = P_k(ci);
            Sp(ci) = -beta_star * rho_ * wv;  // multiplied by k implicitly
        }
        linearized_source(mesh_, Sp, Su, system);

        apply_boundary_conditions(mesh_, k_, gamma_k, mass_flux, system, bc_k);
        under_relax(system, k_, alpha_k);
        k_.values = solve_linear_system(system, k_.values, "bicgstab");

        for (int ci = 0; ci < n; ++ci) {
            k_.values(ci) = std::max(k_.values(ci), 1e-10);
        }
    }

    // ---------------------------------------------------------------
    // omega equation:
    //   d/dt(rho*omega) + div(rho*U*omega)
    //     = div((mu + mu_t*sigma_w)*grad(omega))
    //       + gamma*rho*S^2 - beta*rho*omega^2
    //       + (1-F1)*2*rho*sigma_w2/omega * (grad_k . grad_omega)
    // ---------------------------------------------------------------
    {
        FVMSystem system(n);

        ScalarField gamma_w(mesh_, "gamma_w");
        for (int ci = 0; ci < n; ++ci) {
            gamma_w.values(ci) = mu_ + mu_t(ci) * sigma_w_b(ci);
        }

        diffusion_operator(mesh_, gamma_w, system);
        convection_operator_upwind(mesh_, mass_flux, system);

        // Source: gamma*rho*S^2 - beta*rho*omega^2 + CD_omega
        // Linearized: Su = gamma*rho*S^2 + CD_omega_positive, Sp = -beta*rho*omega
        Eigen::VectorXd Su(n), Sp(n);
        for (int ci = 0; ci < n; ++ci) {
            double wv = std::max(omega_.values(ci), 1e-10);
            // Generation term: gamma * rho * S^2
            Su(ci) = gamma_b(ci) * rho_ * S_sq(ci);
            // Cross-diffusion added to Su (can be negative; handled explicitly)
            Su(ci) += CD_omega(ci);
            // Destruction: beta * rho * omega^2 -> linearize as beta*rho*omega * omega
            Sp(ci) = -beta_b(ci) * rho_ * wv;
        }
        linearized_source(mesh_, Sp, Su, system);

        apply_boundary_conditions(mesh_, omega_, gamma_w, mass_flux, system, bc_w);
        under_relax(system, omega_, alpha_w);
        omega_.values = solve_linear_system(system, omega_.values, "bicgstab");

        for (int ci = 0; ci < n; ++ci) {
            omega_.values(ci) = std::max(omega_.values(ci), 1e-10);
        }
    }
}

// ---------------------------------------------------------------------------
// Wall functions for omega
// ---------------------------------------------------------------------------

void KOmegaSSTModel::apply_wall_functions(
    const VectorField& U, const std::vector<std::string>& wall_patches) {

    for (const auto& patch_name : wall_patches) {
        auto it = mesh_.boundary_patches.find(patch_name);
        if (it == mesh_.boundary_patches.end()) continue;

        for (int fid : it->second) {
            const Face& face = mesh_.faces[fid];
            int ci = face.owner;
            double y_dist = (mesh_.cells[ci].center - face.center).norm();
            if (y_dist < 1e-15) continue;

            double kv = std::max(k_.values(ci), 1e-10);

            // Viscous sublayer: omega_vis = 6*mu/(rho*beta_1*y^2)
            double omega_vis = 6.0 * mu_ / (rho_ * beta_1 * y_dist * y_dist);

            // Log-layer: omega_log = sqrt(k) / (C_mu^0.25 * kappa * y)
            // C_mu^0.25 = beta_star^0.25
            double C_mu_025 = std::pow(beta_star, 0.25);
            double omega_log = std::sqrt(kv) / (C_mu_025 * kappa_vk * y_dist);

            // Blended wall BC (Menter 1994)
            omega_.values(ci) = std::sqrt(omega_vis * omega_vis + omega_log * omega_log);
            omega_.values(ci) = std::max(omega_.values(ci), 1e-10);
        }
    }
}

// ---------------------------------------------------------------------------
// y+ computation
// ---------------------------------------------------------------------------

Eigen::VectorXd KOmegaSSTModel::get_y_plus(
    const VectorField& U, const std::vector<std::string>& wall_patches) const {

    int n = mesh_.n_cells;
    Eigen::VectorXd y_plus = Eigen::VectorXd::Zero(n);

    for (const auto& patch_name : wall_patches) {
        auto it = mesh_.boundary_patches.find(patch_name);
        if (it == mesh_.boundary_patches.end()) continue;

        for (int fid : it->second) {
            const Face& face = mesh_.faces[fid];
            int ci = face.owner;
            double y_dist = (mesh_.cells[ci].center - face.center).norm();
            if (y_dist < 1e-15) continue;

            double kv = std::max(k_.values(ci), 1e-10);
            // u_tau = C_mu^0.25 * sqrt(k)
            double u_tau = std::pow(beta_star, 0.25) * std::sqrt(kv);
            y_plus(ci) = rho_ * u_tau * y_dist / mu_;
        }
    }
    return y_plus;
}

} // namespace twofluid
