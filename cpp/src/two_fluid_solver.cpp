#include "twofluid/two_fluid_solver.hpp"
#include "twofluid/gradient.hpp"
#include "twofluid/linear_solver.hpp"
#include "twofluid/interpolation.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

namespace twofluid {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TwoFluidSolver::TwoFluidSolver(FVMesh& mesh)
    : mesh_(mesh),
      alpha_g_(mesh, "alpha_gas"),
      alpha_l_(mesh, "alpha_liquid"),
      p_(mesh, "pressure"),
      T_l_(mesh, "temperature_liquid"),
      T_g_(mesh, "temperature_gas"),
      U_l_(mesh, "velocity_liquid"),
      U_g_(mesh, "velocity_gas")
{
    int ndim = mesh.ndim;
    g = Eigen::VectorXd::Zero(ndim);
    g[ndim - 1] = -9.81;

    // Default initialization
    alpha_l_.set_uniform(0.95);
    alpha_g_.set_uniform(0.05);
    T_l_.set_uniform(300.0);
    T_g_.set_uniform(300.0);

    int n = mesh.n_cells;
    aP_l_ = Eigen::VectorXd::Ones(n);
    aP_g_storage_ = Eigen::VectorXd::Ones(n);

    build_face_bc_cache();
}

// ---------------------------------------------------------------------------
// Initialize
// ---------------------------------------------------------------------------

void TwoFluidSolver::initialize(double alpha_g_init) {
    alpha_g_.set_uniform(alpha_g_init);
    alpha_l_.set_uniform(1.0 - alpha_g_init);
}

// ---------------------------------------------------------------------------
// Boundary conditions
// ---------------------------------------------------------------------------

void TwoFluidSolver::set_inlet_bc(const std::string& patch, double alpha_g_val,
                                    const Eigen::VectorXd& U_l_val,
                                    const Eigen::VectorXd& U_g_val,
                                    double T_l_val, double T_g_val) {
    if (mesh_.boundary_patches.find(patch) == mesh_.boundary_patches.end())
        return;

    alpha_g_.set_boundary(patch, alpha_g_val);
    alpha_l_.set_boundary(patch, 1.0 - alpha_g_val);
    U_l_.set_boundary(patch, U_l_val);
    U_g_.set_boundary(patch, U_g_val);

    bc_alpha_[patch] = {"dirichlet"};
    bc_u_l_[patch] = {"dirichlet"};
    bc_u_g_[patch] = {"dirichlet"};
    bc_p_[patch] = {"zero_gradient"};

    if (T_l_val != 0.0) {
        T_l_.set_boundary(patch, T_l_val);
        bc_T_l_[patch] = {"dirichlet"};
    }
    if (T_g_val != 0.0) {
        T_g_.set_boundary(patch, T_g_val);
        bc_T_g_[patch] = {"dirichlet"};
    }

    build_face_bc_cache();
}

void TwoFluidSolver::set_outlet_bc(const std::string& patch, double p_val) {
    if (mesh_.boundary_patches.find(patch) == mesh_.boundary_patches.end())
        return;

    p_.set_boundary(patch, p_val);
    bc_p_[patch] = {"dirichlet"};
    bc_u_l_[patch] = {"zero_gradient"};
    bc_u_g_[patch] = {"zero_gradient"};
    bc_alpha_[patch] = {"zero_gradient"};
    bc_T_l_[patch] = {"zero_gradient"};
    bc_T_g_[patch] = {"zero_gradient"};

    build_face_bc_cache();
}

void TwoFluidSolver::set_wall_bc(const std::string& patch, double q_wall) {
    if (mesh_.boundary_patches.find(patch) == mesh_.boundary_patches.end())
        return;

    int ndim = mesh_.ndim;
    int nf = static_cast<int>(mesh_.boundary_patches[patch].size());

    // Liquid: no-slip (Dirichlet zero)
    Eigen::MatrixXd zero_u = Eigen::MatrixXd::Zero(nf, ndim);
    U_l_.set_boundary(patch, zero_u);
    bc_u_l_[patch] = {"dirichlet"};

    // Gas: free-slip (zero gradient)
    bc_u_g_[patch] = {"zero_gradient"};

    bc_p_[patch] = {"zero_gradient"};
    bc_alpha_[patch] = {"zero_gradient"};

    if (q_wall != 0.0) {
        wall_heat_flux_[patch] = q_wall;
    }
    bc_T_l_[patch] = {"zero_gradient"};
    bc_T_g_[patch] = {"zero_gradient"};

    build_face_bc_cache();
}

// ---------------------------------------------------------------------------
// Face BC cache
// ---------------------------------------------------------------------------

void TwoFluidSolver::build_face_bc_cache() {
    face_bc_cache_.clear();
    for (const auto& [bname, fids] : mesh_.boundary_patches) {
        for (int li = 0; li < static_cast<int>(fids.size()); ++li) {
            face_bc_cache_[fids[li]] = {bname, li};
        }
    }
}

// ---------------------------------------------------------------------------
// Update zero-gradient boundaries
// ---------------------------------------------------------------------------

void TwoFluidSolver::update_zero_gradient_boundaries() {
    int ndim = mesh_.ndim;

    // Velocity fields
    auto update_vec = [&](const std::unordered_map<std::string, BoundaryCondition>& bc_dict,
                          VectorField& field) {
        for (const auto& [bname, bc] : bc_dict) {
            if (bc.type == "zero_gradient" &&
                mesh_.boundary_patches.find(bname) != mesh_.boundary_patches.end()) {
                const auto& fids = mesh_.boundary_patches.at(bname);
                auto bv_it = field.boundary_values.find(bname);
                if (bv_it == field.boundary_values.end()) {
                    // Initialize boundary values if not present
                    Eigen::MatrixXd bv = Eigen::MatrixXd::Zero(
                        static_cast<int>(fids.size()), ndim);
                    field.boundary_values[bname] = bv;
                    bv_it = field.boundary_values.find(bname);
                }
                for (int li = 0; li < static_cast<int>(fids.size()); ++li) {
                    int owner = mesh_.faces[fids[li]].owner;
                    bv_it->second.row(li) = field.values.row(owner);
                }
            }
        }
    };

    update_vec(bc_u_l_, U_l_);
    update_vec(bc_u_g_, U_g_);

    // Scalar fields
    auto update_scalar = [&](const std::unordered_map<std::string, BoundaryCondition>& bc_dict,
                             ScalarField& field) {
        for (const auto& [bname, bc] : bc_dict) {
            if (bc.type == "zero_gradient" &&
                mesh_.boundary_patches.find(bname) != mesh_.boundary_patches.end()) {
                const auto& fids = mesh_.boundary_patches.at(bname);
                auto bv_it = field.boundary_values.find(bname);
                if (bv_it == field.boundary_values.end()) {
                    Eigen::VectorXd bv = Eigen::VectorXd::Zero(
                        static_cast<int>(fids.size()));
                    field.boundary_values[bname] = bv;
                    bv_it = field.boundary_values.find(bname);
                }
                for (int li = 0; li < static_cast<int>(fids.size()); ++li) {
                    int owner = mesh_.faces[fids[li]].owner;
                    bv_it->second[li] = field.values[owner];
                }
            }
        }
    };

    if (solve_energy) {
        update_scalar(bc_T_l_, T_l_);
        update_scalar(bc_T_g_, T_g_);
    }

    // Volume fraction
    for (const auto& [bname, bc] : bc_alpha_) {
        if (bc.type == "zero_gradient" &&
            mesh_.boundary_patches.find(bname) != mesh_.boundary_patches.end()) {
            const auto& fids = mesh_.boundary_patches.at(bname);
            // alpha_g
            auto bv_ag = alpha_g_.boundary_values.find(bname);
            if (bv_ag == alpha_g_.boundary_values.end()) {
                alpha_g_.boundary_values[bname] = Eigen::VectorXd::Zero(
                    static_cast<int>(fids.size()));
                bv_ag = alpha_g_.boundary_values.find(bname);
            }
            auto bv_al = alpha_l_.boundary_values.find(bname);
            if (bv_al == alpha_l_.boundary_values.end()) {
                alpha_l_.boundary_values[bname] = Eigen::VectorXd::Zero(
                    static_cast<int>(fids.size()));
                bv_al = alpha_l_.boundary_values.find(bname);
            }
            for (int li = 0; li < static_cast<int>(fids.size()); ++li) {
                int owner = mesh_.faces[fids[li]].owner;
                bv_ag->second[li] = alpha_g_.values[owner];
                bv_al->second[li] = alpha_l_.values[owner];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Transient solve
// ---------------------------------------------------------------------------

SolveResult TwoFluidSolver::solve_transient(double t_end, double dt_in,
                                             int report_interval) {
    dt = dt_in;
    auto t_start = std::chrono::high_resolution_clock::now();

    double t = 0.0;
    int step = 0;
    std::vector<double> residuals;

    while (t < t_end - 1e-15) {
        // Store old values
        alpha_g_.store_old();
        alpha_l_.store_old();
        U_l_.store_old();
        U_g_.store_old();
        if (solve_energy) {
            T_l_.store_old();
            T_g_.store_old();
        }

        // SIMPLE inner iteration
        double max_res = simple_iteration();

        t += dt;
        step += 1;
        residuals.push_back(max_res);

        if (step % report_interval == 0) {
            std::cout << "  Step " << step << ", t=" << t
                      << ", residual=" << max_res
                      << ", alpha_g: [" << alpha_g_.min()
                      << ", " << alpha_g_.max() << "]" << std::endl;
        }
    }

    auto t_end_clock = std::chrono::high_resolution_clock::now();
    double wall = std::chrono::duration<double>(t_end_clock - t_start).count();

    bool converged = !residuals.empty() && residuals.back() < tol;
    return {converged, step, residuals, wall};
}

// ---------------------------------------------------------------------------
// SIMPLE iteration
// ---------------------------------------------------------------------------

double TwoFluidSolver::simple_iteration() {
    int n_inner = std::min(max_outer_iter, 30);
    int ndim = mesh_.ndim;
    double max_res = 1e10;

    for (int inner = 0; inner < n_inner; ++inner) {
        // Update zero-gradient boundaries
        update_zero_gradient_boundaries();

        // Mass flux for both phases
        Eigen::VectorXd mf_l = compute_mass_flux(U_l_, rho_l, mesh_);
        Eigen::VectorXd mf_g = compute_mass_flux(U_g_, rho_g, mesh_);

        std::vector<double> res_mom;
        double res_p = 0.0;
        double res_alpha = 0.0;

        if (solve_momentum) {
            // Drag coefficient
            Eigen::VectorXd K_drag = drag_coefficient_implicit(
                alpha_g_.values, rho_l,
                U_g_.values, U_l_.values,
                d_b, mu_l
            );

            // Momentum equations (each phase, each component)
            for (int comp = 0; comp < ndim; ++comp) {
                res_mom.push_back(solve_phase_momentum("liquid", comp, mf_l, K_drag));
            }
            for (int comp = 0; comp < ndim; ++comp) {
                res_mom.push_back(solve_phase_momentum("gas", comp, mf_g, K_drag));
            }

            // Pressure correction
            res_p = solve_pressure_correction(mf_l, mf_g);

            // Volume fraction
            Eigen::VectorXd dot_m = solve_energy
                ? compute_phase_change_rate()
                : Eigen::VectorXd::Zero(mesh_.n_cells);
            res_alpha = solve_volume_fraction(mf_g, dot_m);
        } else {
            Eigen::VectorXd dot_m = solve_energy
                ? compute_phase_change_rate()
                : Eigen::VectorXd::Zero(mesh_.n_cells);
            if (solve_energy) {
                res_alpha = solve_volume_fraction(mf_g, dot_m);
            }
        }

        // Energy equations
        double res_energy = 0.0;
        if (solve_energy) {
            Eigen::VectorXd dot_m = compute_phase_change_rate();
            Eigen::VectorXd mf_l2 = compute_mass_flux(U_l_, rho_l, mesh_);
            Eigen::VectorXd mf_g2 = compute_mass_flux(U_g_, rho_g, mesh_);
            res_energy = solve_coupled_energy(mf_l2, mf_g2, dot_m);
        }

        std::vector<double> all_res = res_mom;
        all_res.push_back(res_p);
        all_res.push_back(res_alpha);
        all_res.push_back(res_energy);

        max_res = 0.0;
        for (double r : all_res) {
            double val = std::isfinite(r) ? r : 1e10;
            max_res = std::max(max_res, val);
        }

        if (max_res < tol) {
            break;
        }
    }

    return std::isfinite(max_res) ? max_res : 1e10;
}

// ---------------------------------------------------------------------------
// Phase change rate (Lee model)
// ---------------------------------------------------------------------------

Eigen::VectorXd TwoFluidSolver::compute_phase_change_rate() {
    int n = mesh_.n_cells;
    Eigen::VectorXd dot_m = Eigen::VectorXd::Zero(n);
    double r = r_phase_change;

    for (int ci = 0; ci < n; ++ci) {
        double tl = T_l_.values[ci];
        double tg = T_g_.values[ci];
        double al = alpha_l_.values[ci];
        double ag = alpha_g_.values[ci];

        // Evaporation: T_l > T_sat
        if (tl > T_sat && al > 1e-6) {
            dot_m[ci] += r * al * rho_l * (tl - T_sat) / T_sat;
        }

        // Condensation: T_g < T_sat
        if (tg < T_sat && ag > 1e-6) {
            dot_m[ci] -= r * ag * rho_g * (T_sat - tg) / T_sat;
        }
    }

    return dot_m;
}

// ---------------------------------------------------------------------------
// Phase momentum equation
// ---------------------------------------------------------------------------

double TwoFluidSolver::solve_phase_momentum(const std::string& phase, int comp,
                                              const Eigen::VectorXd& mass_flux,
                                              const Eigen::VectorXd& K_drag) {
    int n = mesh_.n_cells;
    int ndim = mesh_.ndim;
    FVMSystem system(n);

    bool is_liquid = (phase == "liquid");
    VectorField& U = is_liquid ? U_l_ : U_g_;
    ScalarField& alpha = is_liquid ? alpha_l_ : alpha_g_;
    double rho = is_liquid ? rho_l : rho_g;
    double mu = is_liquid ? mu_l : mu_g;
    VectorField& U_other = is_liquid ? U_g_ : U_l_;
    auto& bc = is_liquid ? bc_u_l_ : bc_u_g_;

    // Extract scalar component field
    ScalarField phi(mesh_, "u_" + phase + "_" + std::to_string(comp));
    phi.values = U.values.col(comp);
    for (auto& [bname, bv] : U.boundary_values) {
        phi.boundary_values[bname] = bv.col(comp);
    }
    if (U.old_values.has_value()) {
        phi.old_values = U.old_values.value().col(comp);
    }

    // Diffusion: alpha * mu
    ScalarField gamma(mesh_, "gamma");
    gamma.values = alpha.values * mu;

    // BIT for liquid phase
    Eigen::VectorXd mu_BIT = sato_bubble_induced_turbulence(
        alpha_g_.values, rho_l,
        U_g_.values, U_l_.values, d_b
    );
    if (is_liquid) {
        gamma.values += alpha.values.cwiseProduct(mu_BIT);
    }

    diffusion_operator(mesh_, gamma, system);

    // Convection: alpha * mass_flux
    // Get alpha at face owner for upwind
    Eigen::VectorXd alpha_mf(mesh_.n_faces);
    for (int fid = 0; fid < mesh_.n_faces; ++fid) {
        int owner = mesh_.faces[fid].owner;
        alpha_mf[fid] = mass_flux[fid] * alpha.values[owner];
    }
    convection_operator_upwind(mesh_, alpha_mf, system);

    // Temporal term
    if (phi.old_values.has_value()) {
        for (int ci = 0; ci < n; ++ci) {
            double vol = mesh_.cells[ci].volume;
            double coeff = alpha.values[ci] * rho * vol / dt;
            system.add_diagonal(ci, coeff);
            system.add_source(ci, coeff * phi.old_values.value()[ci]);
        }
    }

    // Pressure gradient: -alpha * p_f * n_comp * A_f
    for (int fid = 0; fid < mesh_.n_faces; ++fid) {
        const Face& face = mesh_.faces[fid];
        int o = face.owner;
        double p_f;
        if (face.neighbour >= 0) {
            p_f = 0.5 * (p_.values[o] + p_.values[face.neighbour]);
        } else {
            p_f = p_.values[o];
        }
        double force = -alpha.values[o] * p_f * face.normal[comp] * face.area;
        system.add_source(o, force);
        if (face.neighbour >= 0) {
            int nb = face.neighbour;
            double force_nb = -alpha.values[nb] * p_f * face.normal[comp] * face.area;
            system.add_source(nb, -force_nb);
        }
    }

    // Drag (implicit)
    for (int ci = 0; ci < n; ++ci) {
        double vol = mesh_.cells[ci].volume;
        system.add_diagonal(ci, K_drag[ci] * vol);
        system.add_source(ci, K_drag[ci] * vol * U_other.values(ci, comp));
    }

    // Gravity
    for (int ci = 0; ci < n; ++ci) {
        double vol = mesh_.cells[ci].volume;
        system.add_source(ci, alpha.values[ci] * rho * g[comp] * vol);
    }

    // Boundary conditions
    apply_boundary_conditions(mesh_, phi, gamma, alpha_mf, system, bc);

    // Ensure diagonal minimum
    for (int ci = 0; ci < n; ++ci) {
        if (system.diag[ci] < 1e-20) {
            system.diag[ci] = 1e-10;
        }
    }

    // Under-relaxation
    under_relax(system, phi, alpha_u);

    // Solve
    phi.values = solve_linear_system(system, phi.values, "direct");

    // NaN/Inf protection
    if (phi.values.hasNaN() || !phi.values.allFinite()) {
        phi.values = U.values.col(comp);
    }

    U.values.col(comp) = phi.values;

    // Store aP (maximum across all components)
    Eigen::VectorXd aP_cur = system.diag.cwiseMax(1e-20);
    if (is_liquid) {
        if (comp == 0 || !has_aP_l_) {
            aP_l_ = aP_cur;
            has_aP_l_ = true;
        } else {
            aP_l_ = aP_l_.cwiseMax(aP_cur);
        }
    } else {
        if (comp == 0 || !has_aP_g_) {
            aP_g_storage_ = aP_cur;
            has_aP_g_ = true;
        } else {
            aP_g_storage_ = aP_g_storage_.cwiseMax(aP_cur);
        }
    }

    // Residual
    Eigen::SparseMatrix<double> A = system.to_sparse();
    Eigen::VectorXd r = A * phi.values - system.rhs;
    double b_norm = std::max(system.rhs.norm(), 1e-15);
    return r.norm() / b_norm;
}

// ---------------------------------------------------------------------------
// Two-phase pressure correction
// ---------------------------------------------------------------------------

double TwoFluidSolver::solve_pressure_correction(const Eigen::VectorXd& mf_l,
                                                   const Eigen::VectorXd& mf_g) {
    int n = mesh_.n_cells;
    int ndim = mesh_.ndim;
    FVMSystem system(n);

    for (int fid = 0; fid < mesh_.n_faces; ++fid) {
        const Face& face = mesh_.faces[fid];
        int o = face.owner;

        if (face.neighbour >= 0) {
            int nb = face.neighbour;
            Eigen::Vector3d d_vec = mesh_.cells[nb].center - mesh_.cells[o].center;
            double d_PN = d_vec.norm();
            if (d_PN < 1e-30) continue;

            // Two-phase contributions
            double al_f = 0.5 * (alpha_l_.values[o] + alpha_l_.values[nb]);
            double ag_f = 0.5 * (alpha_g_.values[o] + alpha_g_.values[nb]);
            double aP_l_f = 0.5 * (aP_l_[o] + aP_l_[nb]);
            double aP_g_f = 0.5 * (aP_g_storage_[o] + aP_g_storage_[nb]);

            double vol_f = 0.5 * (mesh_.cells[o].volume + mesh_.cells[nb].volume);

            double coeff_l = rho_l * al_f * vol_f / std::max(aP_l_f, 1e-30);
            double coeff_g = rho_g * ag_f * vol_f / std::max(aP_g_f, 1e-30);
            double coeff = (coeff_l + coeff_g) * face.area / d_PN;

            system.add_diagonal(o, coeff);
            system.add_diagonal(nb, coeff);
            system.add_off_diagonal(o, nb, -coeff);
            system.add_off_diagonal(nb, o, -coeff);

            // RHS: mass imbalance
            double total_mf = mf_l[fid] * alpha_l_.values[o]
                            + mf_g[fid] * alpha_g_.values[o];
            system.add_source(o, -total_mf);
            system.add_source(nb, total_mf);
        } else {
            double total_mf = mf_l[fid] * alpha_l_.values[o]
                            + mf_g[fid] * alpha_g_.values[o];
            system.add_source(o, -total_mf);
        }
    }

    // Reference pressure fix
    system.add_diagonal(0, 1e10);

    // Solve for p'
    Eigen::VectorXd p_prime = solve_linear_system(system, Eigen::VectorXd::Zero(n), "direct");

    // NaN/Inf protection
    if (p_prime.hasNaN() || !p_prime.allFinite()) {
        p_prime = Eigen::VectorXd::Zero(n);
    }

    // Update pressure
    p_.values += alpha_p * p_prime;

    // Velocity correction via gradient of p'
    Eigen::MatrixXd grad_pp = Eigen::MatrixXd::Zero(n, ndim);
    for (int fid = 0; fid < mesh_.n_faces; ++fid) {
        const Face& face = mesh_.faces[fid];
        int o = face.owner;
        if (face.neighbour >= 0) {
            int nb = face.neighbour;
            double dp = p_prime[nb] - p_prime[o];
            for (int d = 0; d < ndim; ++d) {
                double flux = dp * face.normal[d] * face.area;
                grad_pp(o, d) += flux;
                grad_pp(nb, d) -= flux;
            }
        }
    }
    for (int ci = 0; ci < n; ++ci) {
        double vol = mesh_.cells[ci].volume;
        if (vol > 1e-30) {
            grad_pp.row(ci) /= vol;
        }
    }

    // Correct velocities
    for (int ci = 0; ci < n; ++ci) {
        double vol = mesh_.cells[ci].volume;
        for (int d = 0; d < ndim; ++d) {
            double corr_l = -alpha_l_.values[ci] * vol
                            / std::max(aP_l_[ci], 1e-20) * grad_pp(ci, d);
            double corr_g = -alpha_g_.values[ci] * vol
                            / std::max(aP_g_storage_[ci], 1e-20) * grad_pp(ci, d);
            U_l_.values(ci, d) += alpha_p * corr_l;
            U_g_.values(ci, d) += alpha_p * corr_g;
        }
    }

    // Velocity NaN/Inf cleanup and physical limit clipping
    constexpr double U_MAX = 10.0;
    for (auto* U_field : {&U_l_, &U_g_}) {
        for (int ci = 0; ci < n; ++ci) {
            for (int d = 0; d < ndim; ++d) {
                double& v = U_field->values(ci, d);
                if (!std::isfinite(v)) v = 0.0;
                v = std::clamp(v, -U_MAX, U_MAX);
            }
        }
    }

    return p_prime.norm() / std::max(p_.values.norm(), 1e-15);
}

// ---------------------------------------------------------------------------
// Volume fraction transport
// ---------------------------------------------------------------------------

double TwoFluidSolver::solve_volume_fraction(const Eigen::VectorXd& mf_g,
                                               const Eigen::VectorXd& dot_m) {
    int n = mesh_.n_cells;
    FVMSystem system(n);

    // Zero diffusion
    ScalarField gamma_zero(mesh_, "gamma_zero");
    gamma_zero.set_uniform(1e-6);

    convection_operator_upwind(mesh_, mf_g, system);

    // Temporal
    if (alpha_g_.old_values.has_value()) {
        temporal_operator(mesh_, rho_g, dt, alpha_g_.old_values.value(), system);
    }

    // Phase change source: dot_m / rho_g
    for (int ci = 0; ci < n; ++ci) {
        double vol = mesh_.cells[ci].volume;
        if (std::abs(dot_m[ci]) > 1e-30) {
            system.add_source(ci, dot_m[ci] / rho_g * vol);
        }
    }

    apply_boundary_conditions(mesh_, alpha_g_, gamma_zero, mf_g, system, bc_alpha_);

    // Ensure diagonal minimum
    for (int ci = 0; ci < n; ++ci) {
        if (system.diag[ci] < 1e-20) {
            system.diag[ci] = 1e-10;
        }
    }

    under_relax(system, alpha_g_, alpha_alpha);

    Eigen::VectorXd alpha_old = alpha_g_.values;
    alpha_g_.values = solve_linear_system(system, alpha_g_.values, "direct");

    // NaN protection
    if (alpha_g_.values.hasNaN()) {
        alpha_g_.values = alpha_old;
    }

    // Physical limits: clip to [0, 0.9]
    for (int ci = 0; ci < n; ++ci) {
        alpha_g_.values[ci] = std::clamp(alpha_g_.values[ci], 0.0, 0.9);
    }
    alpha_l_.values = Eigen::VectorXd::Ones(n) - alpha_g_.values;

    // Residual: max change from old
    double res = 0.0;
    if (alpha_g_.old_values.has_value()) {
        res = (alpha_g_.values - alpha_g_.old_values.value()).cwiseAbs().maxCoeff();
    }
    return res;
}

// ---------------------------------------------------------------------------
// Coupled energy equations
// ---------------------------------------------------------------------------

double TwoFluidSolver::solve_coupled_energy(const Eigen::VectorXd& mf_l,
                                              const Eigen::VectorXd& mf_g,
                                              const Eigen::VectorXd& dot_m) {
    // Interfacial heat transfer coefficients (Ranz-Marshall)
    auto [h_i, a_i] = interfacial_heat_transfer(
        alpha_g_.values, rho_l, mu_l, cp_l, k_l,
        U_g_.values, U_l_.values, d_b
    );

    // Scale factors (matching Python h_i_coeff and a_i_coeff which default to 1.0)

    // Liquid energy
    double res_T_l = solve_phase_energy(
        "liquid", dt,
        alpha_l_, rho_l, cp_l, k_l,
        U_l_, mf_l, T_l_, T_g_,
        h_i, a_i, dot_m, bc_T_l_
    );

    // Gas energy (note: dot_m sign is negated for gas phase)
    Eigen::VectorXd neg_dot_m = -dot_m;
    double res_T_g = solve_phase_energy(
        "gas", dt,
        alpha_g_, rho_g, cp_g, k_g,
        U_g_, mf_g, T_g_, T_l_,
        h_i, a_i, neg_dot_m, bc_T_g_
    );

    return std::max(res_T_l, res_T_g);
}

// ---------------------------------------------------------------------------
// Single-phase energy equation
// ---------------------------------------------------------------------------

double TwoFluidSolver::solve_phase_energy(
    const std::string& phase, double dt_local,
    ScalarField& alpha, double rho, double cp, double k_cond,
    VectorField& U, const Eigen::VectorXd& mass_flux,
    ScalarField& T, ScalarField& T_other,
    const Eigen::VectorXd& h_i, const Eigen::VectorXd& a_i,
    const Eigen::VectorXd& dot_m_in,
    const std::unordered_map<std::string, BoundaryCondition>& bc_T)
{
    int n = mesh_.n_cells;
    FVMSystem system(n);

    // Diffusion: alpha * k_cond
    ScalarField gamma(mesh_, "gamma_T_" + phase);
    gamma.values = alpha.values * k_cond;
    diffusion_operator(mesh_, gamma, system);

    // Convection: alpha * cp * mass_flux (upwind alpha interpolation)
    Eigen::VectorXd alpha_cp_mf(mesh_.n_faces);
    for (int fid = 0; fid < mesh_.n_faces; ++fid) {
        const Face& face = mesh_.faces[fid];
        double alpha_f;
        if (face.neighbour >= 0) {
            double F = mass_flux[fid];
            alpha_f = (F >= 0) ? alpha.values[face.owner] : alpha.values[face.neighbour];
        } else {
            alpha_f = alpha.values[face.owner];
        }
        alpha_cp_mf[fid] = mass_flux[fid] * alpha_f * cp;
    }
    convection_operator_upwind(mesh_, alpha_cp_mf, system);

    // Temporal
    if (T.old_values.has_value()) {
        for (int ci = 0; ci < n; ++ci) {
            double vol = mesh_.cells[ci].volume;
            double coeff = alpha.values[ci] * rho * cp * vol / dt_local;
            system.add_diagonal(ci, coeff);
            system.add_source(ci, coeff * T.old_values.value()[ci]);
        }
    }

    // Interfacial heat transfer linearization: Q_i = h_i*a_i*(T_other - T)
    //   Su += h_i*a_i*T_other*V, Sp += h_i*a_i*V (added to diagonal)
    for (int ci = 0; ci < n; ++ci) {
        double vol = mesh_.cells[ci].volume;
        double hi_ai = h_i[ci] * a_i[ci] * vol;
        system.add_source(ci, hi_ai * T_other.values[ci]);
        system.add_diagonal(ci, hi_ai);
    }

    // Phase change latent heat
    for (int ci = 0; ci < n; ++ci) {
        double vol = mesh_.cells[ci].volume;
        // Both phases: -dot_m * h_fg * V
        // For liquid: dot_m > 0 means evaporation -> cooling
        // For gas: dot_m_in is already negated (-dot_m), so same formula
        system.add_source(ci, -dot_m_in[ci] * h_fg * vol);
    }

    // Wall heat flux
    bool is_liquid = (phase == "liquid");
    for (const auto& [patch_name, q_wall] : wall_heat_flux_) {
        auto it = mesh_.boundary_patches.find(patch_name);
        if (it != mesh_.boundary_patches.end()) {
            for (int fid : it->second) {
                const Face& face = mesh_.faces[fid];
                int owner = face.owner;
                double al = alpha_l_.values[owner];
                double ag = alpha_g_.values[owner];
                if (is_liquid) {
                    system.add_source(owner, q_wall * face.area * al);
                } else {
                    system.add_source(owner, q_wall * face.area * ag);
                }
            }
        }
    }

    // Boundary conditions
    apply_boundary_conditions(mesh_, T, gamma, alpha_cp_mf, system, bc_T);

    // Ensure diagonal minimum
    for (int ci = 0; ci < n; ++ci) {
        if (system.diag[ci] < 1e-20) {
            system.diag[ci] = 1e-10;
        }
    }

    // Under-relaxation
    under_relax(system, T, alpha_T);

    // Solve
    Eigen::VectorXd T_old_vals = T.values;
    T.values = solve_linear_system(system, T.values, "direct");

    // NaN protection and temperature clipping
    if (T.values.hasNaN() || !T.values.allFinite()) {
        T.values = T_old_vals;
    } else {
        for (int ci = 0; ci < n; ++ci) {
            T.values[ci] = std::clamp(T.values[ci], 280.0, 450.0);
        }
    }

    // Residual
    Eigen::SparseMatrix<double> A = system.to_sparse();
    Eigen::VectorXd r = A * T.values - system.rhs;
    double b_norm = std::max(system.rhs.norm(), 1e-15);
    return r.norm() / b_norm;
}

} // namespace twofluid
