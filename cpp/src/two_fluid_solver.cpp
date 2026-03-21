#include "twofluid/two_fluid_solver.hpp"
#include "twofluid/gradient.hpp"
#include "twofluid/linear_solver.hpp"
#include "twofluid/interpolation.hpp"
#include "twofluid/surface_tension.hpp"
#include "twofluid/steam_tables.hpp"
#include "twofluid/time_control.hpp"
#include "twofluid/chemistry.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#ifdef _OPENMP
#include <omp.h>
#endif

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

    if (solve_species) {
        if (species_k_r > 0) {
            reaction_ = std::make_unique<FirstOrderReaction>(species_k_r);
        }
        species_solver_ = std::make_unique<SpeciesTransportSolver>(
            mesh_, rho_l, species_D, reaction_.get());
        species_solver_->C.set_uniform(0.0);
        if (!species_bc_inlet_patch.empty()) {
            species_solver_->set_bc(species_bc_inlet_patch, "dirichlet", species_C_inlet);
        }
        // Set all other patches to zero_gradient
        for (auto& [name, fids] : mesh_.boundary_patches) {
            if (name != species_bc_inlet_patch) {
                species_solver_->set_bc(name, "zero_gradient");
            }
        }
    }
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

    // Adaptive time stepping controller (created only when enabled)
    std::unique_ptr<AdaptiveTimeControl> atc;
    if (adaptive_dt) {
        atc = std::make_unique<AdaptiveTimeControl>(dt_in, dt_min, dt_max, cfl_target);
    }

    while (t < t_end - 1e-15) {
        // Compute adaptive dt before storing old values
        if (adaptive_dt && atc) {
            // Velocity magnitude: max of both phases
            int ndim = mesh_.ndim;
            Eigen::VectorXd u_mag(mesh_.n_cells);
            for (int ci = 0; ci < mesh_.n_cells; ++ci) {
                double ul = U_l_.values.row(ci).head(ndim).norm();
                double ug = U_g_.values.row(ci).head(ndim).norm();
                u_mag[ci] = std::max(ul, ug);
            }
            auto info = atc->compute_dt(mesh_, u_mag);
            dt = info.dt;

            if (step % report_interval == 0) {
                std::cout << "  [ATC] CFL_max=" << info.cfl_max
                          << ", dt=" << dt
                          << " (limited by " << info.dt_limited_by << ")" << std::endl;
            }
        }

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

    // Update properties from IAPWS-IF97 if enabled
    if (property_model == "iapws97") {
        double T_avg_l = T_l_.mean();
        double T_avg_g = T_g_.mean();
        auto liq = IAPWS_IF97::liquid(T_avg_l, system_pressure);
        auto vap = IAPWS_IF97::vapor(T_avg_g, system_pressure);
        rho_l = liq.rho;
        mu_l  = liq.mu;
        cp_l  = liq.cp;
        k_l   = liq.k;
        rho_g = vap.rho;
        mu_g  = vap.mu;
        cp_g  = vap.cp;
        k_g   = vap.k;
        h_fg  = IAPWS_IF97::h_fg(system_pressure);
        T_sat = IAPWS_IF97::T_sat(system_pressure);
        if (sigma_surface <= 0.0) {
            sigma_surface = IAPWS_IF97::surface_tension(T_avg_l);
        }
    }

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
            // Drag model selection
            Eigen::VectorXd K_drag;
            if (drag_model == "grace") {
                K_drag = grace_drag(alpha_g_.values, rho_l, rho_g,
                                   U_g_.values, U_l_.values, d_b, mu_l, sigma_surface);
            } else if (drag_model == "tomiyama") {
                K_drag = tomiyama_drag(alpha_g_.values, rho_l, rho_g,
                                      U_g_.values, U_l_.values, d_b, mu_l, sigma_surface);
            } else if (drag_model == "ishii_zuber") {
                K_drag = ishii_zuber_drag(alpha_g_.values, rho_l,
                                          U_g_.values, U_l_.values, d_b, mu_l);
            } else {
                K_drag = drag_coefficient_implicit(alpha_g_.values, rho_l,
                                                   U_g_.values, U_l_.values, d_b, mu_l);
            }

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

        // Species transport (after energy solve)
        if (solve_species && species_solver_) {
            Eigen::VectorXd mf_l_sp = compute_mass_flux(U_l_, rho_l, mesh_);
            species_solver_->solve_steady(U_l_, mf_l_sp, 1, 1e-6);
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

#pragma omp parallel for schedule(static)
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
    if (U.old_old_values.has_value()) {
        phi.old_old_values = U.old_old_values.value().col(comp);
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

    if (n_nonorth_correctors > 0) {
        diffusion_operator_corrected(mesh_, gamma, phi, system, n_nonorth_correctors);
    } else {
        diffusion_operator(mesh_, gamma, system);
    }

    // Convection: alpha * mass_flux
    // Get alpha at face owner for upwind
    Eigen::VectorXd alpha_mf(mesh_.n_faces);
    for (int fid = 0; fid < mesh_.n_faces; ++fid) {
        int owner = mesh_.faces[fid].owner;
        alpha_mf[fid] = mass_flux[fid] * alpha.values[owner];
    }
    convection_operator_upwind(mesh_, alpha_mf, system);

    // MUSCL deferred correction for 2nd-order convection (C4 fix)
    // Pass the raw mass_flux (not alpha_mf) so the r-ratio and |F| scaling in
    // muscl_deferred_correction are computed against the actual momentum flux,
    // not a pre-multiplied alpha-flux that distorts the limiter on coarse grids.
    if (convection_scheme == "muscl") {
        auto grad_phi = green_gauss_gradient(phi);
        auto dc = muscl_deferred_correction(
            mesh_, phi, mass_flux, grad_phi, muscl_limiter);
        for (int ci = 0; ci < n; ++ci) {
            system.add_source(ci, dc[ci]);
        }
    }

    // Temporal term
    if (time_scheme == "bdf2" && phi.old_values.has_value() && phi.old_old_values.has_value()) {
        // BDF2: (3/(2*dt)) * rho*alpha*V * phi = (4/(2*dt)) * rho*alpha*V * phi^n
        //                                       - (1/(2*dt)) * rho*alpha*V * phi^{n-1}
        for (int ci = 0; ci < n; ++ci) {
            double vol = mesh_.cells[ci].volume;
            double coeff = alpha.values[ci] * rho * vol;
            system.add_diagonal(ci, 1.5 * coeff / dt);
            system.add_source(ci, (2.0 * coeff / dt) * phi.old_values.value()[ci]
                                - (0.5 * coeff / dt) * phi.old_old_values.value()[ci]);
        }
    } else if (phi.old_values.has_value()) {
        for (int ci = 0; ci < n; ++ci) {
            double vol = mesh_.cells[ci].volume;
            double coeff = alpha.values[ci] * rho * vol / dt;
            system.add_diagonal(ci, coeff);
            system.add_source(ci, coeff * phi.old_values.value()[ci]);
        }
    }

    // Pressure gradient: -alpha * V * (grad_p)_comp
    // Using face-based gradient: (p_N - p_O) contribution per face
    for (int fid = 0; fid < mesh_.n_faces; ++fid) {
        const Face& face = mesh_.faces[fid];
        int o = face.owner;

        if (face.neighbour >= 0) {
            int nb = face.neighbour;
            double dp = p_.values[nb] - p_.values[o];
            // Use face-averaged alpha so the pressure force is conservative:
            // both owner and neighbour see the same alpha weight on this face.
            // Using cell-local alpha (al_o for owner, al_nb for neighbour) breaks
            // momentum conservation when alpha varies across the face (sharp gradients).
            double al_f = 0.5 * (alpha.values[o] + alpha.values[nb]);

            // Pressure gradient contribution to owner and neighbour
            double force_o = -al_f * dp * face.normal[comp] * face.area;
            double force_nb =  al_f * dp * face.normal[comp] * face.area;
            system.add_source(o, force_o);
            system.add_source(nb, force_nb);
        } else {
            // Boundary: check for Dirichlet pressure BC, else zero-gradient (dp=0)
            double p_b = p_.values[o];  // zero-gradient default
            auto cache_it = face_bc_cache_.find(fid);
            if (cache_it != face_bc_cache_.end()) {
                auto& [bname, li] = cache_it->second;
                if (bc_p_.count(bname) && bc_p_.at(bname).type == "dirichlet") {
                    auto bv_it = p_.boundary_values.find(bname);
                    if (bv_it != p_.boundary_values.end() &&
                        li < static_cast<int>(bv_it->second.size())) {
                        p_b = bv_it->second[li];
                    }
                }
            }
            double dp = p_b - p_.values[o];
            system.add_source(o, -alpha.values[o] * dp * face.normal[comp] * face.area);
        }
    }

    // Drag (implicit)
    for (int ci = 0; ci < n; ++ci) {
        double vol = mesh_.cells[ci].volume;
        system.add_diagonal(ci, K_drag[ci] * vol);
        system.add_source(ci, K_drag[ci] * vol * U_other.values(ci, comp));
    }

    // Gravity: alpha * rho * g (body force on this phase)
    // Buoyancy arises from the combination of this term and -alpha*grad(p).
    // In a hydrostatic liquid: -alpha_l*grad(p) + alpha_l*rho_l*g = 0
    // For gas: -alpha_g*grad(p) + alpha_g*rho_g*g = alpha_g*(rho_g-rho_l)*g (net upward)
    for (int ci = 0; ci < n; ++ci) {
        double vol = mesh_.cells[ci].volume;
        system.add_source(ci, alpha.values[ci] * rho * g[comp] * vol);
    }

    // Additional interfacial forces (liquid phase only)
    if (is_liquid) {
        if (enable_lift_force && sigma_surface > 0.0) {
            // Compute curl of liquid velocity (2D: curl_z = dv/dx - du/dy)
            ScalarField ux(mesh_, "ux");
            ux.values = U_l_.values.col(0);
            for (auto& [bname, bv] : U_l_.boundary_values)
                ux.boundary_values[bname] = bv.col(0);

            ScalarField uy(mesh_, "uy");
            uy.values = U_l_.values.col(1);
            for (auto& [bname, bv] : U_l_.boundary_values)
                uy.boundary_values[bname] = bv.col(1);

            auto grad_ux = green_gauss_gradient(ux);
            auto grad_uy = green_gauss_gradient(uy);

            Eigen::MatrixXd curl_Ul = Eigen::MatrixXd::Zero(n, ndim);
            if (ndim == 2) {
                for (int ci = 0; ci < n; ++ci) {
                    curl_Ul(ci, 0) = grad_uy(ci, 0) - grad_ux(ci, 1);
                }
            }

            auto F_lift = lift_force_tomiyama(
                alpha_g_.values, rho_l, rho_g,
                U_g_.values, U_l_.values, d_b, mu_l, sigma_surface, curl_Ul);
            for (int ci = 0; ci < n; ++ci) {
                system.add_source(ci, F_lift(ci) * mesh_.cells[ci].volume);
            }
        }

        if (enable_turbulent_dispersion) {
            auto grad_alpha = green_gauss_gradient(alpha_g_);
            Eigen::VectorXd mu_t = sato_bubble_induced_turbulence(
                alpha_g_.values, rho_l, U_g_.values, U_l_.values, d_b);
            auto F_td = turbulent_dispersion_burns(alpha_g_.values, mu_t, grad_alpha, C_td);
            for (int ci = 0; ci < n; ++ci) {
                system.add_source(ci, F_td(ci) * mesh_.cells[ci].volume);
            }
        }
    }

    // Surface tension (CSF)
    if (sigma_surface > 0.0) {
        CSFSurfaceTension csf(mesh_, sigma_surface);
        auto F_sigma = csf.compute_force(alpha_g_);
        for (int ci = 0; ci < n; ++ci) {
            system.add_source(ci, F_sigma(ci, comp) * mesh_.cells[ci].volume);
        }
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

            // For Dirichlet pressure BC (outlet): enforce p' = 0 at the outlet
            // boundary using the large-number method. Without this, outlet cells
            // have no constraint on p' and the correction can grow unboundedly,
            // causing velocity blowup through the velocity correction step.
            auto cache_it = face_bc_cache_.find(fid);
            if (cache_it != face_bc_cache_.end()) {
                auto& [bname, li] = cache_it->second;
                if (bc_p_.count(bname) && bc_p_.at(bname).type == "dirichlet") {
                    // Estimate face diffusion length for coefficient scaling
                    double d_Pf = (face.center - mesh_.cells[o].center).norm();
                    if (d_Pf < 1e-30) d_Pf = 1e-3;
                    double al_f = alpha_l_.values[o];
                    double ag_f = alpha_g_.values[o];
                    double aP_l_f = std::max(aP_l_[o], 1e-30);
                    double aP_g_f = std::max(aP_g_storage_[o], 1e-30);
                    double vol_f = mesh_.cells[o].volume;
                    double coeff_l = rho_l * al_f * vol_f / aP_l_f;
                    double coeff_g = rho_g * ag_f * vol_f / aP_g_f;
                    double coeff = (coeff_l + coeff_g) * face.area / d_Pf;
                    // p' = 0 at outlet: add coeff to diagonal, zero to source
                    system.add_diagonal(o, coeff);
                }
            }
        }
    }

    // Reference pressure fix — scaled to problem magnitude
    double diag_max = system.diag.cwiseAbs().maxCoeff();
    double p_ref_scale = std::max(diag_max * 1e3, 1.0);
    system.add_diagonal(0, p_ref_scale);

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

    // Correct velocities: U_k' = -V / aP_k * grad(p')
    // NOTE: alpha_k is NOT included here -- it is already accounted for in the
    // pressure correction equation coefficients (coeff_k = rho_k * alpha_k * V / aP_k).
    // Including alpha_k again would suppress the velocity correction in cells with
    // small volume fraction, preventing the gas phase from developing buoyancy-driven
    // upward velocity in dilute regions.
#pragma omp parallel for schedule(static)
    for (int ci = 0; ci < n; ++ci) {
        double vol = mesh_.cells[ci].volume;
        for (int d = 0; d < ndim; ++d) {
            double corr_l = -vol / std::max(aP_l_[ci], 1e-20) * grad_pp(ci, d);
            double corr_g = -vol / std::max(aP_g_storage_[ci], 1e-20) * grad_pp(ci, d);
            U_l_.values(ci, d) += alpha_p * corr_l;
            U_g_.values(ci, d) += alpha_p * corr_g;
        }
    }

    // Velocity NaN/Inf cleanup with user-configurable limits
    for (auto* U_field : {&U_l_, &U_g_}) {
#pragma omp parallel for schedule(static)
        for (int ci = 0; ci < n; ++ci) {
            for (int d = 0; d < ndim; ++d) {
                double& v = U_field->values(ci, d);
                if (!std::isfinite(v)) v = 0.0;
                v = std::clamp(v, -U_max, U_max);
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
    if (time_scheme == "bdf2" && alpha_g_.old_values.has_value() && alpha_g_.old_old_values.has_value()) {
        temporal_operator_bdf2(mesh_, rho_g, dt,
                               alpha_g_.old_values.value(),
                               alpha_g_.old_old_values.value(), system);
    } else if (alpha_g_.old_values.has_value()) {
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

    // Physical limits: clip to [alpha_min_floor, alpha_max].
    // The lower bound 1e-20 prevents subnormal floating-point values in cells
    // that are effectively pure liquid (alpha_g ≈ 0). Without this floor, the
    // volume fraction equation can drive alpha_g to subnormal values (~1e-220)
    // in pure-liquid cells, which triggers false divergence detection and causes
    // downstream division issues in drag and pressure correction terms.
    constexpr double alpha_min_floor = 1e-20;
#pragma omp parallel for schedule(static)
    for (int ci = 0; ci < n; ++ci) {
        alpha_g_.values[ci] = std::clamp(alpha_g_.values[ci], alpha_min_floor, alpha_max);
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
    if (n_nonorth_correctors > 0) {
        diffusion_operator_corrected(mesh_, gamma, T, system, n_nonorth_correctors);
    } else {
        diffusion_operator(mesh_, gamma, system);
    }

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
    if (time_scheme == "bdf2" && T.old_values.has_value() && T.old_old_values.has_value()) {
        for (int ci = 0; ci < n; ++ci) {
            double vol = mesh_.cells[ci].volume;
            double coeff = alpha.values[ci] * rho * cp * vol;
            system.add_diagonal(ci, 1.5 * coeff / dt_local);
            system.add_source(ci, (2.0 * coeff / dt_local) * T.old_values.value()[ci]
                                - (0.5 * coeff / dt_local) * T.old_old_values.value()[ci]);
        }
    } else if (T.old_values.has_value()) {
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

    // NaN protection and temperature clipping (user-configurable limits)
    if (T.values.hasNaN() || !T.values.allFinite()) {
        T.values = T_old_vals;
    } else {
        for (int ci = 0; ci < n; ++ci) {
            T.values[ci] = std::clamp(T.values[ci], T_min, T_max);
        }
    }

    // Residual
    Eigen::SparseMatrix<double> A = system.to_sparse();
    Eigen::VectorXd r = A * T.values - system.rhs;
    double b_norm = std::max(system.rhs.norm(), 1e-15);
    return r.norm() / b_norm;
}

} // namespace twofluid
