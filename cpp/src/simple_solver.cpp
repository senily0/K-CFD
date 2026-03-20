#include "twofluid/simple_solver.hpp"
#include "twofluid/gradient.hpp"
#include "twofluid/linear_solver.hpp"
#include "twofluid/fvm_operators.hpp"

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

SIMPLESolver::SIMPLESolver(FVMesh& mesh, double rho, double mu)
    : mesh_(mesh), rho_(rho), mu_(mu),
      U_(mesh, "velocity"), p_(mesh, "pressure") {
    int n = mesh.n_cells;
    int ndim = mesh.ndim;
    for (int c = 0; c < ndim; ++c) {
        aP_[c] = Eigen::VectorXd::Ones(n);
    }
    build_face_bc_cache();
}

// ---------------------------------------------------------------------------
// Boundary condition setters
// ---------------------------------------------------------------------------

void SIMPLESolver::set_inlet(const std::string& patch,
                              const Eigen::MatrixXd& U_vals) {
    if (mesh_.boundary_patches.find(patch) == mesh_.boundary_patches.end())
        return;
    U_.set_boundary(patch, U_vals);
    bc_u_[patch] = {"dirichlet"};
    bc_p_[patch] = {"zero_gradient"};
    build_face_bc_cache();
}

void SIMPLESolver::set_wall(const std::string& patch) {
    if (mesh_.boundary_patches.find(patch) == mesh_.boundary_patches.end())
        return;
    // Wall: zero velocity (Dirichlet), zero-gradient pressure
    int ndim = mesh_.ndim;
    int nf = static_cast<int>(mesh_.boundary_patches[patch].size());
    Eigen::MatrixXd zero_u = Eigen::MatrixXd::Zero(nf, ndim);
    U_.set_boundary(patch, zero_u);
    bc_u_[patch] = {"dirichlet"};
    bc_p_[patch] = {"zero_gradient"};
    wall_patches_.push_back(patch);
    build_face_bc_cache();
}

void SIMPLESolver::enable_turbulence(double k_init, double eps_init) {
    turb_ = std::make_unique<KEpsilonModel>(mesh_, rho_, mu_);
    turb_->initialize(k_init, eps_init);
}

void SIMPLESolver::set_outlet(const std::string& patch, double p_val) {
    if (mesh_.boundary_patches.find(patch) == mesh_.boundary_patches.end())
        return;
    p_.set_boundary(patch, p_val);
    bc_u_[patch] = {"zero_gradient"};
    bc_p_[patch] = {"dirichlet"};
    build_face_bc_cache();
}

// ---------------------------------------------------------------------------
// Face BC cache
// ---------------------------------------------------------------------------

void SIMPLESolver::build_face_bc_cache() {
    face_bc_cache_.clear();
    for (const auto& [bname, fids] : mesh_.boundary_patches) {
        for (int li = 0; li < static_cast<int>(fids.size()); ++li) {
            face_bc_cache_[fids[li]] = {bname, li};
        }
    }
}

// ---------------------------------------------------------------------------
// Geometric interpolation weight
// ---------------------------------------------------------------------------

double SIMPLESolver::gc(int fid) const {
    const Face& face = mesh_.faces[fid];
    const Eigen::Vector3d& xO = mesh_.cells[face.owner].center;
    const Eigen::Vector3d& xN = mesh_.cells[face.neighbour].center;
    const Eigen::Vector3d& xF = face.center;
    double dO = (xF - xO).norm();
    double dN = (xF - xN).norm();
    double t = dO + dN;
    return (t > 1e-30) ? dN / t : 0.5;
}

// ---------------------------------------------------------------------------
// Face mass flux
// ---------------------------------------------------------------------------

Eigen::VectorXd SIMPLESolver::compute_face_mass_flux() {
    int nf = mesh_.n_faces;
    int ndim = mesh_.ndim;
    Eigen::VectorXd mf = Eigen::VectorXd::Zero(nf);

    // Compute cell-center pressure gradient for Rhie-Chow interpolation
    grad_p_ = green_gauss_gradient(p_);

    for (int fid = 0; fid < nf; ++fid) {
        const Face& face = mesh_.faces[fid];
        int o = face.owner;
        Eigen::VectorXd uf;

        if (face.neighbour >= 0) {
            int nb = face.neighbour;
            double w = gc(fid);
            // Linear interpolation of velocity
            uf = w * U_.values.row(o).head(ndim).transpose()
                 + (1.0 - w) * U_.values.row(nb).head(ndim).transpose();

            // Rhie-Chow momentum interpolation (skip first iteration when aP is not yet meaningful)
            if (aP_.count(0) && aP_[0].size() == mesh_.n_cells
                && aP_[0].maxCoeff() > 1.1) {  // aP initialized to 1.0; skip until real values
                double aP_avg_o = 0.0, aP_avg_nb = 0.0;
                for (int c = 0; c < ndim; ++c) {
                    aP_avg_o += aP_[c](o);
                    aP_avg_nb += aP_[c](nb);
                }
                aP_avg_o = std::max(aP_avg_o / ndim, 1e-30);
                aP_avg_nb = std::max(aP_avg_nb / ndim, 1e-30);

                double vol_o = mesh_.cells[o].volume;
                double vol_nb = mesh_.cells[nb].volume;
                double dP_o = vol_o / aP_avg_o;
                double dP_nb = vol_nb / aP_avg_nb;
                double dP_f = w * dP_o + (1.0 - w) * dP_nb;

                Eigen::Vector3d d_vec = mesh_.cells[nb].center - mesh_.cells[o].center;
                double d_mag = d_vec.head(ndim).norm();
                if (d_mag > 1e-30) {
                    // Compact face pressure gradient (scalar along d_vec direction)
                    double dp_compact = (p_.values[nb] - p_.values[o]) / d_mag;

                    // Interpolated cell-center pressure gradient dotted with unit d_vec
                    double dp_interp = 0.0;
                    for (int dd = 0; dd < ndim; ++dd) {
                        double n_comp = d_vec[dd] / d_mag;
                        double grad_p_o = grad_p_(o, dd);
                        double grad_p_nb = grad_p_(nb, dd);
                        double grad_p_f = w * grad_p_o + (1.0 - w) * grad_p_nb;
                        dp_interp += grad_p_f * n_comp;
                    }

                    // Rhie-Chow correction: difference between compact and interpolated
                    for (int dd = 0; dd < ndim; ++dd) {
                        double n_comp = d_vec[dd] / d_mag;
                        uf[dd] -= dP_f * (dp_compact - dp_interp) * n_comp;
                    }
                }
            }
        } else {
            auto it = face_bc_cache_.find(fid);
            if (it != face_bc_cache_.end()) {
                const auto& [bname, li] = it->second;
                auto bc_it = bc_u_.find(bname);
                std::string bc_type = (bc_it != bc_u_.end()) ? bc_it->second.type : "zero_gradient";
                if (bc_type == "dirichlet") {
                    auto bv_it = U_.boundary_values.find(bname);
                    if (bv_it != U_.boundary_values.end()) {
                        uf = bv_it->second.row(li).head(ndim).transpose();
                    } else {
                        uf = U_.values.row(o).head(ndim).transpose();
                    }
                } else {
                    uf = U_.values.row(o).head(ndim).transpose();
                }
            } else {
                uf = U_.values.row(o).head(ndim).transpose();
            }
        }

        double dot = 0.0;
        for (int d = 0; d < ndim; ++d) {
            dot += uf[d] * face.normal[d];
        }
        mf[fid] = rho_ * dot * face.area;
    }

    return mf;
}

// ---------------------------------------------------------------------------
// Momentum equation (one velocity component)
// ---------------------------------------------------------------------------

double SIMPLESolver::solve_momentum(int comp, const Eigen::VectorXd& mf) {
    int n = mesh_.n_cells;
    int ndim = mesh_.ndim;
    double mu = mu_;

    // Effective viscosity: mu + mu_t when turbulence is enabled
    Eigen::VectorXd mu_eff_vec;
    if (turb_) {
        Eigen::VectorXd mu_t = turb_->get_mu_t();
        mu_eff_vec.resize(n);
        for (int ci = 0; ci < n; ++ci) {
            mu_eff_vec(ci) = mu_ + mu_t(ci);
        }
    }

    Eigen::VectorXd aP_local = Eigen::VectorXd::Zero(n);
    std::vector<int> aN_r, aN_c;
    std::vector<double> aN_v;
    Eigen::VectorXd b = Eigen::VectorXd::Zero(n);

    // Reserve space for off-diagonal entries
    aN_r.reserve(mesh_.n_faces * 2);
    aN_c.reserve(mesh_.n_faces * 2);
    aN_v.reserve(mesh_.n_faces * 2);

    for (int fid = 0; fid < mesh_.n_faces; ++fid) {
        const Face& face = mesh_.faces[fid];
        int o = face.owner;
        double F = mf[fid];

        if (face.neighbour >= 0) {
            int nb = face.neighbour;
            Eigen::Vector3d d_vec = mesh_.cells[nb].center - mesh_.cells[o].center;
            double d = d_vec.norm();
            if (d < 1e-30) continue;

            double mu_o = turb_ ? mu_eff_vec(o) : mu;
            double mu_n = turb_ ? mu_eff_vec(nb) : mu;

            // Non-orthogonal correction: use orthogonal distance
            double d_orth = std::abs(d_vec.dot(face.normal));
            d_orth = std::max(d_orth, 0.1 * d);  // safety limit
            double Df = 2.0 * mu_o * mu_n / std::max(mu_o + mu_n, 1e-30) * face.area / d_orth;

            // Owner equation
            aP_local[o] += Df + std::max(F, 0.0);
            aN_r.push_back(o); aN_c.push_back(nb);
            aN_v.push_back(-(Df + std::max(-F, 0.0)));

            // Neighbour equation (flux direction reversed)
            aP_local[nb] += Df + std::max(-F, 0.0);
            aN_r.push_back(nb); aN_c.push_back(o);
            aN_v.push_back(-(Df + std::max(F, 0.0)));

        } else {
            auto it = face_bc_cache_.find(fid);
            if (it == face_bc_cache_.end()) continue;
            const auto& [bname, li] = it->second;

            auto bc_it = bc_u_.find(bname);
            std::string bc_type = (bc_it != bc_u_.end()) ? bc_it->second.type : "zero_gradient";

            double d = (face.center - mesh_.cells[o].center).norm();
            if (d < 1e-30) continue;
            double mu_o = turb_ ? mu_eff_vec(o) : mu;
            double Df = mu_o * face.area / d;

            if (bc_type == "dirichlet") {
                double phi_b = 0.0;
                auto bv_it = U_.boundary_values.find(bname);
                if (bv_it != U_.boundary_values.end()) {
                    phi_b = bv_it->second(li, comp);
                }
                // Diffusion
                aP_local[o] += Df;
                b[o] += Df * phi_b;
                // Convection boundary
                if (F >= 0) {
                    aP_local[o] += F;
                } else {
                    b[o] += (-F) * phi_b;
                }
            } else {
                // zero_gradient: no diffusion flux, convection uses cell value
                if (F >= 0) {
                    aP_local[o] += F;
                } else {
                    aP_local[o] += (-F);
                }
            }
        }
    }

    // Pressure gradient source: -sum_f p_f * n_comp * A_f
    for (int fid = 0; fid < mesh_.n_faces; ++fid) {
        const Face& face = mesh_.faces[fid];
        int o = face.owner;

        if (face.neighbour >= 0) {
            int nb = face.neighbour;
            double pf = 0.5 * (p_.values[o] + p_.values[nb]);
            double src = pf * face.normal[comp] * face.area;
            b[o] -= src;
            b[nb] += src;
        } else {
            double pf;
            auto it = face_bc_cache_.find(fid);
            if (it != face_bc_cache_.end()) {
                const auto& [bname, li] = it->second;
                auto pbc_it = bc_p_.find(bname);
                std::string pbc_type = (pbc_it != bc_p_.end()) ? pbc_it->second.type : "zero_gradient";
                if (pbc_type == "dirichlet") {
                    auto bv_it = p_.boundary_values.find(bname);
                    pf = (bv_it != p_.boundary_values.end()) ? bv_it->second[li] : p_.values[o];
                } else {
                    pf = p_.values[o];
                }
            } else {
                pf = p_.values[o];
            }
            b[o] -= pf * face.normal[comp] * face.area;
        }
    }

    // Temporal term: rho*V/dt * (phi - phi_old)
    if (dt_ > 0.0 && U_.old_values.has_value()) {
        for (int ci = 0; ci < n; ++ci) {
            double coeff = rho_ * mesh_.cells[ci].volume / dt_;
            aP_local[ci] += coeff;
            b[ci] += coeff * U_.old_values.value()(ci, comp);
        }
    }

    // Under-relaxation
    double alpha = alpha_u;
    Eigen::VectorXd phi_old = U_.values.col(comp);
    b += (1.0 - alpha) / alpha * aP_local.cwiseProduct(phi_old);
    aP_local /= alpha;

    // Save unrelaxed aP for pressure correction
    aP_[comp] = aP_local * alpha;

    // Assemble sparse matrix and solve
    int total_entries = static_cast<int>(aN_r.size()) + n;
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(total_entries);

    for (size_t k = 0; k < aN_r.size(); ++k) {
        triplets.emplace_back(aN_r[k], aN_c[k], aN_v[k]);
    }
    for (int i = 0; i < n; ++i) {
        triplets.emplace_back(i, i, aP_local[i]);
    }

    Eigen::SparseMatrix<double> A(n, n);
    A.setFromTriplets(triplets.begin(), triplets.end());
    A.makeCompressed();

    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.analyzePattern(A);
    solver.factorize(A);

    Eigen::VectorXd phi;
    if (solver.info() == Eigen::Success) {
        phi = solver.solve(b);
    } else {
        phi = phi_old;
    }

    // Check for NaN
    if (phi.hasNaN()) {
        phi = phi_old;
    }

    U_.values.col(comp) = phi;

    // Compute residual
    Eigen::VectorXd r = A * phi - b;
    double b_norm = b.norm();
    return r.norm() / std::max(b_norm, 1e-15);
}

// ---------------------------------------------------------------------------
// Pressure correction equation
// ---------------------------------------------------------------------------

double SIMPLESolver::solve_pressure_correction(const Eigen::VectorXd& mf) {
    int n = mesh_.n_cells;
    int ndim = mesh_.ndim;

    // d_P = V_P / aP_avg for each cell
    Eigen::VectorXd aP_sum = Eigen::VectorXd::Zero(n);
    for (int c = 0; c < ndim; ++c) {
        aP_sum += aP_[c];
    }
    Eigen::VectorXd aP_mom = (aP_sum / ndim).cwiseMax(1e-30);
    Eigen::VectorXd dP(n);
    for (int ci = 0; ci < n; ++ci) {
        dP[ci] = mesh_.cells[ci].volume / aP_mom[ci];
    }

    // Build pressure correction system
    std::vector<int> rows, cols;
    std::vector<double> vals;
    Eigen::VectorXd aP_pp = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(n);

    rows.reserve(mesh_.n_faces * 2 + n);
    cols.reserve(mesh_.n_faces * 2 + n);
    vals.reserve(mesh_.n_faces * 2 + n);

    for (int fid = 0; fid < mesh_.n_faces; ++fid) {
        const Face& face = mesh_.faces[fid];
        int o = face.owner;

        if (face.neighbour >= 0) {
            int nb = face.neighbour;
            Eigen::Vector3d d_vec = mesh_.cells[nb].center - mesh_.cells[o].center;
            double d = d_vec.norm();
            if (d < 1e-30) continue;

            double d_orth = std::abs(d_vec.dot(face.normal));
            d_orth = std::max(d_orth, 0.1 * d);
            double df = rho_ * 0.5 * (dP[o] + dP[nb]) * face.area / d_orth;

            aP_pp[o] += df;
            aP_pp[nb] += df;
            rows.push_back(o); cols.push_back(nb); vals.push_back(-df);
            rows.push_back(nb); cols.push_back(o); vals.push_back(-df);

            b[o] -= mf[fid];
            b[nb] += mf[fid];
        } else {
            // Boundary mass imbalance
            b[o] -= mf[fid];

            // Dirichlet pressure boundaries: p' = 0 at boundary
            auto it = face_bc_cache_.find(fid);
            if (it != face_bc_cache_.end()) {
                const auto& [bname, li] = it->second;
                auto pbc_it = bc_p_.find(bname);
                std::string pbc_type = (pbc_it != bc_p_.end()) ? pbc_it->second.type : "zero_gradient";
                if (pbc_type == "dirichlet") {
                    double d = (face.center - mesh_.cells[o].center).norm();
                    if (d > 1e-30) {
                        double df = rho_ * dP[o] * face.area / d;
                        aP_pp[o] += df;
                    }
                }
            }
        }
    }

    // Diagonal
    for (int ci = 0; ci < n; ++ci) {
        rows.push_back(ci); cols.push_back(ci); vals.push_back(aP_pp[ci]);
    }

    // Reference pressure if no Dirichlet p BC
    bool has_p_dir = false;
    for (const auto& [name, info] : bc_p_) {
        if (info.type == "dirichlet") { has_p_dir = true; break; }
    }
    if (!has_p_dir) {
        // Zero out row 0 and set diagonal = 1, b = 0
        for (size_t i = 0; i < rows.size(); ++i) {
            if (rows[i] == 0) {
                vals[i] = 0.0;
            }
        }
        rows.push_back(0); cols.push_back(0); vals.push_back(1.0);
        b[0] = 0.0;
    }

    // Assemble and solve
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(rows.size());
    for (size_t k = 0; k < rows.size(); ++k) {
        triplets.emplace_back(rows[k], cols[k], vals[k]);
    }

    Eigen::SparseMatrix<double> A(n, n);
    A.setFromTriplets(triplets.begin(), triplets.end());
    A.makeCompressed();

    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.analyzePattern(A);
    solver.factorize(A);

    Eigen::VectorXd pp;
    if (solver.info() == Eigen::Success) {
        pp = solver.solve(b);
    } else {
        pp = Eigen::VectorXd::Zero(n);
    }

    if (pp.hasNaN()) {
        pp = Eigen::VectorXd::Zero(n);
    }

    // Velocity correction via Green-Gauss gradient of p'
    ScalarField pp_field(mesh_, "pp");
    pp_field.values = pp;
    for (const auto& [bname, fids] : mesh_.boundary_patches) {
        auto pbc_it = bc_p_.find(bname);
        std::string pbc_type = (pbc_it != bc_p_.end()) ? pbc_it->second.type : "zero_gradient";
        if (pbc_type == "dirichlet") {
            pp_field.set_boundary(bname, 0.0);
        } else {
            auto bv_it = pp_field.boundary_values.find(bname);
            if (bv_it != pp_field.boundary_values.end()) {
                for (int li = 0; li < static_cast<int>(fids.size()); ++li) {
                    bv_it->second[li] = pp[mesh_.faces[fids[li]].owner];
                }
            }
        }
    }

    Eigen::MatrixXd grad_pp = green_gauss_gradient(pp_field);

    for (int ci = 0; ci < n; ++ci) {
        for (int comp = 0; comp < ndim; ++comp) {
            U_.values(ci, comp) -= dP[ci] * grad_pp(ci, comp);
        }
    }

    // Pressure update
    p_.values += alpha_p * pp;

    return pp.norm() / std::max(p_.values.norm() + 1e-10, 1e-10);
}

// ---------------------------------------------------------------------------
// Main SIMPLE loop
// ---------------------------------------------------------------------------

SolveResult SIMPLESolver::solve_steady() {
    auto t_start = std::chrono::high_resolution_clock::now();

    int n = mesh_.n_cells;
    int ndim = mesh_.ndim;
    for (int c = 0; c < ndim; ++c) {
        aP_[c] = Eigen::VectorXd::Ones(n);
    }

    std::vector<double> residuals;
    double res0 = -1.0;

    for (int it = 0; it < max_iter; ++it) {
        Eigen::VectorXd mf = compute_face_mass_flux();

        // Solve momentum for each component
        std::vector<double> res_mom;
        for (int comp = 0; comp < ndim; ++comp) {
            res_mom.push_back(solve_momentum(comp, mf));
        }

        // Recompute mass flux with updated velocity
        mf = compute_face_mass_flux();

        // Pressure correction
        double res_p = solve_pressure_correction(mf);

        // Turbulence equations (if enabled)
        if (turb_) {
            Eigen::VectorXd mf_turb = compute_face_mass_flux();
            // Build BC map: walls get zero_gradient, others get zero_gradient too
            std::unordered_map<std::string, std::string> turb_bc;
            for (const auto& [patch, info] : bc_u_) {
                turb_bc[patch] = "zero_gradient";
            }
            turb_->solve(U_, mf_turb, turb_bc);
            turb_->apply_wall_functions(U_, wall_patches_);
        }

        double res = std::max(*std::max_element(res_mom.begin(), res_mom.end()), res_p);

        // Normalize by initial residual
        if (res0 < 0.0 && res > 1e-30) {
            res0 = res;
        }
        double res_norm = (res0 > 1e-30) ? res / res0 : res;
        residuals.push_back(res_norm);

        if (it < 5 || it % 100 == 0) {
            std::cout << "    iter " << it << ": res=" << res_norm;
            for (int c = 0; c < ndim; ++c) {
                std::cout << " c" << c << "=" << res_mom[c];
            }
            std::cout << " p=" << res_p << std::endl;
        }

        if (res_norm < tol) {
            auto t_end = std::chrono::high_resolution_clock::now();
            double wall = std::chrono::duration<double>(t_end - t_start).count();
            return {true, it + 1, residuals, wall};
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double wall = std::chrono::duration<double>(t_end - t_start).count();
    return {false, max_iter, residuals, wall};
}

// ---------------------------------------------------------------------------
// PISO transient step
// ---------------------------------------------------------------------------

SolveResult SIMPLESolver::solve_transient_step(double dt) {
    auto t_start = std::chrono::high_resolution_clock::now();

    int n = mesh_.n_cells;
    int ndim = mesh_.ndim;
    for (int c = 0; c < ndim; ++c) {
        aP_[c] = Eigen::VectorXd::Ones(n);
    }

    std::vector<double> residuals;

    // Store old velocity for temporal term
    U_.store_old();

    // Momentum predictor: no under-relaxation for PISO
    double saved_alpha_u = alpha_u;
    alpha_u = 1.0;
    dt_ = dt;

    Eigen::VectorXd mf = compute_face_mass_flux();

    std::vector<double> res_mom;
    for (int comp = 0; comp < ndim; ++comp) {
        res_mom.push_back(solve_momentum(comp, mf));
    }

    alpha_u = saved_alpha_u;
    dt_ = 0.0;

    // PISO correction loop
    double res_p = 0.0;
    for (int corr = 0; corr < piso_correctors; ++corr) {
        mf = compute_face_mass_flux();
        res_p = solve_pressure_correction(mf);
    }

    double res = std::max(*std::max_element(res_mom.begin(), res_mom.end()), res_p);
    residuals.push_back(res);

    auto t_end = std::chrono::high_resolution_clock::now();
    double wall = std::chrono::duration<double>(t_end - t_start).count();

    // PISO single-pass always "converges" per time step
    return {true, 1, residuals, wall};
}

} // namespace twofluid
