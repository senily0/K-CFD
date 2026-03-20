#include "twofluid/distributed_solver.hpp"
#include "twofluid/gradient.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <Eigen/Sparse>

namespace twofluid {

// ===================================================================
// Distributed BiCGSTAB
// ===================================================================

Eigen::VectorXd distributed_bicgstab(
    const Eigen::SparseMatrix<double>& A_local,
    const Eigen::VectorXd& b_local,
    const Eigen::VectorXd& x0,
    const DistributedMesh& dmesh,
    double tol,
    int maxiter)
{
    const int n = static_cast<int>(A_local.rows());
    const int n_owned = dmesh.n_owned;

    Eigen::VectorXd x = x0;
    if (x.size() != n) {
        x = Eigen::VectorXd::Zero(n);
    }

    // r = b - A*x
    Eigen::VectorXd r = b_local - A_local * x;

    // Exchange ghost values of r
    // (const_cast is safe here: we need to exchange in-place)
    const_cast<DistributedMesh&>(dmesh).exchange_scalar(r);

    Eigen::VectorXd r_hat = r;  // arbitrary, but r is a good choice

    double rho_old = 1.0, alpha = 1.0, omega = 1.0;
    Eigen::VectorXd v = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd p = Eigen::VectorXd::Zero(n);

    // Compute initial residual norm (global, owned cells only)
    double b_norm = dmesh.global_norm(b_local);
    if (b_norm < 1e-30) b_norm = 1.0;

    double r_norm = dmesh.global_norm(r);
    if (r_norm / b_norm < tol) return x;

    for (int iter = 0; iter < maxiter; ++iter) {
        double rho_new = dmesh.global_dot(r_hat, r);

        // Breakdown check
        if (std::abs(rho_new) < 1e-30) break;

        double beta = (rho_new / rho_old) * (alpha / omega);

        // p = r + beta * (p - omega * v)
        p = r + beta * (p - omega * v);
        const_cast<DistributedMesh&>(dmesh).exchange_scalar(p);

        // v = A * p
        v = A_local * p;
        const_cast<DistributedMesh&>(dmesh).exchange_scalar(v);

        double r_hat_v = dmesh.global_dot(r_hat, v);
        if (std::abs(r_hat_v) < 1e-30) break;

        alpha = rho_new / r_hat_v;

        // s = r - alpha * v
        Eigen::VectorXd s = r - alpha * v;

        // Check convergence on s
        double s_norm = dmesh.global_norm(s);
        if (s_norm / b_norm < tol) {
            // x += alpha * p (owned cells only, ghosts get exchanged)
            for (int i = 0; i < n_owned; ++i) {
                x[i] += alpha * p[i];
            }
            const_cast<DistributedMesh&>(dmesh).exchange_scalar(x);
            return x;
        }

        // t = A * s
        const_cast<DistributedMesh&>(dmesh).exchange_scalar(s);
        Eigen::VectorXd t = A_local * s;
        const_cast<DistributedMesh&>(dmesh).exchange_scalar(t);

        double t_dot_s = dmesh.global_dot(t, s);
        double t_dot_t = dmesh.global_dot(t, t);
        omega = (std::abs(t_dot_t) > 1e-30) ? t_dot_s / t_dot_t : 0.0;

        // x += alpha * p + omega * s (owned cells only)
        for (int i = 0; i < n_owned; ++i) {
            x[i] += alpha * p[i] + omega * s[i];
        }
        const_cast<DistributedMesh&>(dmesh).exchange_scalar(x);

        // r = s - omega * t
        r = s - omega * t;

        r_norm = dmesh.global_norm(r);
        if (r_norm / b_norm < tol) break;

        if (std::abs(omega) < 1e-30) break;

        rho_old = rho_new;
    }

    return x;
}

// ===================================================================
// DistributedSIMPLESolver
// ===================================================================

DistributedSIMPLESolver::DistributedSIMPLESolver(
    DistributedMesh& dmesh, double rho, double mu)
    : dmesh_(dmesh), mesh_(dmesh.local_mesh), rho_(rho), mu_(mu),
      U_(mesh_, "velocity"), p_(mesh_, "pressure")
{
    int n = mesh_.n_cells;  // n_owned + n_ghost
    int ndim = mesh_.ndim;
    for (int c = 0; c < ndim; ++c) {
        aP_[c] = Eigen::VectorXd::Ones(n);
    }
    build_face_bc_cache();
}

// ---------------------------------------------------------------------------
// Boundary condition setters
// ---------------------------------------------------------------------------

void DistributedSIMPLESolver::set_inlet(const std::string& patch,
                                         const Eigen::MatrixXd& U_vals) {
    if (mesh_.boundary_patches.find(patch) == mesh_.boundary_patches.end())
        return;
    U_.set_boundary(patch, U_vals);
    bc_u_[patch] = {"dirichlet"};
    bc_p_[patch] = {"zero_gradient"};
    build_face_bc_cache();
}

void DistributedSIMPLESolver::set_wall(const std::string& patch) {
    if (mesh_.boundary_patches.find(patch) == mesh_.boundary_patches.end())
        return;
    int ndim = mesh_.ndim;
    int nf = static_cast<int>(mesh_.boundary_patches[patch].size());
    Eigen::MatrixXd zero_u = Eigen::MatrixXd::Zero(nf, ndim);
    U_.set_boundary(patch, zero_u);
    bc_u_[patch] = {"dirichlet"};
    bc_p_[patch] = {"zero_gradient"};
    build_face_bc_cache();
}

void DistributedSIMPLESolver::set_outlet(const std::string& patch, double p_val) {
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

void DistributedSIMPLESolver::build_face_bc_cache() {
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

double DistributedSIMPLESolver::gc(int fid) const {
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
// Face mass flux (identical to SIMPLESolver but on the distributed mesh)
// ---------------------------------------------------------------------------

Eigen::VectorXd DistributedSIMPLESolver::compute_face_mass_flux() {
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
            uf = w * U_.values.row(o).head(ndim).transpose()
                 + (1.0 - w) * U_.values.row(nb).head(ndim).transpose();

            // Rhie-Chow momentum interpolation
            if (aP_.count(0) && aP_[0].size() == mesh_.n_cells
                && aP_[0].maxCoeff() > 1.1) {
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
                    double dp_compact = (p_.values[nb] - p_.values[o]) / d_mag;

                    double dp_interp = 0.0;
                    for (int dd = 0; dd < ndim; ++dd) {
                        double n_comp = d_vec[dd] / d_mag;
                        double grad_p_o = grad_p_(o, dd);
                        double grad_p_nb = grad_p_(nb, dd);
                        double grad_p_f = w * grad_p_o + (1.0 - w) * grad_p_nb;
                        dp_interp += grad_p_f * n_comp;
                    }

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
// Uses distributed_bicgstab for the linear solve.
// ---------------------------------------------------------------------------

double DistributedSIMPLESolver::solve_momentum(int comp, const Eigen::VectorXd& mf) {
    int n = mesh_.n_cells;  // n_owned + n_ghost
    int ndim = mesh_.ndim;
    double mu = mu_;

    Eigen::VectorXd aP_local = Eigen::VectorXd::Zero(n);
    std::vector<int> aN_r, aN_c;
    std::vector<double> aN_v;
    Eigen::VectorXd b = Eigen::VectorXd::Zero(n);

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

            double d_orth = std::abs(d_vec.dot(face.normal));
            d_orth = std::max(d_orth, 0.1 * d);
            double Df = mu * face.area / d_orth;

            // Owner equation
            aP_local[o] += Df + std::max(F, 0.0);
            aN_r.push_back(o); aN_c.push_back(nb);
            aN_v.push_back(-(Df + std::max(-F, 0.0)));

            // Neighbour equation
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
            double Df = mu * face.area / d;

            if (bc_type == "dirichlet") {
                double phi_b = 0.0;
                auto bv_it = U_.boundary_values.find(bname);
                if (bv_it != U_.boundary_values.end()) {
                    phi_b = bv_it->second(li, comp);
                }
                aP_local[o] += Df;
                b[o] += Df * phi_b;
                if (F >= 0) {
                    aP_local[o] += F;
                } else {
                    b[o] += (-F) * phi_b;
                }
            } else {
                if (F >= 0) {
                    aP_local[o] += F;
                } else {
                    aP_local[o] += (-F);
                }
            }
        }
    }

    // Pressure gradient source
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

    // Under-relaxation
    double alpha = alpha_u;
    Eigen::VectorXd phi_old = U_.values.col(comp);
    b += (1.0 - alpha) / alpha * aP_local.cwiseProduct(phi_old);
    aP_local /= alpha;

    // Save unrelaxed aP
    aP_[comp] = aP_local * alpha;

    // Assemble sparse matrix
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

    // Solve using distributed BiCGSTAB
    Eigen::VectorXd phi = distributed_bicgstab(A, b, phi_old, dmesh_, 1e-6, 500);

    if (phi.hasNaN()) {
        phi = phi_old;
    }

    U_.values.col(comp) = phi;

    // Compute residual (global)
    Eigen::VectorXd r = A * phi - b;
    double b_norm = dmesh_.global_norm(b);
    return dmesh_.global_norm(r) / std::max(b_norm, 1e-15);
}

// ---------------------------------------------------------------------------
// Pressure correction equation
// Uses distributed_bicgstab for the linear solve.
// ---------------------------------------------------------------------------

double DistributedSIMPLESolver::solve_pressure_correction(const Eigen::VectorXd& mf) {
    int n = mesh_.n_cells;
    int ndim = mesh_.ndim;

    // d_P = V_P / aP_avg
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
            b[o] -= mf[fid];

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

    // Reference pressure: if no Dirichlet p BC on this rank, we need to
    // handle the singular system. In the distributed case, we fix pressure
    // at cell 0 of rank 0.
    bool has_p_dir = false;
    for (const auto& [name, info] : bc_p_) {
        if (info.type == "dirichlet") { has_p_dir = true; break; }
    }

    // Determine globally if any rank has a pressure Dirichlet BC
    int local_has_p = has_p_dir ? 1 : 0;
    int global_has_p = local_has_p;
#ifdef USE_MPI
    MPI_Allreduce(&local_has_p, &global_has_p, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif

    if (!global_has_p) {
        // Fix pressure at global cell 0 (which is on some rank)
        // Each rank checks if it owns global cell 0
        auto it = dmesh_.global_to_local.find(0);
        if (it != dmesh_.global_to_local.end() && it->second < dmesh_.n_owned) {
            int ref_cell = it->second;
            for (size_t i = 0; i < rows.size(); ++i) {
                if (rows[i] == ref_cell) {
                    vals[i] = 0.0;
                }
            }
            rows.push_back(ref_cell); cols.push_back(ref_cell); vals.push_back(1.0);
            b[ref_cell] = 0.0;
        }
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

    Eigen::VectorXd pp = distributed_bicgstab(
        A, b, Eigen::VectorXd::Zero(n), dmesh_, 1e-6, 1000);

    if (pp.hasNaN()) {
        pp = Eigen::VectorXd::Zero(n);
    }

    // Exchange ghost values of pp
    dmesh_.exchange_scalar(pp);

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

SolveResult DistributedSIMPLESolver::solve_steady() {
    auto t_start = std::chrono::high_resolution_clock::now();

    int n = mesh_.n_cells;
    int ndim = mesh_.ndim;

    // Initialize aP
    for (int c = 0; c < ndim; ++c) {
        aP_[c] = Eigen::VectorXd::Ones(n);
    }

    std::vector<double> residuals;
    double res0 = -1.0;

    for (int it = 0; it < max_iter; ++it) {
        // Exchange ghost values before each iteration
        dmesh_.exchange_vector(U_.values);
        dmesh_.exchange_scalar(p_.values);

        Eigen::VectorXd mf = compute_face_mass_flux();

        // Momentum for each component
        std::vector<double> res_mom;
        for (int comp = 0; comp < ndim; ++comp) {
            res_mom.push_back(solve_momentum(comp, mf));
        }

        // Exchange after momentum solve
        dmesh_.exchange_vector(U_.values);

        // Recompute mass flux
        mf = compute_face_mass_flux();

        // Pressure correction
        double res_p = solve_pressure_correction(mf);

        // Exchange after correction
        dmesh_.exchange_vector(U_.values);
        dmesh_.exchange_scalar(p_.values);

        double res = std::max(*std::max_element(res_mom.begin(), res_mom.end()), res_p);

        // Use global max of raw residual for consistent normalization
        double global_res_raw = dmesh_.global_max(res);

        // Normalize by initial residual (globally consistent)
        if (res0 < 0.0 && global_res_raw > 1e-30) {
            res0 = global_res_raw;
        }
        double res_norm = (res0 > 1e-30) ? global_res_raw / res0 : global_res_raw;
        residuals.push_back(res_norm);

        if (it < 5 || it % 100 == 0) {
            // Only rank 0 prints (caller checks)
            int my_rank = 0;
#ifdef USE_MPI
            MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
#endif
            if (my_rank == 0) {
                std::cout << "    iter " << it << ": res=" << res_norm
                          << " p=" << global_res_raw << std::endl;
            }
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

} // namespace twofluid
