#include "twofluid/conjugate_ht.hpp"
#include "twofluid/fvm_operators.hpp"
#include "twofluid/interpolation.hpp"
#include "twofluid/linear_solver.hpp"
#include <cmath>
#include <iostream>
#include <numeric>

namespace twofluid {

CHTCoupling::CHTCoupling(FVMesh& mesh,
                           const std::vector<int>& fluid_cells,
                           const std::vector<int>& solid_cells)
    : mesh_(mesh),
      fluid_cells_(fluid_cells.begin(), fluid_cells.end()),
      solid_cells_(solid_cells.begin(), solid_cells.end()),
      fluid_solver_(mesh, 998.2, 1.003e-3),
      solid_solver_(mesh, solid_cells),
      T_fluid_(mesh, "T_fluid") {
    T_fluid_.set_uniform(300.0);
    find_interface_faces();
}

void CHTCoupling::find_interface_faces() {
    interface_faces_.clear();
    for (int fid = 0; fid < mesh_.n_faces; ++fid) {
        const Face& face = mesh_.faces[fid];
        if (face.neighbour < 0) continue;
        int owner = face.owner;
        int nb = face.neighbour;

        if (fluid_cells_.count(owner) && solid_cells_.count(nb)) {
            interface_faces_.emplace_back(fid, owner, nb);
        } else if (solid_cells_.count(owner) && fluid_cells_.count(nb)) {
            interface_faces_.emplace_back(fid, nb, owner);
        }
    }
}

CHTResult CHTCoupling::solve_steady() {
    // 1) Solve fluid flow
    std::cout << "  [CHT] Solving fluid flow...\n";
    auto flow_result = fluid_solver_.solve_steady();
    std::cout << "  [CHT] Flow converged: " << flow_result.converged
              << ", iterations: " << flow_result.iterations << "\n";

    // 2) CHT iteration
    for (int cht_iter = 0; cht_iter < max_cht_iter; ++cht_iter) {
        auto T_int_old = get_interface_temperatures();

        solve_fluid_energy();

        auto q_interface = compute_interface_heat_flux();
        apply_interface_bc_to_solid(q_interface);
        solid_solver_.solve_steady();

        auto T_int_new = get_interface_temperatures();
        apply_interface_bc_to_fluid(T_int_new);

        // Convergence check
        if (!T_int_old.empty() && !T_int_new.empty()) {
            double change = 0.0, norm = 0.0;
            for (size_t i = 0; i < T_int_new.size(); ++i) {
                double diff = T_int_new[i] - T_int_old[i];
                change += diff * diff;
                norm += T_int_new[i] * T_int_new[i];
            }
            change = std::sqrt(change);
            norm = std::max(std::sqrt(norm), 1e-15);

            if (change / norm < tol_cht) {
                double T_avg = 0.0;
                for (double t : T_int_new) T_avg += t;
                T_avg /= T_int_new.size();

                double q_avg = 0.0;
                for (int i = 0; i < q_interface.size(); ++i)
                    q_avg += std::abs(q_interface(i));
                q_avg /= q_interface.size();

                return {true, cht_iter + 1, T_avg, q_avg};
            }
        }
    }

    auto T_int = get_interface_temperatures();
    double T_avg = T_int.empty() ? 300.0
        : std::accumulate(T_int.begin(), T_int.end(), 0.0) / T_int.size();
    return {false, max_cht_iter, T_avg, 0.0};
}

void CHTCoupling::solve_fluid_energy() {
    int n = mesh_.n_cells;
    FVMSystem system(n);

    ScalarField gamma(mesh_, "k_fluid");
    gamma.set_uniform(k_f);
    diffusion_operator(mesh_, gamma, system);

    // Convection: mass_flux * cp
    Eigen::VectorXd mass_flux = compute_mass_flux(
        fluid_solver_.velocity(), rho_f, mesh_);
    Eigen::VectorXd mf_cp = mass_flux * cp_f;
    convection_operator_upwind(mesh_, mf_cp, system);

    // Boundary conditions
    std::unordered_map<std::string, BoundaryCondition> bc_T;
    for (auto& [bname, fids] : mesh_.boundary_patches) {
        if (bname.find("inlet") != std::string::npos) {
            bc_T[bname] = {"dirichlet", 0.0};
            T_fluid_.set_boundary(bname, 300.0);
        } else {
            bc_T[bname] = {"zero_gradient", 0.0};
        }
    }
    apply_boundary_conditions(mesh_, T_fluid_, gamma, mf_cp, system, bc_T);
    under_relax(system, T_fluid_, 0.8);

    T_fluid_.values = solve_linear_system(system, T_fluid_.values, "bicgstab");
}

Eigen::VectorXd CHTCoupling::compute_interface_heat_flux() {
    int nf = static_cast<int>(interface_faces_.size());
    Eigen::VectorXd q(nf);

    for (int idx = 0; idx < nf; ++idx) {
        auto [fid, fc, sc] = interface_faces_[idx];
        double T_f = T_fluid_.values(fc);
        double T_s = solid_solver_.T.values(sc);

        Vec3 diff = mesh_.cells[fc].center - mesh_.cells[sc].center;
        double d = diff.head(mesh_.ndim).norm();
        if (d < 1e-30) { q(idx) = 0.0; continue; }

        double k_eff = 2.0 * k_f * solid_solver_.k_s
                       / (k_f + solid_solver_.k_s);
        q(idx) = k_eff * (T_s - T_f) / d;
    }
    return q;
}

std::vector<double> CHTCoupling::get_interface_temperatures() {
    std::vector<double> temps;
    temps.reserve(interface_faces_.size());

    for (auto& [fid, fc, sc] : interface_faces_) {
        double T_f = T_fluid_.values(fc);
        double T_s = solid_solver_.T.values(sc);
        double T_int = (k_f * T_f + solid_solver_.k_s * T_s)
                       / (k_f + solid_solver_.k_s);
        temps.push_back(T_int);
    }
    return temps;
}

void CHTCoupling::apply_interface_bc_to_solid(const Eigen::VectorXd& q) {
    for (int idx = 0; idx < static_cast<int>(interface_faces_.size()); ++idx) {
        auto [fid, fc, sc] = interface_faces_[idx];
        double vol = mesh_.cells[sc].volume;
        if (vol > 1e-30) {
            solid_solver_.q_vol(sc) = q(idx) * mesh_.faces[fid].area / vol;
        }
    }
}

void CHTCoupling::apply_interface_bc_to_fluid(const std::vector<double>& T_int) {
    for (int idx = 0; idx < static_cast<int>(interface_faces_.size()); ++idx) {
        if (idx >= static_cast<int>(T_int.size())) break;
        auto [fid, fc, sc] = interface_faces_[idx];
        T_fluid_.values(fc) = alpha_cht * T_int[idx]
                               + (1.0 - alpha_cht) * T_fluid_.values(fc);
    }
}

} // namespace twofluid
