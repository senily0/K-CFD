#include "twofluid/solid_conduction.hpp"
#include "twofluid/linear_solver.hpp"
#include <cmath>

namespace twofluid {

SolidConductionSolver::SolidConductionSolver(
    FVMesh& mesh, const std::vector<int>& cell_ids)
    : mesh_(mesh), T(mesh, "temperature_solid"),
      q_vol(Eigen::VectorXd::Zero(mesh.n_cells)) {
    if (cell_ids.empty()) {
        cell_ids_.resize(mesh.n_cells);
        for (int i = 0; i < mesh.n_cells; ++i) cell_ids_[i] = i;
    } else {
        cell_ids_ = cell_ids;
    }
    T.set_uniform(300.0);
}

void SolidConductionSolver::set_material(double rho_in, double cp_in, double k_in) {
    rho = rho_in;
    cp = cp_in;
    k_s = k_in;
}

void SolidConductionSolver::set_heat_source(double q, const std::vector<int>& cells) {
    const auto& target = cells.empty() ? cell_ids_ : cells;
    for (int ci : target) {
        q_vol(ci) = q;
    }
}

SolidSolveResult SolidConductionSolver::solve_steady(int max_iter, double tol) {
    int n = mesh_.n_cells;

    for (int iter = 0; iter < max_iter; ++iter) {
        FVMSystem system(n);

        ScalarField gamma(mesh_, "k_solid");
        gamma.set_uniform(k_s);

        diffusion_operator(mesh_, gamma, system);
        source_term(mesh_, q_vol, system);

        Eigen::VectorXd mass_flux = Eigen::VectorXd::Zero(mesh_.n_faces);
        apply_boundary_conditions(mesh_, T, gamma, mass_flux, system, bc_T);
        under_relax(system, T, alpha_T);

        Eigen::VectorXd T_old = T.values;
        T.values = solve_linear_system(system, T.values, "direct");

        double change = (T.values - T_old).norm();
        double norm = std::max(T.values.norm(), 1e-15);
        double res = change / norm;

        if (res < tol) {
            double T_max = -1e30, T_min = 1e30;
            for (int ci : cell_ids_) {
                T_max = std::max(T_max, T.values(ci));
                T_min = std::min(T_min, T.values(ci));
            }
            return {true, iter + 1, T_max, T_min};
        }
    }

    double T_max = -1e30, T_min = 1e30;
    for (int ci : cell_ids_) {
        T_max = std::max(T_max, T.values(ci));
        T_min = std::min(T_min, T.values(ci));
    }
    return {false, max_iter, T_max, T_min};
}

void SolidConductionSolver::solve_one_step(double dt_in) {
    if (dt_in > 0.0) dt = dt_in;

    int n = mesh_.n_cells;
    FVMSystem system(n);

    T.store_old();

    ScalarField gamma(mesh_, "k_solid");
    gamma.set_uniform(k_s);
    diffusion_operator(mesh_, gamma, system);

    temporal_operator(mesh_, rho * cp, dt, *T.old_values, system);
    source_term(mesh_, q_vol, system);

    Eigen::VectorXd mass_flux = Eigen::VectorXd::Zero(mesh_.n_faces);
    apply_boundary_conditions(mesh_, T, gamma, mass_flux, system, bc_T);

    T.values = solve_linear_system(system, T.values, "direct");
}

} // namespace twofluid
