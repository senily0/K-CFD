#pragma once

#include <set>
#include <tuple>
#include <vector>
#include <Eigen/Dense>
#include "twofluid/mesh.hpp"
#include "twofluid/fields.hpp"
#include "twofluid/simple_solver.hpp"
#include "twofluid/solid_conduction.hpp"

namespace twofluid {

struct CHTResult {
    bool converged;
    int iterations;
    double T_interface_avg;
    double heat_flux;
};

class CHTCoupling {
public:
    CHTCoupling(FVMesh& mesh, const std::vector<int>& fluid_cells,
                const std::vector<int>& solid_cells);

    double rho_f = 998.2, cp_f = 4182.0, k_f = 0.6, mu_f = 1.003e-3;
    int max_cht_iter = 50;
    double tol_cht = 1e-5;
    double alpha_cht = 0.5;

    CHTResult solve_steady();

    SIMPLESolver& fluid_solver() { return fluid_solver_; }
    SolidConductionSolver& solid_solver() { return solid_solver_; }
    ScalarField& T_fluid() { return T_fluid_; }

private:
    FVMesh& mesh_;
    std::set<int> fluid_cells_, solid_cells_;
    SIMPLESolver fluid_solver_;
    SolidConductionSolver solid_solver_;
    ScalarField T_fluid_;

    std::vector<std::tuple<int, int, int>> interface_faces_;

    void find_interface_faces();
    void solve_fluid_energy();
    Eigen::VectorXd compute_interface_heat_flux();
    std::vector<double> get_interface_temperatures();
    void apply_interface_bc_to_solid(const Eigen::VectorXd& q);
    void apply_interface_bc_to_fluid(const std::vector<double>& T_int);
};

} // namespace twofluid
