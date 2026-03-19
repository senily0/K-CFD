#pragma once

#include <vector>
#include <unordered_map>
#include <Eigen/Dense>
#include "twofluid/mesh.hpp"
#include "twofluid/fields.hpp"
#include "twofluid/fvm_operators.hpp"

namespace twofluid {

struct SolidSolveResult {
    bool converged;
    int iterations;
    double T_max, T_min;
};

class SolidConductionSolver {
public:
    SolidConductionSolver(FVMesh& mesh, const std::vector<int>& cell_ids = {});

    double rho = 8960.0;
    double cp = 385.0;
    double k_s = 401.0;

    ScalarField T;
    Eigen::VectorXd q_vol;
    std::unordered_map<std::string, BoundaryCondition> bc_T;

    double alpha_T = 0.9;
    double dt = 0.01;

    void set_material(double rho_in, double cp_in, double k_in);
    void set_heat_source(double q, const std::vector<int>& cells = {});

    SolidSolveResult solve_steady(int max_iter = 200, double tol = 1e-6);
    void solve_one_step(double dt_in = -1.0);

private:
    FVMesh& mesh_;
    std::vector<int> cell_ids_;
};

} // namespace twofluid
