#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <Eigen/Dense>
#include "twofluid/mesh.hpp"
#include "twofluid/fields.hpp"
#include "twofluid/fvm_operators.hpp"
#include "twofluid/closure.hpp"
#include "twofluid/simple_solver.hpp"  // for SolveResult

namespace twofluid {

class TwoFluidSolver {
public:
    TwoFluidSolver(FVMesh& mesh);

    // Physical properties
    double rho_l = 998.2, rho_g = 1.225;
    double mu_l = 1.003e-3, mu_g = 1.789e-5;
    double cp_l = 4182.0, cp_g = 1006.0;
    double k_l = 0.6, k_g = 0.0257;
    double d_b = 0.005;
    double h_fg = 2.257e6;
    double T_sat = 373.15;
    double r_phase_change = 0.1;

    // Solver parameters
    double alpha_u = 0.5, alpha_p = 0.3;
    double alpha_alpha = 0.5, alpha_T = 0.7;
    double tol = 1e-4;
    int max_outer_iter = 200;
    bool solve_energy = false;
    bool solve_momentum = true;
    Eigen::VectorXd g;  // gravity vector (ndim)

    double dt = 0.001;

    // Initialize
    void initialize(double alpha_g_init = 0.05);

    // Boundary conditions
    void set_wall_bc(const std::string& patch, double q_wall = 0.0);
    void set_inlet_bc(const std::string& patch, double alpha_g,
                      const Eigen::VectorXd& U_l, const Eigen::VectorXd& U_g,
                      double T_l = 0.0, double T_g = 0.0);
    void set_outlet_bc(const std::string& patch, double p_val = 0.0);

    // Solve
    SolveResult solve_transient(double t_end, double dt_in, int report_interval = 100);

    // Access fields
    ScalarField& alpha_g_field() { return alpha_g_; }
    ScalarField& alpha_l_field() { return alpha_l_; }
    VectorField& U_l_field() { return U_l_; }
    VectorField& U_g_field() { return U_g_; }
    ScalarField& pressure() { return p_; }
    ScalarField& T_l_field() { return T_l_; }
    ScalarField& T_g_field() { return T_g_; }

private:
    FVMesh& mesh_;

    // Fields
    ScalarField alpha_g_, alpha_l_, p_, T_l_, T_g_;
    VectorField U_l_, U_g_;

    // BC type maps
    std::unordered_map<std::string, BoundaryCondition> bc_u_l_, bc_u_g_;
    std::unordered_map<std::string, BoundaryCondition> bc_p_, bc_alpha_;
    std::unordered_map<std::string, BoundaryCondition> bc_T_l_, bc_T_g_;

    // Wall heat flux storage
    std::unordered_map<std::string, double> wall_heat_flux_;

    // Face BC cache
    std::unordered_map<int, std::pair<std::string, int>> face_bc_cache_;
    void build_face_bc_cache();

    // Update zero-gradient boundary values from adjacent cells
    void update_zero_gradient_boundaries();

    // SIMPLE iteration (returns max residual)
    double simple_iteration();

    // Sub-solvers
    Eigen::VectorXd compute_phase_change_rate();
    double solve_phase_momentum(const std::string& phase, int comp,
                                const Eigen::VectorXd& mass_flux,
                                const Eigen::VectorXd& K_drag);
    double solve_pressure_correction(const Eigen::VectorXd& mf_l,
                                     const Eigen::VectorXd& mf_g);
    double solve_volume_fraction(const Eigen::VectorXd& mf_g,
                                 const Eigen::VectorXd& dot_m);
    double solve_coupled_energy(const Eigen::VectorXd& mf_l,
                                const Eigen::VectorXd& mf_g,
                                const Eigen::VectorXd& dot_m);
    double solve_phase_energy(const std::string& phase, double dt_local,
                              ScalarField& alpha, double rho, double cp,
                              double k_cond, VectorField& U,
                              const Eigen::VectorXd& mass_flux,
                              ScalarField& T, ScalarField& T_other,
                              const Eigen::VectorXd& h_i,
                              const Eigen::VectorXd& a_i,
                              const Eigen::VectorXd& dot_m,
                              const std::unordered_map<std::string, BoundaryCondition>& bc_T);

    // Per-component aP storage for pressure correction
    Eigen::VectorXd aP_l_, aP_g_storage_;
    bool has_aP_l_ = false, has_aP_g_ = false;
};

} // namespace twofluid
