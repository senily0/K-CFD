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
#include "twofluid/chemistry.hpp"
#include "twofluid/wall_boiling.hpp"

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

    // Adaptive time stepping
    bool adaptive_dt = false;
    double cfl_target = 0.5;
    double dt_min = 1e-8;
    double dt_max = 1.0;

    // Physical limits (user-configurable, no hardcoded clipping)
    double U_max = 1e6;       // velocity clamp [m/s] — set per problem
    double T_min = 1.0;       // temperature floor [K]
    double T_max = 1e5;       // temperature ceiling [K]
    double alpha_max = 1.0;   // max gas volume fraction (1.0 = no restriction)

    // Divergence detection
    int divergence_count = 0;
    int max_divergence = 5;   // stop after this many divergent steps

    // Convection scheme: "upwind" or "muscl" (default: muscl for 2nd order)
    std::string convection_scheme = "muscl";
    std::string muscl_limiter = "van_leer";

    // Drag model selection: "schiller_naumann", "grace", "tomiyama", "ishii_zuber"
    std::string drag_model = "schiller_naumann";
    double sigma_surface = 0.0;  // surface tension coefficient [N/m] (0 = disabled)

    // Interfacial forces (enable flags)
    bool enable_lift_force = false;
    bool enable_wall_lubrication = false;
    bool enable_turbulent_dispersion = false;
    double C_td = 1.0;  // turbulent dispersion coefficient
    bool enable_virtual_mass = false;
    double C_vm = 0.5;  // virtual mass coefficient (0.5 for spheres, Lamb 1932)

    // Property model: "constant" or "iapws97"
    std::string property_model = "constant";
    double system_pressure = 101325.0;  // [Pa] for IAPWS property evaluation

    // Non-orthogonal correction
    int n_nonorth_correctors = 0;  // 0 = disabled

    // Time scheme: "euler" or "bdf2"
    std::string time_scheme = "euler";

    // RPI wall boiling model (Kurul-Podowski 1991)
    bool enable_wall_boiling = false;
    double wall_boiling_contact_angle = 80.0;

    // Species transport (optional)
    bool solve_species = false;
    double species_D = 1e-5;          // diffusion coefficient
    double species_k_r = 0.0;         // reaction rate (0 = no reaction)
    std::string species_bc_inlet_patch;
    double species_C_inlet = 1.0;

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

    // RPI wall boiling: per-cell evaporation mass source [kg/(m^3·s)]
    // Filled during liquid energy solve; consumed by solve_volume_fraction.
    Eigen::VectorXd rpi_wall_dot_m_;

    // Cell-local property fields (used when property_model == "iapws97")
    Eigen::VectorXd rho_l_field, rho_g_field;
    Eigen::VectorXd mu_l_field, mu_g_field;
    Eigen::VectorXd cp_l_field, cp_g_field;
    Eigen::VectorXd k_l_field, k_g_field;

    // Species transport
    std::unique_ptr<SpeciesTransportSolver> species_solver_;
    std::unique_ptr<FirstOrderReaction> reaction_;
};

} // namespace twofluid
