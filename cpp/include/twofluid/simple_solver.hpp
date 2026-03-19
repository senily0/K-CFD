#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <Eigen/Dense>
#include "twofluid/mesh.hpp"
#include "twofluid/fields.hpp"
#include "twofluid/turbulence.hpp"

namespace twofluid {

struct SolveResult {
    bool converged;
    int iterations;
    std::vector<double> residuals;
    double wall_time;
};

class SIMPLESolver {
public:
    SIMPLESolver(FVMesh& mesh, double rho, double mu);

    // Parameters
    double alpha_u = 0.7;   // velocity under-relaxation
    double alpha_p = 0.3;   // pressure under-relaxation
    int max_iter = 500;
    double tol = 1e-5;

    // Boundary conditions
    void set_inlet(const std::string& patch, const Eigen::MatrixXd& U_vals);
    void set_wall(const std::string& patch);
    void set_outlet(const std::string& patch, double p_val = 0.0);

    // Turbulence (optional)
    void enable_turbulence(double k_init = 0.001, double eps_init = 0.01);
    KEpsilonModel* turbulence_model() { return turb_.get(); }

    // Solve
    SolveResult solve_steady();

    // Access fields
    VectorField& velocity() { return U_; }
    ScalarField& pressure() { return p_; }

private:
    FVMesh& mesh_;
    double rho_, mu_;
    VectorField U_;
    ScalarField p_;

    // Turbulence model (optional)
    std::unique_ptr<KEpsilonModel> turb_;
    std::vector<std::string> wall_patches_;

    // Per-component aP coefficients for pressure correction
    std::unordered_map<int, Eigen::VectorXd> aP_;

    // BC storage
    struct BCInfo { std::string type; };
    std::unordered_map<std::string, BCInfo> bc_u_, bc_p_;
    std::unordered_map<int, std::pair<std::string, int>> face_bc_cache_;

    void build_face_bc_cache();
    Eigen::VectorXd compute_face_mass_flux();
    double solve_momentum(int comp, const Eigen::VectorXd& mass_flux);
    double solve_pressure_correction(const Eigen::VectorXd& mass_flux);

    /// Geometric interpolation weight (owner-side) for internal face.
    double gc(int fid) const;
};

} // namespace twofluid
