#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <Eigen/Dense>
#include "twofluid/mesh.hpp"
#include "twofluid/fields.hpp"

namespace twofluid {

class KEpsilonModel {
public:
    // Standard k-epsilon constants
    static constexpr double C_mu = 0.09;
    static constexpr double C_1 = 1.44;
    static constexpr double C_2 = 1.92;
    static constexpr double sigma_k = 1.0;
    static constexpr double sigma_eps = 1.3;
    static constexpr double kappa = 0.41;    // von Karman constant
    static constexpr double E_wall = 9.793;  // wall function constant

    KEpsilonModel(FVMesh& mesh, double rho, double mu);

    ScalarField& k() { return k_; }
    ScalarField& epsilon() { return epsilon_; }
    const ScalarField& k() const { return k_; }
    const ScalarField& epsilon() const { return epsilon_; }

    /// Turbulent viscosity: mu_t = rho * C_mu * k^2 / epsilon
    Eigen::VectorXd get_mu_t() const;

    /// Solve k and epsilon transport equations.
    void solve(const VectorField& U, const Eigen::VectorXd& mass_flux,
               const std::unordered_map<std::string, std::string>& bc_types,
               double alpha_k = 0.7, double alpha_eps = 0.7);

    /// Apply wall functions (modifies k, epsilon near walls).
    void apply_wall_functions(const VectorField& U,
                              const std::vector<std::string>& wall_patches);

    /// Initialize k and epsilon uniformly.
    void initialize(double k_init, double eps_init);

private:
    FVMesh& mesh_;
    double rho_, mu_;
    ScalarField k_, epsilon_;

    /// Production term: P_k = mu_t * 2 * S_ij * S_ij
    Eigen::VectorXd compute_production(const VectorField& U,
                                       const Eigen::VectorXd& mu_t) const;
};

} // namespace twofluid
