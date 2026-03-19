#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <Eigen/Dense>
#include "twofluid/mesh.hpp"
#include "twofluid/fields.hpp"
#include "twofluid/fvm_operators.hpp"

namespace twofluid {

/// Menter k-omega SST turbulence model (1994)
/// Blends k-omega near walls with k-epsilon in freestream
class KOmegaSSTModel {
public:
    // SST constants
    // Set 1 (k-omega): sigma_k1=0.85, sigma_w1=0.5, beta1=0.075, beta_star=0.09, kappa=0.41, a1=0.31
    // Set 2 (k-epsilon): sigma_k2=1.0, sigma_w2=0.856, beta2=0.0828
    // Blending via F1, F2 functions

    static constexpr double beta_star = 0.09;
    static constexpr double kappa_vk = 0.41;
    static constexpr double a1 = 0.31;

    // Set 1 (inner/k-omega)
    static constexpr double sigma_k1 = 0.85;
    static constexpr double sigma_w1 = 0.5;
    static constexpr double beta_1 = 0.075;
    static constexpr double gamma_1 = 5.0/9.0;  // beta_1/beta_star - sigma_w1*kappa^2/sqrt(beta_star)

    // Set 2 (outer/k-epsilon)
    static constexpr double sigma_k2 = 1.0;
    static constexpr double sigma_w2 = 0.856;
    static constexpr double beta_2 = 0.0828;
    static constexpr double gamma_2 = 0.44;

    KOmegaSSTModel(FVMesh& mesh, double rho, double mu);

    ScalarField& k() { return k_; }
    ScalarField& omega() { return omega_; }
    const ScalarField& k() const { return k_; }
    const ScalarField& omega() const { return omega_; }

    Eigen::VectorXd get_mu_t() const;

    void solve(const VectorField& U, const Eigen::VectorXd& mass_flux,
               const std::unordered_map<std::string, std::string>& bc_types,
               double alpha_k = 0.7, double alpha_w = 0.7);

    void apply_wall_functions(const VectorField& U,
                              const std::vector<std::string>& wall_patches);

    void initialize(double k_init, double omega_init);

    /// Compute wall distance for each cell (needed for SST blending)
    void compute_wall_distance(const std::vector<std::string>& wall_patches);

    /// Get y+ values for monitoring
    Eigen::VectorXd get_y_plus(const VectorField& U,
                                const std::vector<std::string>& wall_patches) const;

private:
    FVMesh& mesh_;
    double rho_, mu_;
    ScalarField k_, omega_;
    Eigen::VectorXd wall_dist_;  // wall distance per cell
    bool wall_dist_computed_ = false;

    // Blending functions
    Eigen::VectorXd compute_F1() const;
    Eigen::VectorXd compute_F2() const;

    // Production limiter: min(P_k, 10*beta_star*rho*k*omega)
    Eigen::VectorXd compute_production(const VectorField& U,
                                        const Eigen::VectorXd& mu_t) const;

    // Cross-diffusion term for omega equation
    Eigen::VectorXd compute_CDkw() const;
};

} // namespace twofluid
