#pragma once

#include <utility>
#include <Eigen/Dense>

namespace twofluid {

/// Schiller-Naumann drag coefficient.
/// C_D = 24/Re_p * (1 + 0.15 * Re_p^0.687)  for Re_p < 1000
/// C_D = 0.44                                  for Re_p >= 1000
Eigen::VectorXd schiller_naumann_drag(const Eigen::VectorXd& Re_p);

/// Implicit drag coefficient K_drag for momentum coupling.
/// K_drag = 0.75 * C_D * alpha_g * rho_l * |u_rel| / d_b
Eigen::VectorXd drag_coefficient_implicit(
    const Eigen::VectorXd& alpha_g,
    double rho_l,
    const Eigen::MatrixXd& U_g,
    const Eigen::MatrixXd& U_l,
    double d_b,
    double mu_l
);

/// Ranz-Marshall Nusselt number: Nu = 2 + 0.6 * Re_p^0.5 * Pr^0.333
Eigen::VectorXd ranz_marshall_nusselt(
    double rho_l, double mu_l, double cp_l, double k_l,
    const Eigen::MatrixXd& U_g, const Eigen::MatrixXd& U_l,
    double d_b
);

/// Interfacial heat transfer coefficients (h_i, a_i).
/// h_i = Nu * k_l / d_b
/// a_i = 6 * alpha_g / d_b
std::pair<Eigen::VectorXd, Eigen::VectorXd> interfacial_heat_transfer(
    const Eigen::VectorXd& alpha_g, double rho_l, double mu_l,
    double cp_l, double k_l,
    const Eigen::MatrixXd& U_g, const Eigen::MatrixXd& U_l,
    double d_b
);

/// Sato bubble-induced turbulence viscosity.
/// mu_t_BIT = C_mu_b * rho_l * alpha_g * d_b * |u_g - u_l|
/// C_mu_b = 0.6
Eigen::VectorXd sato_bubble_induced_turbulence(
    const Eigen::VectorXd& alpha_g,
    double rho_l,
    const Eigen::MatrixXd& U_g,
    const Eigen::MatrixXd& U_l,
    double d_b
);

// ---------------------------------------------------------------------------
// H4: Additional drag correlations
// ---------------------------------------------------------------------------

/// Grace drag correlation for deformable bubbles in clean systems.
/// Uses Morton number and Eotvos number to determine bubble shape regime.
Eigen::VectorXd grace_drag(const Eigen::VectorXd& alpha_g, double rho_l, double rho_g,
                            const Eigen::MatrixXd& U_g, const Eigen::MatrixXd& U_l,
                            double d_b, double mu_l, double sigma);

/// Tomiyama drag for contaminated bubbly flow.
Eigen::VectorXd tomiyama_drag(const Eigen::VectorXd& alpha_g, double rho_l, double rho_g,
                               const Eigen::MatrixXd& U_g, const Eigen::MatrixXd& U_l,
                               double d_b, double mu_l, double sigma);

/// Ishii-Zuber drag for dense particle/droplet systems.
Eigen::VectorXd ishii_zuber_drag(const Eigen::VectorXd& alpha_g, double rho_l,
                                  const Eigen::MatrixXd& U_g, const Eigen::MatrixXd& U_l,
                                  double d_b, double mu_l);

// ---------------------------------------------------------------------------
// H5: Interfacial forces
// ---------------------------------------------------------------------------

/// Tomiyama lift force coefficient.
/// C_L depends on Eotvos number:
///   Eo < 4: C_L = min(0.288*tanh(0.121*Re), f(Eo))
///   4 <= Eo < 10: f(Eo) = 0.00105*Eo^3 - 0.0159*Eo^2 - 0.0204*Eo + 0.474
///   Eo >= 10: C_L = -0.29
/// F_lift = C_L * alpha_g * rho_l * (u_g - u_l) x curl(u_l)
Eigen::VectorXd lift_force_tomiyama(
    const Eigen::VectorXd& alpha_g, double rho_l, double rho_g,
    const Eigen::MatrixXd& U_g, const Eigen::MatrixXd& U_l,
    double d_b, double mu_l, double sigma,
    const Eigen::MatrixXd& curl_U_l);

/// Antal wall lubrication force.
/// F_wl = -C_wl * alpha_g * rho_l * |u_rel|^2 / d_b * n_wall
/// C_wl = max(0, C_w1/d_b + C_w2/y_wall), C_w1=-0.01, C_w2=0.05
Eigen::VectorXd wall_lubrication_antal(
    const Eigen::VectorXd& alpha_g, double rho_l,
    const Eigen::MatrixXd& U_g, const Eigen::MatrixXd& U_l,
    double d_b, const Eigen::VectorXd& y_wall,
    const Eigen::MatrixXd& n_wall);

/// Burns turbulent dispersion force.
/// F_td = -C_td * mu_t * grad(alpha_g) / (alpha_g * (1-alpha_g) + eps)
Eigen::VectorXd turbulent_dispersion_burns(
    const Eigen::VectorXd& alpha_g,
    const Eigen::VectorXd& mu_t,
    const Eigen::MatrixXd& grad_alpha_g,
    double C_td = 1.0);

} // namespace twofluid
