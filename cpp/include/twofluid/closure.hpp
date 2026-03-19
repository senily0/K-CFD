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

} // namespace twofluid
