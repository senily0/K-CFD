#include "twofluid/closure.hpp"

#include <cmath>
#include <algorithm>

namespace twofluid {

// ---------------------------------------------------------------------------
// Schiller-Naumann drag coefficient
// ---------------------------------------------------------------------------

Eigen::VectorXd schiller_naumann_drag(const Eigen::VectorXd& Re_p_in) {
    int n = static_cast<int>(Re_p_in.size());
    Eigen::VectorXd Re_p = Re_p_in.cwiseMax(1e-10);
    Eigen::VectorXd C_D(n);

    for (int i = 0; i < n; ++i) {
        if (Re_p[i] < 1000.0) {
            C_D[i] = 24.0 / Re_p[i] * (1.0 + 0.15 * std::pow(Re_p[i], 0.687));
        } else {
            C_D[i] = 0.44;
        }
    }
    return C_D;
}

// ---------------------------------------------------------------------------
// Implicit drag coefficient
// ---------------------------------------------------------------------------

Eigen::VectorXd drag_coefficient_implicit(
    const Eigen::VectorXd& alpha_g,
    double rho_l,
    const Eigen::MatrixXd& U_g,
    const Eigen::MatrixXd& U_l,
    double d_b,
    double mu_l)
{
    int n = static_cast<int>(alpha_g.size());
    int ndim = static_cast<int>(U_g.cols());

    // Relative velocity magnitude
    Eigen::VectorXd u_rel_mag(n);
    for (int i = 0; i < n; ++i) {
        double mag_sq = 0.0;
        for (int d = 0; d < ndim; ++d) {
            double diff = U_g(i, d) - U_l(i, d);
            mag_sq += diff * diff;
        }
        u_rel_mag[i] = std::max(std::sqrt(mag_sq), 1e-15);
    }

    Eigen::VectorXd Re_p = (rho_l * u_rel_mag * d_b) / mu_l;
    Eigen::VectorXd C_D = schiller_naumann_drag(Re_p);

    // K_drag = 0.75 * C_D * alpha_g * rho_l * |u_rel| / d_b
    Eigen::VectorXd K_drag(n);
    for (int i = 0; i < n; ++i) {
        K_drag[i] = 0.75 * C_D[i] * alpha_g[i] * rho_l * u_rel_mag[i] / d_b;
    }
    return K_drag;
}

// ---------------------------------------------------------------------------
// Ranz-Marshall Nusselt number
// ---------------------------------------------------------------------------

Eigen::VectorXd ranz_marshall_nusselt(
    double rho_l, double mu_l, double cp_l, double k_l,
    const Eigen::MatrixXd& U_g, const Eigen::MatrixXd& U_l,
    double d_b)
{
    int n = static_cast<int>(U_g.rows());
    int ndim = static_cast<int>(U_g.cols());

    Eigen::VectorXd u_rel_mag(n);
    for (int i = 0; i < n; ++i) {
        double mag_sq = 0.0;
        for (int d = 0; d < ndim; ++d) {
            double diff = U_g(i, d) - U_l(i, d);
            mag_sq += diff * diff;
        }
        u_rel_mag[i] = std::max(std::sqrt(mag_sq), 1e-15);
    }

    Eigen::VectorXd Re_p = (rho_l * u_rel_mag * d_b) / mu_l;
    Re_p = Re_p.cwiseMax(1e-10);
    double Pr = mu_l * cp_l / k_l;

    Eigen::VectorXd Nu(n);
    for (int i = 0; i < n; ++i) {
        Nu[i] = 2.0 + 0.6 * std::pow(Re_p[i], 0.5) * std::pow(Pr, 0.333);
    }
    return Nu;
}

// ---------------------------------------------------------------------------
// Interfacial heat transfer coefficients
// ---------------------------------------------------------------------------

std::pair<Eigen::VectorXd, Eigen::VectorXd> interfacial_heat_transfer(
    const Eigen::VectorXd& alpha_g, double rho_l, double mu_l,
    double cp_l, double k_l,
    const Eigen::MatrixXd& U_g, const Eigen::MatrixXd& U_l,
    double d_b)
{
    Eigen::VectorXd Nu = ranz_marshall_nusselt(rho_l, mu_l, cp_l, k_l,
                                                U_g, U_l, d_b);

    int n = static_cast<int>(alpha_g.size());
    Eigen::VectorXd h_i(n);
    Eigen::VectorXd a_i(n);

    for (int i = 0; i < n; ++i) {
        h_i[i] = Nu[i] * k_l / d_b;
        a_i[i] = 6.0 * alpha_g[i] / std::max(d_b, 1e-15);
    }

    return {h_i, a_i};
}

// ---------------------------------------------------------------------------
// Sato bubble-induced turbulence
// ---------------------------------------------------------------------------

Eigen::VectorXd sato_bubble_induced_turbulence(
    const Eigen::VectorXd& alpha_g,
    double rho_l,
    const Eigen::MatrixXd& U_g,
    const Eigen::MatrixXd& U_l,
    double d_b)
{
    constexpr double C_MU_B = 0.6;
    int n = static_cast<int>(alpha_g.size());
    int ndim = static_cast<int>(U_g.cols());

    Eigen::VectorXd mu_t_BIT(n);
    for (int i = 0; i < n; ++i) {
        double mag_sq = 0.0;
        for (int d = 0; d < ndim; ++d) {
            double diff = U_g(i, d) - U_l(i, d);
            mag_sq += diff * diff;
        }
        double u_rel_mag = std::sqrt(mag_sq);
        mu_t_BIT[i] = C_MU_B * rho_l * alpha_g[i] * d_b * u_rel_mag;
    }
    return mu_t_BIT;
}

} // namespace twofluid
