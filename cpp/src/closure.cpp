#include "twofluid/closure.hpp"

#include <cmath>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace twofluid {

// ---------------------------------------------------------------------------
// Schiller-Naumann drag coefficient
// ---------------------------------------------------------------------------

Eigen::VectorXd schiller_naumann_drag(const Eigen::VectorXd& Re_p_in) {
    int n = static_cast<int>(Re_p_in.size());
    Eigen::VectorXd Re_p = Re_p_in.cwiseMax(1e-10);
    Eigen::VectorXd C_D(n);

#pragma omp parallel for schedule(static)
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
#pragma omp parallel for schedule(static)
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
#pragma omp parallel for schedule(static)
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
#pragma omp parallel for schedule(static)
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
#pragma omp parallel for schedule(static)
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

#pragma omp parallel for schedule(static)
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
#pragma omp parallel for schedule(static)
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


// ---------------------------------------------------------------------------
// H4: Grace drag correlation
// ---------------------------------------------------------------------------

Eigen::VectorXd grace_drag(const Eigen::VectorXd& alpha_g, double rho_l, double rho_g,
                            const Eigen::MatrixXd& U_g, const Eigen::MatrixXd& U_l,
                            double d_b, double mu_l, double sigma)
{
    constexpr double g = 9.81;
    constexpr double mu_ref = 0.0009;

    int n = static_cast<int>(alpha_g.size());
    int ndim = static_cast<int>(U_g.cols());

    double Eo = g * (rho_l - rho_g) * d_b * d_b / std::max(sigma, 1e-15);
    double Mo = g * std::pow(mu_l, 4.0) * (rho_l - rho_g)
                / std::max(rho_l * rho_l * std::pow(sigma, 3.0), 1e-15);

    double H = 4.0 / 3.0 * Eo * std::pow(std::max(Mo, 1e-15), -0.149)
               * std::pow(mu_l / mu_ref, -0.14);

    double J;
    if (H > 2.0 && H <= 59.3)
        J = 0.94 * std::pow(H, 0.757);
    else
        J = 3.42 * std::pow(std::max(H, 1e-15), 0.441);

    // Terminal velocity from J (Grace method)
    // U_T = mu_l / (rho_l * d_b) * M^(-0.149) * (J - 0.857)
    double M_pow = std::pow(std::max(Mo, 1e-300), -0.149);
    double U_T = mu_l / std::max(rho_l * d_b, 1e-30) * M_pow * std::max(J - 0.857, 1e-15);

    // C_D from terminal velocity balance: C_D = 4/3 * g * d_b * (rho_l - rho_g) / (rho_l * U_T^2)
    double C_D_grace = 4.0 / 3.0 * g * d_b * (rho_l - rho_g)
                       / std::max(rho_l * U_T * U_T, 1e-30);

    Eigen::VectorXd K_drag(n);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        double mag_sq = 0.0;
        for (int d = 0; d < ndim; ++d) {
            double diff = U_g(i, d) - U_l(i, d);
            mag_sq += diff * diff;
        }
        double u_rel = std::max(std::sqrt(mag_sq), 1e-15);

        K_drag[i] = 0.75 * C_D_grace * alpha_g[i] * rho_l * u_rel / std::max(d_b, 1e-15);
    }
    return K_drag;
}

// ---------------------------------------------------------------------------
// H4: Tomiyama drag correlation
// ---------------------------------------------------------------------------

Eigen::VectorXd tomiyama_drag(const Eigen::VectorXd& alpha_g, double rho_l, double rho_g,
                               const Eigen::MatrixXd& U_g, const Eigen::MatrixXd& U_l,
                               double d_b, double mu_l, double sigma)
{
    constexpr double g = 9.81;

    int n = static_cast<int>(alpha_g.size());
    int ndim = static_cast<int>(U_g.cols());

    double Eo = g * (rho_l - rho_g) * d_b * d_b / std::max(sigma, 1e-15);
    double C_D_eo_limit = 8.0 / 3.0 * Eo / std::max(Eo + 4.0, 1e-15);

    Eigen::VectorXd K_drag(n);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        double mag_sq = 0.0;
        for (int d = 0; d < ndim; ++d) {
            double diff = U_g(i, d) - U_l(i, d);
            mag_sq += diff * diff;
        }
        double u_rel = std::max(std::sqrt(mag_sq), 1e-15);
        double Re_p = std::max(rho_l * u_rel * d_b / std::max(mu_l, 1e-15), 1e-15);

        double C_D_stokes  = 24.0 * (1.0 + 0.15 * std::pow(Re_p, 0.687)) / Re_p;
        double C_D = std::max(C_D_stokes, C_D_eo_limit);

        K_drag[i] = 0.75 * C_D * alpha_g[i] * rho_l * u_rel / std::max(d_b, 1e-15);
    }
    return K_drag;
}

// ---------------------------------------------------------------------------
// H4: Ishii-Zuber drag correlation
// ---------------------------------------------------------------------------

Eigen::VectorXd ishii_zuber_drag(const Eigen::VectorXd& alpha_g, double rho_l,
                                  const Eigen::MatrixXd& U_g, const Eigen::MatrixXd& U_l,
                                  double d_b, double mu_l)
{
    int n = static_cast<int>(alpha_g.size());
    int ndim = static_cast<int>(U_g.cols());

    Eigen::VectorXd K_drag(n);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        double mag_sq = 0.0;
        for (int d = 0; d < ndim; ++d) {
            double diff = U_g(i, d) - U_l(i, d);
            mag_sq += diff * diff;
        }
        double u_rel = std::max(std::sqrt(mag_sq), 1e-15);

        // Mixture viscosity: mu_m = mu_l * (1 - alpha_g)^(-2.5)
        double one_minus_alpha = std::max(1.0 - alpha_g[i], 1e-6);
        double mu_m = mu_l * std::pow(one_minus_alpha, -2.5);

        double Re_m = std::max(rho_l * u_rel * d_b / std::max(mu_m, 1e-15), 1e-15);
        double C_D = 24.0 / Re_m * (1.0 + 0.1 * std::pow(Re_m, 0.75));

        K_drag[i] = 0.75 * C_D * alpha_g[i] * rho_l * u_rel / std::max(d_b, 1e-15);
    }
    return K_drag;
}

// ---------------------------------------------------------------------------
// H5: Tomiyama lift force
// ---------------------------------------------------------------------------

Eigen::VectorXd lift_force_tomiyama(
    const Eigen::VectorXd& alpha_g, double rho_l, double rho_g,
    const Eigen::MatrixXd& U_g, const Eigen::MatrixXd& U_l,
    double d_b, double mu_l, double sigma,
    const Eigen::MatrixXd& curl_U_l)
{
    constexpr double g = 9.81;

    int n = static_cast<int>(alpha_g.size());
    int ndim = static_cast<int>(U_g.cols());

    double Eo = g * (rho_l - rho_g) * d_b * d_b / std::max(sigma, 1e-15);

    // f(Eo) polynomial for shape correction
    auto f_Eo = [](double eo) {
        return 0.00105 * eo * eo * eo - 0.0159 * eo * eo - 0.0204 * eo + 0.474;
    };

    // Determine C_L from Eo regime
    double C_L;
    if (Eo >= 10.0) {
        C_L = -0.29;
    } else if (Eo >= 4.0) {
        C_L = f_Eo(Eo);
    } else {
        // C_L = min(0.288*tanh(0.121*Re), f(Eo)) — Re here is a representative
        // scalar; per Tomiyama the Re limit is applied cell-wise below
        C_L = f_Eo(Eo); // placeholder; overridden per-cell when Eo < 4
    }

    // Output: scalar per cell (magnitude of lift body force component aligned with curl)
    // F_lift = C_L * alpha_g * rho_l * |u_rel x curl_U_l|  (signed via C_L)
    Eigen::VectorXd F_lift(n);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        double mag_sq = 0.0;
        for (int d = 0; d < ndim; ++d) {
            double diff = U_g(i, d) - U_l(i, d);
            mag_sq += diff * diff;
        }
        double u_rel = std::max(std::sqrt(mag_sq), 1e-15);
        double Re_p = rho_l * u_rel * d_b / std::max(mu_l, 1e-15);

        double C_L_cell = C_L;
        if (Eo < 4.0) {
            double cl_re = 0.288 * std::tanh(0.121 * Re_p);
            C_L_cell = std::min(cl_re, f_Eo(Eo));
        }

        // curl magnitude per cell
        double curl_mag = 0.0;
        for (int d = 0; d < curl_U_l.cols(); ++d) {
            curl_mag += curl_U_l(i, d) * curl_U_l(i, d);
        }
        curl_mag = std::sqrt(curl_mag);

        F_lift[i] = C_L_cell * alpha_g[i] * rho_l * u_rel * curl_mag;
    }
    return F_lift;
}

// ---------------------------------------------------------------------------
// H5: Antal wall lubrication force
// ---------------------------------------------------------------------------

Eigen::VectorXd wall_lubrication_antal(
    const Eigen::VectorXd& alpha_g, double rho_l,
    const Eigen::MatrixXd& U_g, const Eigen::MatrixXd& U_l,
    double d_b, const Eigen::VectorXd& y_wall,
    const Eigen::MatrixXd& n_wall)
{
    constexpr double C_w1 = -0.01;
    constexpr double C_w2 =  0.05;

    int n = static_cast<int>(alpha_g.size());
    int ndim = static_cast<int>(U_g.cols());

    // Output: scalar magnitude of wall lubrication force per cell
    Eigen::VectorXd F_wl(n);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        double mag_sq = 0.0;
        for (int d = 0; d < ndim; ++d) {
            double diff = U_g(i, d) - U_l(i, d);
            mag_sq += diff * diff;
        }
        double u_rel_sq = mag_sq; // |u_rel|^2

        double yw = std::max(y_wall[i], 1e-15);
        double C_wl = std::max(0.0, C_w1 / std::max(d_b, 1e-15) + C_w2 / yw);

        // n_wall magnitude for normalisation (should be unit, but guard anyway)
        double n_mag = 0.0;
        for (int d = 0; d < n_wall.cols(); ++d)
            n_mag += n_wall(i, d) * n_wall(i, d);
        n_mag = std::max(std::sqrt(n_mag), 1e-15);

        F_wl[i] = -C_wl * alpha_g[i] * rho_l * u_rel_sq / std::max(d_b, 1e-15) * n_mag;
    }
    return F_wl;
}

// ---------------------------------------------------------------------------
// H5: Burns turbulent dispersion force
// ---------------------------------------------------------------------------

Eigen::VectorXd turbulent_dispersion_burns(
    const Eigen::VectorXd& alpha_g,
    const Eigen::VectorXd& mu_t,
    const Eigen::MatrixXd& grad_alpha_g,
    double C_td)
{
    int n = static_cast<int>(alpha_g.size());
    int ndim = static_cast<int>(grad_alpha_g.cols());

    Eigen::VectorXd F_td(n);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        double denom = std::max(alpha_g[i] * (1.0 - alpha_g[i]), 1e-15);

        // |grad_alpha_g| magnitude
        double grad_mag = 0.0;
        for (int d = 0; d < ndim; ++d)
            grad_mag += grad_alpha_g(i, d) * grad_alpha_g(i, d);
        grad_mag = std::sqrt(grad_mag);

        F_td[i] = -C_td * mu_t[i] * grad_mag / denom;
    }
    return F_td;
}

// ---------------------------------------------------------------------------
// Virtual mass (added mass) force coefficient
// ---------------------------------------------------------------------------

Eigen::VectorXd virtual_mass_coefficient(
    const Eigen::VectorXd& alpha_g, double rho_l, double C_vm)
{
    int n = static_cast<int>(alpha_g.size());
    Eigen::VectorXd K_vm(n);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        K_vm[i] = C_vm * rho_l * alpha_g[i];
    }
    return K_vm;
}

} // namespace twofluid
