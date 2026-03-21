#include "twofluid/wall_boiling.hpp"
#include <cmath>
#include <algorithm>

namespace twofluid {

RPIWallBoiling::RPIWallBoiling(const WallBoilingParams& params)
    : params(params)
{
    d_w_ = departure_diameter();
    f_   = departure_frequency();
}

// ---------------------------------------------------------------------------
// Fritz (1935): bubble departure diameter [m]
//   d_w = 0.0208 * theta * sqrt(sigma / (g * (rho_l - rho_g)))
// theta is contact angle in degrees
// ---------------------------------------------------------------------------
double RPIWallBoiling::departure_diameter() const {
    double theta = params.contact_angle;  // degrees
    double arg   = params.sigma / (params.g * (params.rho_l - params.rho_g));
    return 0.0208 * theta * std::sqrt(arg);
}

// ---------------------------------------------------------------------------
// Cole (1960): departure frequency [1/s]
//   f = sqrt(4 * g * (rho_l - rho_g) / (3 * d_w * rho_l))
// ---------------------------------------------------------------------------
double RPIWallBoiling::departure_frequency() const {
    double d_w = d_w_;
    if (d_w < 1e-10) d_w = 1e-10;
    return std::sqrt(4.0 * params.g * (params.rho_l - params.rho_g)
                     / (3.0 * d_w * params.rho_l));
}

// ---------------------------------------------------------------------------
// Lemmert-Chawla (1977): nucleation site density [1/m^2]
//   N_a = (m * dT_sup)^p   where dT_sup = T_wall - T_sat
// ---------------------------------------------------------------------------
double RPIWallBoiling::nucleation_density(double T_wall) const {
    double dT_sup = T_wall - params.T_sat;
    if (dT_sup <= 0.0) return 0.0;
    return std::pow(params.Na_m * dT_sup, params.Na_p);
}

// ---------------------------------------------------------------------------
// RPI heat flux partition (Kurul-Podowski 1991)
// ---------------------------------------------------------------------------
RPIWallBoiling::HeatFluxPartition
RPIWallBoiling::compute(double T_wall, double T_liquid, double h_conv) const
{
    HeatFluxPartition part{};

    double dT_wall = T_wall - T_liquid;
    if (dT_wall <= 0.0) {
        // No wall superheat: all convective, no boiling
        part.q_conv   = h_conv * dT_wall;
        part.q_quench = 0.0;
        part.q_evap   = 0.0;
        part.q_total  = part.q_conv;
        part.m_dot    = 0.0;
        return part;
    }

    double d_w = d_w_;
    double f   = f_;
    double N_a = nucleation_density(T_wall);

    // Fractional area covered by bubbles (Kurul-Podowski, K=4)
    constexpr double K = 4.0;
    double A_b = std::min(1.0, K * N_a * M_PI * d_w * d_w / 4.0);

    // Convective component
    part.q_conv = h_conv * (1.0 - A_b) * dT_wall;

    // Quenching component
    // alpha_l_th = k_l / (rho_l * cp_l)  (thermal diffusivity)
    double alpha_l_th = params.k_l / (params.rho_l * params.cp_l);
    // t_w = 1/f (waiting time)
    double t_w = (f > 1e-15) ? 1.0 / f : 1e15;
    part.q_quench = 2.0 * params.k_l / std::sqrt(M_PI * alpha_l_th * t_w)
                    * dT_wall * A_b;

    // Evaporative component
    part.q_evap = N_a * f * (M_PI / 6.0) * d_w * d_w * d_w
                  * params.rho_g * params.h_fg;

    part.q_total = part.q_conv + part.q_quench + part.q_evap;

    // Evaporation mass flux [kg/(m^2·s)]
    part.m_dot = (params.h_fg > 0.0) ? part.q_evap / params.h_fg : 0.0;

    return part;
}

} // namespace twofluid
