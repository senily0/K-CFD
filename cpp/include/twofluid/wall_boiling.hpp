#pragma once
#include <Eigen/Dense>
#include "twofluid/mesh.hpp"
#include "twofluid/fields.hpp"

namespace twofluid {

/// Wall boiling model parameters
struct WallBoilingParams {
    double T_sat = 373.15;
    double h_fg = 2.257e6;
    double rho_l = 998.2, rho_g = 1.225;
    double cp_l = 4182.0, k_l = 0.6;
    double mu_l = 1e-3;
    double sigma = 0.059;   // surface tension
    double contact_angle = 80.0;  // degrees
    double g = 9.81;
    // Lemmert-Chawla constants
    double Na_m = 185.0;    // nucleation density coefficient
    double Na_p = 1.805;    // nucleation density exponent
};

/// RPI wall boiling model (Kurul-Podowski 1991)
class RPIWallBoiling {
public:
    RPIWallBoiling(const WallBoilingParams& params);

    /// Compute bubble departure diameter [m] (Fritz 1935)
    double departure_diameter() const;

    /// Compute departure frequency [1/s] (Cole 1960)
    double departure_frequency() const;

    /// Compute nucleation site density [1/m^2] (Lemmert-Chawla 1977)
    double nucleation_density(double T_wall) const;

    /// Compute wall heat flux partition [W/m^2]
    struct HeatFluxPartition {
        double q_conv;    // convective
        double q_quench;  // quenching
        double q_evap;    // evaporative
        double q_total;   // sum
        double m_dot;     // evaporation mass flux [kg/(m^2·s)]
    };
    HeatFluxPartition compute(double T_wall, double T_liquid,
                              double h_conv) const;

    WallBoilingParams params;

private:
    double d_w_;  // cached departure diameter
    double f_;    // cached departure frequency
};

} // namespace twofluid
