#pragma once
#include <cmath>

namespace twofluid {

/// IAPWS-IF97 steam table properties
struct SteamProperties {
    double T;       // temperature [K]
    double p;       // pressure [Pa]
    double rho;     // density [kg/m3]
    double h;       // specific enthalpy [J/kg]
    double cp;      // specific heat [J/(kg.K)]
    double mu;      // dynamic viscosity [Pa.s]
    double k;       // thermal conductivity [W/(m.K)]
    double sigma;   // surface tension [N/m] (liquid-vapor interface)
};

class IAPWS_IF97 {
public:
    /// Saturation temperature from pressure [K]
    static double T_sat(double p_Pa);

    /// Saturation pressure from temperature [Pa]
    static double p_sat(double T_K);

    /// Liquid properties at given T [K] and p [Pa]
    static SteamProperties liquid(double T_K, double p_Pa);

    /// Vapor properties at given T [K] and p [Pa]
    static SteamProperties vapor(double T_K, double p_Pa);

    /// Latent heat of vaporization at given pressure [J/kg]
    static double h_fg(double p_Pa);

    /// Surface tension at given temperature [N/m]
    /// IAPWS 1994 correlation (valid 273.15-647.096 K)
    static double surface_tension(double T_K);

    /// Liquid viscosity [Pa.s] — Vogel-Fulcher-Tammann form
    static double mu_liquid(double T_K, double rho);

    /// Vapor viscosity [Pa.s]
    static double mu_vapor(double T_K, double rho);

    /// Liquid thermal conductivity [W/(m.K)]
    static double k_liquid(double T_K, double rho);

    /// Vapor thermal conductivity [W/(m.K)]
    static double k_vapor(double T_K, double rho);

private:
    // Region 1 (subcooled liquid): specific volume, enthalpy, cp
    static double v1(double T, double p);   // [m3/kg]
    static double h1(double T, double p);   // [J/kg]
    static double cp1(double T, double p);  // [J/(kg.K)]

    // Region 2 (superheated steam): specific volume, enthalpy, cp
    static double v2(double T, double p);   // [m3/kg]
    static double h2(double T, double p);   // [J/kg]
    static double cp2(double T, double p);  // [J/(kg.K)]
};

} // namespace twofluid
