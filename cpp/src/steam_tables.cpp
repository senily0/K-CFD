/// IAPWS-IF97 Steam Tables implementation
/// Reference: IAPWS-IF97 (2012), Wagner & Kruse
/// Implemented: Region 1 (subcooled liquid), Region 2 (superheated steam),
///              Saturation line (Eq. 30 forward / Eq. 31 backward)
#include "twofluid/steam_tables.hpp"
#include <algorithm>
#include <cmath>

namespace twofluid {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static constexpr double R_water = 461.526;   // specific gas constant [J/(kg.K)]
static constexpr double Tc      = 647.096;   // critical temperature [K]
static constexpr double pc      = 22.064e6;  // critical pressure [Pa]
static constexpr double rho_c   = 317.763;   // critical density [kg/m3]

// ---------------------------------------------------------------------------
// Region 1 — dimensionless Gibbs free energy and derivatives
// Reference points: p_star = 16.53 MPa,  T_star = 1386 K
// gamma = sum_i n_i * (7.1 - pi)^I_i * (tau - 1.222)^J_i
// ---------------------------------------------------------------------------
static constexpr double p_star1 = 16.53e6;
static constexpr double T_star1 = 1386.0;

// IAPWS-IF97 Table 2 (34 terms)
static const int r1_I[34] = {
    0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2,
    3, 3, 3,
    4, 4, 4,
    5,
    8, 8,
    21, 23, 29, 30, 31, 32
};
static const int r1_J[34] = {
    -2, -1,  0,  1,  2,  3,  4,  5,
    -9, -7, -1,  0,  1,  3,
    -3,  0,  1,  3, 17,
    -4,  0,  6,
    -5, -2, 10,
    -8,
    -11, -6,
    -29, -31, -38, -39, -40, -41
};
static const double r1_n[34] = {
     0.14632971213167e+00, -0.84548187169114e+00, -0.37563603672040e+01,
     0.33855169168385e+01, -0.95791963387872e+00,  0.15772038513228e+00,
    -0.16616417199501e-01,  0.81214629983568e-03,  0.28319080123804e-03,
    -0.60706301565874e-03, -0.18990068218419e-01, -0.32529748770505e-01,
    -0.21841717175414e-01, -0.52838357969930e-04, -0.47184321073267e-03,
    -0.30001780793026e-03,  0.47661393906987e-04, -0.44141845330846e-05,
    -0.72694996297594e-15, -0.31679644845054e-04, -0.28270797985312e-05,
    -0.85205128120103e-09, -0.22425281908000e-05, -0.65171222895601e-06,
    -0.14341729937924e-12, -0.40516996860117e-06, -0.12734301741682e-08,
    -0.17424871230634e-09, -0.68762131295531e-18,  0.14478307828521e-19,
     0.26335781662795e-22, -0.11947622640071e-22,  0.18228094581404e-23,
    -0.93537087292458e-25
};

// gamma1 and its derivatives w.r.t. pi and tau
static double gamma1(double pi, double tau) {
    double s = 0.0;
    double x = 7.1 - pi;
    double y = tau - 1.222;
    for (int i = 0; i < 34; ++i)
        s += r1_n[i] * std::pow(x, r1_I[i]) * std::pow(y, r1_J[i]);
    return s;
}

static double gamma1_pi(double pi, double tau) {
    double s = 0.0;
    double x = 7.1 - pi;
    double y = tau - 1.222;
    for (int i = 0; i < 34; ++i) {
        if (r1_I[i] == 0) continue;
        s += r1_n[i] * (-r1_I[i]) * std::pow(x, r1_I[i] - 1) * std::pow(y, r1_J[i]);
    }
    return s;
}

static double gamma1_tau(double pi, double tau) {
    double s = 0.0;
    double x = 7.1 - pi;
    double y = tau - 1.222;
    for (int i = 0; i < 34; ++i) {
        if (r1_J[i] == 0) continue;
        s += r1_n[i] * std::pow(x, r1_I[i]) * r1_J[i] * std::pow(y, r1_J[i] - 1);
    }
    return s;
}

static double gamma1_tautau(double pi, double tau) {
    double s = 0.0;
    double x = 7.1 - pi;
    double y = tau - 1.222;
    for (int i = 0; i < 34; ++i) {
        int j = r1_J[i];
        if (j == 0 || j == 1) continue;  // j*(j-1) == 0 for j=0 or j=1
        s += r1_n[i] * std::pow(x, r1_I[i]) * (double)j * (double)(j - 1) * std::pow(y, j - 2);
    }
    return s;
}

// ---------------------------------------------------------------------------
// Region 2 — ideal-gas (o) + residual (r) Gibbs free energy
// Reference points: p_star = 1 MPa,  T_star = 540 K
// ---------------------------------------------------------------------------
static constexpr double p_star2 = 1.0e6;
static constexpr double T_star2 = 540.0;

// Ideal-gas part (9 terms) — IAPWS-IF97 Table 10
static const int    r2o_J[9] = {0, 1, -5, -4, -3, -2, -1, 2, 3};
static const double r2o_n[9] = {
    -9.6927686500217e+00,  1.0086655968018e+01, -5.6087911283020e-03,
     7.1452738081455e-01, -4.0710498223928e+00,  1.4240819171444e+01,
    -4.3839511319450e+01, -2.8408632460772e-01,  2.1268463753307e-02
};

// Residual part (43 terms) — IAPWS-IF97 Table 11
static const int r2r_I[43] = {
    1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 4, 4, 4, 5, 6,
    6, 6, 7, 7, 7, 8, 8, 9,10,10,
   10,16,16,18,20,20,20,21,22,23,
   24,24,24
};
static const int r2r_J[43] = {
     0,  1,  2,  3,  6,  1,  2,  4,  7, 36,
     0,  1,  3,  6, 35,  1,  2,  3,  7,  3,
    16, 35,  0, 11, 25,  8, 36, 13,  4, 10,
    14, 29, 50, 57, 20, 35, 48, 21, 53, 39,
    26, 40, 58
};
static const double r2r_n[43] = {
    -1.7731742473213e-03, -1.7834862292358e-02, -4.5996013696365e-02,
    -5.7581259083432e-02, -5.0325278727930e-02, -3.3032641670203e-05,
    -1.8948987516315e-04, -3.9392777243355e-03, -4.3797295650573e-02,
    -2.6674547914087e-05,  2.0481737692309e-08,  4.3870667284435e-07,
    -3.2277677238570e-05, -1.5033924542148e-03, -4.0668253562950e-02,
    -7.8847309559367e-10,  1.2790717852285e-08,  4.8225372718507e-07,
     2.2922076337661e-06, -1.6714766451061e-11, -2.1171472321355e-03,
    -2.3895741934104e-01, -5.9059564324270e-18, -1.2621808899101e-06,
    -3.8946842435739e-02,  1.1256211360459e-11, -8.2311340897998e-02,
     1.9809712802088e-08,  1.0406965210174e-19, -1.0234747095929e-13,
    -1.0018179379511e-09, -8.0882908646985e-11,  1.0693031879409e+01,
    -2.0604452395820e-02, -1.0462281254338e-01, -1.0843106836260e-01,
    -1.2325566908690e-02,  1.0961403757226e-01, -2.5858298700453e-02,
    -1.1131268130073e-02, -9.3885085046455e-02, -3.3874135836052e-01,
     1.5662895099985e-01
};

// Ideal-gas part derivatives
static double g2o(double pi, double tau) {
    double s = std::log(pi);
    for (int i = 0; i < 9; ++i) s += r2o_n[i] * std::pow(tau, r2o_J[i]);
    return s;
}
static double g2o_pi(double pi, double /*tau*/) { return 1.0 / pi; }
static double g2o_tau(double /*pi*/, double tau) {
    double s = 0.0;
    for (int i = 0; i < 9; ++i)
        if (r2o_J[i] != 0)
            s += r2o_n[i] * r2o_J[i] * std::pow(tau, r2o_J[i] - 1);
    return s;
}
static double g2o_tautau(double /*pi*/, double tau) {
    double s = 0.0;
    for (int i = 0; i < 9; ++i) {
        int j = r2o_J[i];
        if (j == 0 || j == 1) continue;
        s += r2o_n[i] * j * (j - 1) * std::pow(tau, j - 2);
    }
    return s;
}

// Residual part derivatives
static double g2r(double pi, double tau) {
    double s = 0.0;
    double y = tau - 0.5;
    for (int i = 0; i < 43; ++i)
        s += r2r_n[i] * std::pow(pi, r2r_I[i]) * std::pow(y, r2r_J[i]);
    return s;
}
static double g2r_pi(double pi, double tau) {
    double s = 0.0;
    double y = tau - 0.5;
    for (int i = 0; i < 43; ++i)
        s += r2r_n[i] * r2r_I[i] * std::pow(pi, r2r_I[i] - 1) * std::pow(y, r2r_J[i]);
    return s;
}
static double g2r_tau(double pi, double tau) {
    double s = 0.0;
    double y = tau - 0.5;
    for (int i = 0; i < 43; ++i) {
        if (r2r_J[i] == 0) continue;
        s += r2r_n[i] * std::pow(pi, r2r_I[i]) * r2r_J[i] * std::pow(y, r2r_J[i] - 1);
    }
    return s;
}
static double g2r_tautau(double pi, double tau) {
    double s = 0.0;
    double y = tau - 0.5;
    for (int i = 0; i < 43; ++i) {
        int j = r2r_J[i];
        if (j == 0 || j == 1) continue;
        s += r2r_n[i] * std::pow(pi, r2r_I[i]) * (double)j * (double)(j - 1) * std::pow(y, j - 2);
    }
    return s;
}

// ---------------------------------------------------------------------------
// Saturation line — IAPWS-IF97 Eq. 30 (T -> p) and Eq. 31 (p -> T)
// ---------------------------------------------------------------------------
static const double sat_n[10] = {
     1.16705214527670e+03, -7.24213167032060e+05, -1.70738469400920e+01,
     1.20208247024700e+04, -3.23255503223330e+06,  1.49151086135300e+01,
    -4.82326573615910e+03,  4.05113405420570e+05, -2.38555575678490e-01,
     6.50175348447980e+02
};

// p_sat in MPa from T in K (Eq. 30)
static double p_sat_MPa(double T_K) {
    double th = T_K + sat_n[8] / (T_K - sat_n[9]);
    double A  =  th * th    + sat_n[0] * th + sat_n[1];
    double B  =  sat_n[2] * th * th + sat_n[3] * th + sat_n[4];
    double C  =  sat_n[5] * th * th + sat_n[6] * th + sat_n[7];
    double disc = B * B - 4.0 * A * C;
    if (disc < 0.0) disc = 0.0;
    double q = 2.0 * C / (-B + std::sqrt(disc));
    return q * q * q * q;  // [MPa]
}

// T_sat in K from p in MPa (Eq. 31 backward)
static double T_sat_K(double p_MPa) {
    double beta = std::pow(p_MPa, 0.25);
    double E    =  beta * beta + sat_n[2] * beta + sat_n[5];
    double F    =  sat_n[0] * beta * beta + sat_n[3] * beta + sat_n[6];
    double G    =  sat_n[1] * beta * beta + sat_n[4] * beta + sat_n[7];
    double disc = F * F - 4.0 * E * G;
    if (disc < 0.0) disc = 0.0;
    double D    = 2.0 * G / (-F - std::sqrt(disc));
    double inner = (sat_n[9] + D) * (sat_n[9] + D) - 4.0 * (sat_n[8] + sat_n[9] * D);
    if (inner < 0.0) inner = 0.0;
    return 0.5 * (sat_n[9] + D - std::sqrt(inner));
}

// ---------------------------------------------------------------------------
// Region 1 private helpers
// ---------------------------------------------------------------------------
double IAPWS_IF97::v1(double T, double p) {
    double pi  = p / p_star1;
    double tau = T_star1 / T;
    return R_water * T / p * pi * gamma1_pi(pi, tau);
}

double IAPWS_IF97::h1(double T, double p) {
    double pi  = p / p_star1;
    double tau = T_star1 / T;
    return R_water * T * tau * gamma1_tau(pi, tau);
}

double IAPWS_IF97::cp1(double T, double p) {
    double pi  = p / p_star1;
    double tau = T_star1 / T;
    return -R_water * tau * tau * gamma1_tautau(pi, tau);
}

// ---------------------------------------------------------------------------
// Region 2 private helpers
// ---------------------------------------------------------------------------
double IAPWS_IF97::v2(double T, double p) {
    double pi  = p / p_star2;
    double tau = T_star2 / T;
    return R_water * T / p * pi * (g2o_pi(pi, tau) + g2r_pi(pi, tau));
}

double IAPWS_IF97::h2(double T, double p) {
    double pi  = p / p_star2;
    double tau = T_star2 / T;
    return R_water * T * tau * (g2o_tau(pi, tau) + g2r_tau(pi, tau));
}

double IAPWS_IF97::cp2(double T, double p) {
    double pi  = p / p_star2;
    double tau = T_star2 / T;
    return -R_water * tau * tau * (g2o_tautau(pi, tau) + g2r_tautau(pi, tau));
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
double IAPWS_IF97::T_sat(double p_Pa) {
    // Clamp to valid saturation range [triple point, critical point]
    double p_MPa = std::max(611.657e-6, std::min(p_Pa * 1.0e-6, 22.064));
    return T_sat_K(p_MPa);
}

double IAPWS_IF97::p_sat(double T_K) {
    double T = std::max(273.15, std::min(T_K, 647.096));
    return p_sat_MPa(T) * 1.0e6;
}

double IAPWS_IF97::surface_tension(double T_K) {
    // IAPWS 1994 surface tension correlation
    double tau = 1.0 - T_K / Tc;
    if (tau <= 0.0) return 0.0;
    return 0.2358 * std::pow(tau, 1.256) * (1.0 - 0.625 * tau);
}

double IAPWS_IF97::mu_liquid(double T_K, double /*rho*/) {
    // Vogel-Fulcher-Tammann fit to IAPWS 2008 liquid viscosity (273-623 K)
    double T = std::max(273.15, std::min(T_K, 623.0));
    return 2.414e-5 * std::exp(247.8 / (T - 140.0));
}

double IAPWS_IF97::mu_vapor(double T_K, double /*rho*/) {
    // IAPWS 2008 low-density (dilute-gas) limit: mu0(T_bar)
    // mu0 [Pa.s] = 1e-6 * 100 * sqrt(T_bar) / sum(H_i / T_bar^i)
    static const double H[4] = {1.67752, 2.20462, 0.6366564, -0.241605};
    double T_bar = T_K / Tc;
    double denom = 0.0;
    for (int i = 0; i < 4; ++i)
        denom += H[i] / std::pow(T_bar, static_cast<double>(i));
    return 1.0e-4 * std::sqrt(T_bar) / denom;
}

double IAPWS_IF97::k_liquid(double T_K, double rho) {
    // IAPWS 2011 thermal conductivity, liquid branch
    // Lambda_0 (dilute gas): sqrt(T_bar) / sum(L0_i / T_bar^i)  [W/(m.K)] after scaling
    double T_bar   = T_K / Tc;
    double rho_bar = rho / rho_c;

    static const double L0[5] = {
        2.443221e-3, 1.323095e-2, 6.770357e-3, -3.454586e-3, 4.096266e-4
    };
    double denom0 = 0.0;
    for (int i = 0; i < 5; ++i)
        denom0 += L0[i] / std::pow(T_bar, static_cast<double>(i));
    double lam0 = std::sqrt(T_bar) / denom0;  // [W/(m.K)]

    // Lambda_1 polynomial in (1/T_bar - 1) and (rho_bar - 1)
    static const double L1[5][6] = {
        { 1.60397357, -0.646013523,  0.111443906,  0.102997357, -0.0504123634,  0.00609859258},
        { 2.33771842,  1.27380934,  -0.378125153,  0.0,          0.0,           0.0          },
        { 2.19650529, -0.981280100,  0.178976112,  0.0,          0.0,           0.0          },
        {-1.21051378,  0.0,          0.0,          0.0,          0.0,           0.0          },
        {-2.72166500,  0.0,          0.0,          0.0,          0.0,           0.0          }
    };
    double lam1 = 0.0;
    double u = 1.0 / T_bar - 1.0;
    double v = rho_bar - 1.0;
    double up = 1.0;
    for (int i = 0; i < 5; ++i, up *= u) {
        double vp = 1.0;
        for (int j = 0; j < 6; ++j, vp *= v)
            lam1 += L1[i][j] * up * vp;
    }
    return lam0 * std::exp(rho_bar * lam1);
}

double IAPWS_IF97::k_vapor(double T_K, double rho) {
    // Same IAPWS 2011 framework; at low rho the exponential factor -> 1
    double T_bar   = T_K / Tc;
    double rho_bar = rho / rho_c;

    static const double L0[5] = {
        2.443221e-3, 1.323095e-2, 6.770357e-3, -3.454586e-3, 4.096266e-4
    };
    double denom0 = 0.0;
    for (int i = 0; i < 5; ++i)
        denom0 += L0[i] / std::pow(T_bar, static_cast<double>(i));
    double lam0 = std::sqrt(T_bar) / denom0;

    static const double L1[5][6] = {
        { 1.60397357, -0.646013523,  0.111443906,  0.102997357, -0.0504123634,  0.00609859258},
        { 2.33771842,  1.27380934,  -0.378125153,  0.0,          0.0,           0.0          },
        { 2.19650529, -0.981280100,  0.178976112,  0.0,          0.0,           0.0          },
        {-1.21051378,  0.0,          0.0,          0.0,          0.0,           0.0          },
        {-2.72166500,  0.0,          0.0,          0.0,          0.0,           0.0          }
    };
    double lam1 = 0.0;
    double u = 1.0 / T_bar - 1.0;
    double v = rho_bar - 1.0;
    double up = 1.0;
    for (int i = 0; i < 5; ++i, up *= u) {
        double vp = 1.0;
        for (int j = 0; j < 6; ++j, vp *= v)
            lam1 += L1[i][j] * up * vp;
    }
    return lam0 * std::exp(rho_bar * lam1);
}

double IAPWS_IF97::h_fg(double p_Pa) {
    double T  = T_sat(p_Pa);
    double Tc = std::max(273.15, std::min(T, 647.09));
    double pc_clamped = std::max(1.0e3, std::min(p_Pa, ::twofluid::pc - 1.0));
    return std::max(0.0, h2(Tc, pc_clamped) - h1(Tc, pc_clamped));
}

SteamProperties IAPWS_IF97::liquid(double T_K, double p_Pa) {
    double T   = std::max(273.15, std::min(T_K,  623.15));
    double p   = std::max(1.0e5,  std::min(p_Pa, 100.0e6));
    double rho = 1.0 / v1(T, p);
    return {T, p, rho, h1(T, p), cp1(T, p), mu_liquid(T, rho), k_liquid(T, rho), surface_tension(T)};
}

SteamProperties IAPWS_IF97::vapor(double T_K, double p_Pa) {
    double T   = std::max(273.15, std::min(T_K,  1073.15));
    double p   = std::max(1.0e3,  std::min(p_Pa, 100.0e6));
    double rho = 1.0 / v2(T, p);
    return {T, p, rho, h2(T, p), cp2(T, p), mu_vapor(T, rho), k_vapor(T, rho), 0.0};
}

} // namespace twofluid
