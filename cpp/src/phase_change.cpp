#include "twofluid/phase_change.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>

namespace twofluid {

// --- Utility functions ---

double saturation_temperature(double P_Pa) {
    double P_bar = P_Pa / 1e5;
    if (P_bar < 0.01) P_bar = 0.01;

    const double A = 5.0768, B = 1659.793, C = 227.1;
    double log_p = std::log10(P_bar);
    double denom = A - log_p;
    if (denom <= 0.0) return 647.0;

    double T_C = B / denom - C;
    double T_K = T_C + 273.15;
    return std::max(273.15, std::min(T_K, 647.0));
}

double water_latent_heat(double P_Pa) {
    double P_MPa = P_Pa / 1e6;
    const double P_crit = 22.064;
    if (P_MPa >= P_crit) return 0.0;

    double h_fg = 2.257e6 * std::pow(1.0 - P_MPa / P_crit, 0.38);
    return std::max(h_fg, 0.0);
}

WaterProperties water_properties(double P_Pa) {
    double T_sat = saturation_temperature(P_Pa);
    double h_fg = water_latent_heat(P_Pa);

    double rho_l = 1000.0 - 0.5 * (T_sat - 373.15);
    rho_l = std::max(400.0, std::min(rho_l, 1050.0));

    const double R_steam = 461.5;
    double rho_g = (T_sat > 0.0) ? P_Pa / (R_steam * T_sat) : 1.0;
    rho_g = std::max(0.01, std::min(rho_g, 500.0));

    return {T_sat, h_fg, rho_l, rho_g, 4200.0, 2.8e-4, 0.68};
}

// --- Lee Phase Change Model ---

LeePhaseChangeModel::LeePhaseChangeModel(
    const FVMesh& mesh, double T_sat, double r_evap, double r_cond,
    double L_latent, double rho_l, double rho_g)
    : T_sat(T_sat), r_evap(r_evap), r_cond(r_cond),
      L_latent(L_latent), rho_l(rho_l), rho_g(rho_g), mesh_(mesh) {}

Eigen::VectorXd LeePhaseChangeModel::compute_mass_transfer(
    const ScalarField& T, const ScalarField& alpha_l) const {
    int n = mesh_.n_cells;
    Eigen::VectorXd dot_m = Eigen::VectorXd::Zero(n);

    for (int i = 0; i < n; ++i) {
        double T_val = T.values(i);
        double al = alpha_l.values(i);
        double ag = 1.0 - al;

        if (T_val > T_sat) {
            // Evaporation
            dot_m(i) = r_evap * rho_l * al * (T_val - T_sat) / T_sat;
        } else if (T_val < T_sat) {
            // Condensation
            dot_m(i) = -r_cond * rho_g * ag * (T_sat - T_val) / T_sat;
        }
    }
    return dot_m;
}

LeePhaseChangeModel::Sources LeePhaseChangeModel::get_source_terms(
    const ScalarField& T, const ScalarField& alpha_l) const {
    Eigen::VectorXd dot_m = compute_mass_transfer(T, alpha_l);
    Sources s;
    s.alpha_l = -dot_m / rho_l;
    s.alpha_g = dot_m / rho_g;
    s.energy = -dot_m * L_latent;
    return s;
}

// --- Rohsenow Boiling Model ---

RohsenowBoilingModel::RohsenowBoilingModel(
    double T_sat, double h_fg, double rho_l, double rho_g,
    double mu_l, double cp_l, double sigma, double Pr_l,
    double C_sf, double n, double g)
    : T_sat(T_sat), h_fg(h_fg), rho_l(rho_l), rho_g(rho_g),
      mu_l(mu_l), cp_l(cp_l), sigma(sigma), Pr_l(Pr_l),
      C_sf(C_sf), n(n), g(g) {}

double RohsenowBoilingModel::compute_wall_heat_flux(double T_wall) const {
    double dT = T_wall - T_sat;
    if (dT <= 0.0) return 0.0;

    double term1 = mu_l * h_fg;
    double term2 = std::sqrt(g * (rho_l - rho_g) / sigma);
    double term3 = std::pow(cp_l * dT / (C_sf * h_fg * std::pow(Pr_l, n)), 3.0);

    return term1 * term2 * term3;
}

double RohsenowBoilingModel::compute_mass_transfer_wall(
    double T_wall, double A_wall, double V_cell) const {
    double q = compute_wall_heat_flux(T_wall);
    if (V_cell <= 0.0 || h_fg <= 0.0) return 0.0;
    return q * A_wall / (h_fg * V_cell);
}

// --- Zuber CHF Model ---

ZuberCHFModel::ZuberCHFModel(double h_fg, double rho_l, double rho_g,
                               double sigma, double g)
    : h_fg(h_fg), rho_l(rho_l), rho_g(rho_g), sigma(sigma), g(g) {}

double ZuberCHFModel::compute_chf() const {
    double term = std::pow(
        sigma * g * (rho_l - rho_g) / (rho_g * rho_g), 0.25);
    return 0.131 * rho_g * h_fg * term;
}

CHFMargin ZuberCHFModel::check_margin(double q_wall) const {
    double chf = compute_chf();
    double ratio = (chf > 0.0) ? q_wall / chf
                                : std::numeric_limits<double>::infinity();
    if (ratio >= 1.0) {
        std::cerr << "[CHF WARNING] Wall heat flux (" << q_wall
                  << " W/m2) reached CHF (" << chf << " W/m2)\n";
    }
    return {chf, ratio, ratio < 1.0};
}

// --- Nusselt Condensation Model ---

NusseltCondensationModel::NusseltCondensationModel(
    double T_sat, double h_fg, double rho_l, double rho_g,
    double mu_l, double k_l, double g)
    : T_sat(T_sat), h_fg(h_fg), rho_l(rho_l), rho_g(rho_g),
      mu_l(mu_l), k_l(k_l), g(g) {}

double NusseltCondensationModel::compute_heat_transfer_coeff(
    double L_plate, double delta_T_sub) const {
    if (delta_T_sub <= 0.0 || L_plate <= 0.0) return 0.0;

    double numer = rho_l * (rho_l - rho_g) * g * h_fg
                   * k_l * k_l * k_l;
    double denom = mu_l * L_plate * delta_T_sub;
    if (denom <= 0.0) return 0.0;

    return 0.943 * std::pow(numer / denom, 0.25);
}

double NusseltCondensationModel::compute_condensation_rate(
    double L_plate, double T_wall, double A_wall, double V_cell) const {
    double dT_sub = T_sat - T_wall;
    if (dT_sub <= 0.0) return 0.0;

    double h = compute_heat_transfer_coeff(L_plate, dT_sub);
    double q = h * dT_sub;

    if (V_cell <= 0.0 || h_fg <= 0.0) return 0.0;
    return -q * A_wall / (h_fg * V_cell);
}

// --- Phase Change Manager ---

PhaseChangeManager::PhaseChangeManager(
    const FVMesh& mesh, double T_sat, double h_fg,
    double rho_l, double rho_g)
    : mesh_(mesh), T_sat_(T_sat), h_fg_(h_fg), rho_l_(rho_l), rho_g_(rho_g),
      lee_model_(mesh, T_sat, 0.1, 0.1, h_fg, rho_l, rho_g) {}

void PhaseChangeManager::set_lee_params(double r_evap, double r_cond) {
    lee_model_.r_evap = r_evap;
    lee_model_.r_cond = r_cond;
}

void PhaseChangeManager::enable_boiling(
    double mu_l, double cp_l, double sigma, double Pr_l,
    double C_sf, double n) {
    boiling_model_ = std::make_unique<RohsenowBoilingModel>(
        T_sat_, h_fg_, rho_l_, rho_g_, mu_l, cp_l, sigma, Pr_l, C_sf, n);
}

void PhaseChangeManager::enable_condensation(double mu_l, double k_l) {
    condensation_model_ = std::make_unique<NusseltCondensationModel>(
        T_sat_, h_fg_, rho_l_, rho_g_, mu_l, k_l);
}

Eigen::VectorXd PhaseChangeManager::compute_total_mass_transfer(
    const ScalarField& T, const ScalarField& alpha_l,
    const std::vector<int>& wall_cells, double T_wall,
    double A_wall, double V_cell, double L_plate) const {

    Eigen::VectorXd dot_m = lee_model_.compute_mass_transfer(T, alpha_l);

    // Wall boiling (Rohsenow)
    if (boiling_model_ && !wall_cells.empty() && T_wall > T_sat_) {
        for (int ci : wall_cells) {
            dot_m(ci) += boiling_model_->compute_mass_transfer_wall(
                T_wall, A_wall, V_cell);
        }
    }

    // Wall condensation (Nusselt)
    if (condensation_model_ && !wall_cells.empty() && T_wall < T_sat_) {
        for (int ci : wall_cells) {
            dot_m(ci) += condensation_model_->compute_condensation_rate(
                L_plate, T_wall, A_wall, V_cell);
        }
    }

    return dot_m;
}

PhaseChangeSources PhaseChangeManager::get_source_terms(
    const ScalarField& T, const ScalarField& alpha_l,
    const std::vector<int>& wall_cells, double T_wall,
    double A_wall, double V_cell, double L_plate) const {

    Eigen::VectorXd dot_m = compute_total_mass_transfer(
        T, alpha_l, wall_cells, T_wall, A_wall, V_cell, L_plate);

    return {-dot_m / rho_l_, dot_m / rho_g_, -dot_m * h_fg_};
}

} // namespace twofluid
