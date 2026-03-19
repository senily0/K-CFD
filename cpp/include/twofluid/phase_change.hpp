#pragma once

#include <memory>
#include <vector>
#include <Eigen/Dense>
#include "twofluid/mesh.hpp"
#include "twofluid/fields.hpp"

namespace twofluid {

// --- Utility functions ---

struct WaterProperties {
    double T_sat, h_fg, rho_l, rho_g, cp_l, mu_l, k_l;
};

double saturation_temperature(double P_Pa);
double water_latent_heat(double P_Pa);
WaterProperties water_properties(double P_Pa);

// --- Lee Phase Change Model ---

class LeePhaseChangeModel {
public:
    LeePhaseChangeModel(const FVMesh& mesh, double T_sat = 373.15,
                         double r_evap = 0.1, double r_cond = 0.1,
                         double L_latent = 2.26e6,
                         double rho_l = 1000.0, double rho_g = 1.0);

    Eigen::VectorXd compute_mass_transfer(const ScalarField& T,
                                           const ScalarField& alpha_l) const;

    struct Sources {
        Eigen::VectorXd alpha_l, alpha_g, energy;
    };
    Sources get_source_terms(const ScalarField& T,
                             const ScalarField& alpha_l) const;

    double T_sat, r_evap, r_cond, L_latent, rho_l, rho_g;

private:
    const FVMesh& mesh_;
};

// --- Rohsenow Boiling Model ---

class RohsenowBoilingModel {
public:
    RohsenowBoilingModel(double T_sat, double h_fg,
                          double rho_l, double rho_g,
                          double mu_l, double cp_l,
                          double sigma, double Pr_l,
                          double C_sf = 0.013, double n = 1.0,
                          double g = 9.81);

    double compute_wall_heat_flux(double T_wall) const;
    double compute_mass_transfer_wall(double T_wall, double A_wall,
                                       double V_cell) const;

    double T_sat, h_fg, rho_l, rho_g, mu_l, cp_l;
    double sigma, Pr_l, C_sf, n, g;
};

// --- Zuber CHF Model ---

struct CHFMargin {
    double chf, ratio;
    bool safe;
};

class ZuberCHFModel {
public:
    ZuberCHFModel(double h_fg, double rho_l, double rho_g,
                   double sigma, double g = 9.81);

    double compute_chf() const;
    CHFMargin check_margin(double q_wall) const;

    double h_fg, rho_l, rho_g, sigma, g;
};

// --- Nusselt Condensation Model ---

class NusseltCondensationModel {
public:
    NusseltCondensationModel(double T_sat, double h_fg,
                              double rho_l, double rho_g,
                              double mu_l, double k_l,
                              double g = 9.81);

    double compute_heat_transfer_coeff(double L_plate, double delta_T_sub) const;
    double compute_condensation_rate(double L_plate, double T_wall,
                                      double A_wall, double V_cell) const;

    double T_sat, h_fg, rho_l, rho_g, mu_l, k_l, g;
};

// --- Phase Change Manager ---

struct PhaseChangeSources {
    Eigen::VectorXd alpha_l, alpha_g, energy;
};

class PhaseChangeManager {
public:
    PhaseChangeManager(const FVMesh& mesh, double T_sat = 373.15,
                        double h_fg = 2.26e6,
                        double rho_l = 1000.0, double rho_g = 1.0);

    void set_lee_params(double r_evap, double r_cond);
    void enable_boiling(double mu_l, double cp_l, double sigma,
                        double Pr_l, double C_sf = 0.013, double n = 1.0);
    void enable_condensation(double mu_l, double k_l);

    Eigen::VectorXd compute_total_mass_transfer(
        const ScalarField& T, const ScalarField& alpha_l,
        const std::vector<int>& wall_cells = {},
        double T_wall = 0.0, double A_wall = 1.0,
        double V_cell = 1.0, double L_plate = 0.1) const;

    PhaseChangeSources get_source_terms(
        const ScalarField& T, const ScalarField& alpha_l,
        const std::vector<int>& wall_cells = {},
        double T_wall = 0.0, double A_wall = 1.0,
        double V_cell = 1.0, double L_plate = 0.1) const;

private:
    const FVMesh& mesh_;
    double T_sat_, h_fg_, rho_l_, rho_g_;
    LeePhaseChangeModel lee_model_;
    std::unique_ptr<RohsenowBoilingModel> boiling_model_;
    std::unique_ptr<NusseltCondensationModel> condensation_model_;
};

} // namespace twofluid
