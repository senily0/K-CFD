#include "twofluid/time_control.hpp"
#include <algorithm>
#include <cmath>

namespace twofluid {

AdaptiveTimeControl::AdaptiveTimeControl(
    double dt_init, double dt_min, double dt_max,
    double cfl_target, double cfl_max, double fourier_max,
    double growth_factor, double shrink_factor, double safety_factor)
    : dt_(dt_init), dt_min_(dt_min), dt_max_(dt_max),
      cfl_target_(cfl_target), cfl_max_val_(cfl_max),
      fourier_max_(fourier_max),
      growth_factor_(growth_factor), shrink_factor_(shrink_factor),
      safety_factor_(safety_factor) {}

std::pair<double, Eigen::VectorXd>
AdaptiveTimeControl::compute_cfl(const FVMesh& mesh,
                                  const Eigen::VectorXd& u_mag,
                                  double dt) const {
    int n = mesh.n_cells;
    double inv_ndim = 1.0 / mesh.ndim;
    Eigen::VectorXd cfl(n);

    for (int i = 0; i < n; ++i) {
        double dx = std::pow(std::max(mesh.cells[i].volume, 1e-30), inv_ndim);
        cfl(i) = u_mag(i) * dt / dx;
    }
    return {cfl.maxCoeff(), cfl};
}

std::pair<double, Eigen::VectorXd>
AdaptiveTimeControl::compute_fourier(const FVMesh& mesh,
                                      double alpha_diff,
                                      double dt) const {
    int n = mesh.n_cells;
    double inv_ndim = 1.0 / mesh.ndim;
    Eigen::VectorXd fo(n);

    for (int i = 0; i < n; ++i) {
        double dx = std::pow(std::max(mesh.cells[i].volume, 1e-30), inv_ndim);
        fo(i) = alpha_diff * dt / (dx * dx);
    }
    return {fo.maxCoeff(), fo};
}

TimeStepInfo AdaptiveTimeControl::compute_dt(
    const FVMesh& mesh, const Eigen::VectorXd& u_mag,
    double alpha_diff, bool converged) {

    int n = mesh.n_cells;
    double inv_ndim = 1.0 / mesh.ndim;

    // CFL constraint: dt_cfl = cfl_target * dx / |u|
    double dt_cfl = dt_max_;
    for (int i = 0; i < n; ++i) {
        double dx = std::pow(std::max(mesh.cells[i].volume, 1e-30), inv_ndim);
        double u = std::max(u_mag(i), 1e-30);
        double dt_i = cfl_target_ * dx / u;
        dt_cfl = std::min(dt_cfl, dt_i);
    }
    dt_cfl *= safety_factor_;

    double dt_new = dt_cfl;
    std::string limited_by = "cfl";

    // Fourier constraint
    if (alpha_diff > 0.0) {
        double dt_fo = dt_max_;
        for (int i = 0; i < n; ++i) {
            double dx = std::pow(std::max(mesh.cells[i].volume, 1e-30), inv_ndim);
            double dt_i = fourier_max_ * dx * dx / std::max(alpha_diff, 1e-30);
            dt_fo = std::min(dt_fo, dt_i);
        }
        dt_fo *= safety_factor_;
        if (dt_fo < dt_new) {
            dt_new = dt_fo;
            limited_by = "fourier";
        }
    }

    // Growth limit
    dt_new = std::min(dt_new, growth_factor_ * dt_);

    // Shrink on divergence
    if (!converged) {
        dt_new = shrink_factor_ * dt_;
        limited_by = "divergence";
    }

    // Clip
    dt_new = std::max(dt_min_, std::min(dt_new, dt_max_));

    // Compute actual CFL at new dt
    auto [cfl_max, cfl_arr] = compute_cfl(mesh, u_mag, dt_new);
    double fo_max = 0.0;
    if (alpha_diff > 0.0) {
        auto [fm, fa] = compute_fourier(mesh, alpha_diff, dt_new);
        fo_max = fm;
    }

    dt_ = dt_new;
    dt_history.push_back(dt_new);
    cfl_history.push_back(cfl_max);
    fourier_history.push_back(fo_max);

    return {cfl_max, fo_max, limited_by, dt_new};
}

} // namespace twofluid
