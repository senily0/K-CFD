#pragma once

#include <string>
#include <vector>
#include <Eigen/Dense>
#include "twofluid/mesh.hpp"

namespace twofluid {

struct TimeStepInfo {
    double cfl_max = 0.0;
    double fourier_max = 0.0;
    std::string dt_limited_by;
    double dt = 0.0;
};

class AdaptiveTimeControl {
public:
    AdaptiveTimeControl(double dt_init, double dt_min = 1e-8, double dt_max = 1.0,
                        double cfl_target = 0.5, double cfl_max = 1.0,
                        double fourier_max = 0.5,
                        double growth_factor = 1.2, double shrink_factor = 0.5,
                        double safety_factor = 0.9);

    double dt() const { return dt_; }

    /// Compute CFL number per cell.
    std::pair<double, Eigen::VectorXd> compute_cfl(
        const FVMesh& mesh, const Eigen::VectorXd& u_mag, double dt) const;

    /// Compute Fourier number per cell.
    std::pair<double, Eigen::VectorXd> compute_fourier(
        const FVMesh& mesh, double alpha_diff, double dt) const;

    /// Compute next time step.
    TimeStepInfo compute_dt(const FVMesh& mesh, const Eigen::VectorXd& u_mag,
                            double alpha_diff = -1.0, bool converged = true);

    std::vector<double> dt_history, cfl_history, fourier_history;

private:
    double dt_, dt_min_, dt_max_;
    double cfl_target_, cfl_max_val_, fourier_max_;
    double growth_factor_, shrink_factor_, safety_factor_;
};

} // namespace twofluid
