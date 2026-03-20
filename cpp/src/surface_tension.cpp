#include "twofluid/surface_tension.hpp"
#include "twofluid/gradient.hpp"
#include <cmath>

namespace twofluid {

CSFSurfaceTension::CSFSurfaceTension(const FVMesh& mesh, double sigma_)
    : sigma(sigma_), mesh_(mesh)
{}

// ---------------------------------------------------------------------------
// compute_delta: |grad(alpha)| per cell
// ---------------------------------------------------------------------------
Eigen::VectorXd CSFSurfaceTension::compute_delta(const ScalarField& alpha) const
{
    const int n = mesh_.n_cells;
    Eigen::MatrixXd grad = green_gauss_gradient(alpha);  // (n, ndim)
    Eigen::VectorXd delta(n);
    for (int i = 0; i < n; ++i) {
        delta(i) = grad.row(i).norm();
    }
    return delta;
}

// ---------------------------------------------------------------------------
// compute_curvature: kappa = -div(n_hat)
// n_hat = grad(alpha) / |grad(alpha)|
// Divergence estimated via Green-Gauss on each component of n_hat.
// ---------------------------------------------------------------------------
Eigen::VectorXd CSFSurfaceTension::compute_curvature(const ScalarField& alpha) const
{
    const int n = mesh_.n_cells;
    const int ndim = mesh_.ndim;
    const double eps = 1e-30;

    Eigen::MatrixXd grad = green_gauss_gradient(alpha);  // (n, ndim)

    // Build unit-normal components as ScalarFields so we can
    // run green_gauss_gradient on each one.
    std::vector<ScalarField> nhat_fields;
    nhat_fields.reserve(ndim);
    for (int d = 0; d < ndim; ++d) {
        ScalarField sf(mesh_, "nhat_" + std::to_string(d), 0.0);
        for (int i = 0; i < n; ++i) {
            double mag = grad.row(i).norm();
            sf.values(i) = (mag > eps) ? grad(i, d) / mag : 0.0;
        }
        nhat_fields.push_back(std::move(sf));
    }

    // kappa = -div(n_hat) = -sum_d  d(n_hat_d)/dx_d
    Eigen::VectorXd kappa = Eigen::VectorXd::Zero(n);
    for (int d = 0; d < ndim; ++d) {
        Eigen::MatrixXd gnhat = green_gauss_gradient(nhat_fields[d]);  // (n, ndim)
        for (int i = 0; i < n; ++i) {
            kappa(i) -= gnhat(i, d);  // diagonal entry = d/dx_d
        }
    }
    return kappa;
}

// ---------------------------------------------------------------------------
// compute_force: F = sigma * kappa * grad(alpha)  [N/m^3]
// Returns (n_cells, ndim) matrix.
// ---------------------------------------------------------------------------
Eigen::MatrixXd CSFSurfaceTension::compute_force(const ScalarField& alpha) const
{
    const int n = mesh_.n_cells;
    const int ndim = mesh_.ndim;

    Eigen::MatrixXd grad = green_gauss_gradient(alpha);  // (n, ndim)
    Eigen::VectorXd kappa = compute_curvature(alpha);

    Eigen::MatrixXd F(n, ndim);
    for (int i = 0; i < n; ++i) {
        F.row(i) = sigma * kappa(i) * grad.row(i);
    }
    return F;
}

} // namespace twofluid
