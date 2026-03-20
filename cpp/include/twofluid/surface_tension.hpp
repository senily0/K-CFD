#pragma once

#include <Eigen/Dense>
#include "twofluid/mesh.hpp"
#include "twofluid/fields.hpp"

namespace twofluid {

/// Continuum Surface Force (CSF) model for surface tension.
/// Brackbill et al. (1992)
/// F_sigma = sigma * kappa * grad(alpha)
/// where kappa = -div(n_hat), n_hat = grad(alpha) / |grad(alpha)|
class CSFSurfaceTension {
public:
    CSFSurfaceTension(const FVMesh& mesh, double sigma);

    double sigma;  // surface tension coefficient [N/m]

    /// Compute surface tension force per unit volume [N/m^3].
    /// Returns (n_cells, ndim) matrix of force vectors.
    Eigen::MatrixXd compute_force(const ScalarField& alpha) const;

    /// Compute interface curvature [1/m] per cell.
    /// kappa = -div(grad(alpha) / |grad(alpha)|)
    Eigen::VectorXd compute_curvature(const ScalarField& alpha) const;

    /// Compute smoothed interface delta function |grad(alpha)| per cell.
    Eigen::VectorXd compute_delta(const ScalarField& alpha) const;

private:
    const FVMesh& mesh_;
};

} // namespace twofluid
