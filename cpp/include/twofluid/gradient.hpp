#pragma once

#include <Eigen/Dense>
#include "twofluid/mesh.hpp"
#include "twofluid/fields.hpp"

namespace twofluid {

/// Green-Gauss gradient reconstruction.
/// Returns (n_cells, ndim) gradient matrix.
Eigen::MatrixXd green_gauss_gradient(const ScalarField& phi);

/// Least Squares gradient reconstruction.
/// Returns (n_cells, ndim) gradient matrix.
Eigen::MatrixXd least_squares_gradient(const ScalarField& phi);

namespace detail {

/// Interpolation weight for internal face (owner-side weight).
double interpolation_weight(const FVMesh& mesh, int face_id);

/// Boundary face scalar value lookup.
double get_boundary_face_value(const ScalarField& phi, int face_id);

} // namespace detail

/// Apply Barth-Jespersen gradient limiter to prevent oscillations.
/// Limits the gradient so reconstructed face values stay within
/// the min/max of neighboring cell values.
/// Returns per-cell limiter values phi_i in [0,1].
Eigen::VectorXd barth_jespersen_limiter(const FVMesh& mesh,
                                         const ScalarField& phi,
                                         const Eigen::MatrixXd& grad_phi);

/// Apply Venkatakrishnan gradient limiter (smoother than B-J).
Eigen::VectorXd venkatakrishnan_limiter(const FVMesh& mesh,
                                          const ScalarField& phi,
                                          const Eigen::MatrixXd& grad_phi,
                                          double epsilon = 1e-6);

/// Compute limited gradient: grad_limited = limiter * grad
Eigen::MatrixXd limit_gradient(const Eigen::MatrixXd& grad_phi,
                                 const Eigen::VectorXd& limiter);

} // namespace twofluid
