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

} // namespace twofluid
