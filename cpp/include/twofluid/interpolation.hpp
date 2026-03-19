#pragma once

#include <string>
#include <Eigen/Dense>
#include "twofluid/mesh.hpp"
#include "twofluid/fields.hpp"

namespace twofluid {

/// Compute face mass flux from velocity field and density.
/// F_f = rho * (u_f . n_f) * A_f
Eigen::VectorXd compute_mass_flux(const VectorField& U, double rho,
                                  const FVMesh& mesh);

// =====================================================================
// TVD limiters (scalar versions)
// =====================================================================

double limiter_minmod(double r);
double limiter_van_leer(double r);
double limiter_superbee(double r);
double limiter_van_albada(double r);

// =====================================================================
// MUSCL deferred correction
// =====================================================================

/// Compute MUSCL deferred correction source term.
/// Returns (n_cells,) correction vector to be added to RHS.
///
/// The deferred correction pattern:
///   phi_f^{MUSCL} = phi_f^{UW} + psi(r) * (phi_f^{HO} - phi_f^{UW})
///
/// The implicit part uses upwind (in the matrix), high-order correction goes to RHS.
Eigen::VectorXd muscl_deferred_correction(
    const FVMesh& mesh,
    const ScalarField& phi,
    const Eigen::VectorXd& mass_flux,
    const Eigen::MatrixXd& grad_phi,
    const std::string& limiter_name = "van_leer");

namespace detail {

/// Owner-cell interpolation weight for an internal face.
double interp_weight(const FVMesh& mesh, int face_id);

/// Boundary face scalar value lookup.
double get_boundary_value(const ScalarField& phi, int face_id);

/// Boundary face vector value lookup.
Eigen::VectorXd get_boundary_vector_value(const VectorField& U, int face_id);

} // namespace detail

} // namespace twofluid
