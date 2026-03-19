#include "twofluid/interpolation.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace twofluid {

namespace detail {

double interp_weight(const FVMesh& mesh, int face_id) {
    const Face& face = mesh.faces[face_id];
    const Eigen::Vector3d& xO = mesh.cells[face.owner].center;
    const Eigen::Vector3d& xN = mesh.cells[face.neighbour].center;
    const Eigen::Vector3d& xF = face.center;

    double dO = (xF - xO).norm();
    double dN = (xF - xN).norm();
    double total = dO + dN;
    if (total < 1e-30) {
        return 0.5;
    }
    return dN / total;
}

double get_boundary_value(const ScalarField& phi, int face_id) {
    const FVMesh& mesh = phi.mesh();
    auto cache_it = mesh.boundary_face_cache.find(face_id);
    if (cache_it != mesh.boundary_face_cache.end()) {
        auto& [bname, li] = cache_it->second;
        auto bv_it = phi.boundary_values.find(bname);
        if (bv_it != phi.boundary_values.end()) {
            return bv_it->second[li];
        }
    }
    // Fallback: owner cell value
    return phi.values[mesh.faces[face_id].owner];
}

Eigen::VectorXd get_boundary_vector_value(const VectorField& U, int face_id) {
    const FVMesh& mesh = U.mesh();
    int ndim = mesh.ndim;
    auto cache_it = mesh.boundary_face_cache.find(face_id);
    if (cache_it != mesh.boundary_face_cache.end()) {
        auto& [bname, li] = cache_it->second;
        auto bv_it = U.boundary_values.find(bname);
        if (bv_it != U.boundary_values.end()) {
            return bv_it->second.row(li).transpose();
        }
    }
    // Fallback: owner cell value
    int owner = mesh.faces[face_id].owner;
    return U.values.row(owner).head(ndim);
}

} // namespace detail

// =====================================================================
// Mass flux computation
// =====================================================================

Eigen::VectorXd compute_mass_flux(const VectorField& U, double rho,
                                  const FVMesh& mesh) {
    int n_faces = mesh.n_faces;
    int ndim = mesh.ndim;
    Eigen::VectorXd mass_flux = Eigen::VectorXd::Zero(n_faces);

    for (int fid = 0; fid < n_faces; ++fid) {
        const Face& face = mesh.faces[fid];
        Eigen::VectorXd u_f;

        if (face.neighbour >= 0) {
            // Internal face: weighted interpolation
            double gc = detail::interp_weight(mesh, fid);
            u_f = gc * U.values.row(face.owner).head(ndim).transpose()
                  + (1.0 - gc) * U.values.row(face.neighbour).head(ndim).transpose();
        } else {
            // Boundary face
            u_f = detail::get_boundary_vector_value(U, fid);
        }

        // F_f = rho * (u_f . n_f) * A_f
        double dot = 0.0;
        for (int d = 0; d < ndim; ++d) {
            dot += u_f[d] * face.normal[d];
        }
        mass_flux[fid] = rho * dot * face.area;
    }

    return mass_flux;
}

// =====================================================================
// TVD limiters (scalar versions)
// =====================================================================

double limiter_minmod(double r) {
    // max(0, min(1, r))
    return std::max(0.0, std::min(1.0, r));
}

double limiter_van_leer(double r) {
    // (r + |r|) / (1 + |r|), but 0 if r <= 0
    if (r > 0.0) {
        return 2.0 * r / (1.0 + r);
    }
    return 0.0;
}

double limiter_superbee(double r) {
    // max(0, min(2r, 1), min(r, 2))
    return std::max(0.0, std::max(std::min(2.0 * r, 1.0),
                                   std::min(r, 2.0)));
}

double limiter_van_albada(double r) {
    // (r^2 + r) / (r^2 + 1), but 0 if r <= 0
    if (r > 0.0) {
        return (r * r + r) / (r * r + 1.0);
    }
    return 0.0;
}

/// Select a limiter function by name.
using LimiterFn = double(*)(double);

static LimiterFn get_limiter(const std::string& name) {
    if (name == "minmod")     return limiter_minmod;
    if (name == "van_leer")   return limiter_van_leer;
    if (name == "superbee")   return limiter_superbee;
    if (name == "van_albada") return limiter_van_albada;
    throw std::invalid_argument("Unknown limiter: " + name
        + ". Available: minmod, van_leer, superbee, van_albada");
}

// =====================================================================
// MUSCL deferred correction
// =====================================================================

Eigen::VectorXd muscl_deferred_correction(
    const FVMesh& mesh,
    const ScalarField& phi,
    const Eigen::VectorXd& mass_flux,
    const Eigen::MatrixXd& grad_phi,
    const std::string& limiter_name) {

    LimiterFn limiter_func = get_limiter(limiter_name);
    int n_cells = mesh.n_cells;
    int ndim = mesh.ndim;
    Eigen::VectorXd correction = Eigen::VectorXd::Zero(n_cells);

    for (int fid = 0; fid < mesh.n_faces; ++fid) {
        const Face& face = mesh.faces[fid];
        if (face.neighbour < 0) {
            continue; // Skip boundary faces
        }

        int owner = face.owner;
        int neighbour = face.neighbour;
        double F = mass_flux[fid];

        int upwind_cell, downwind_cell;
        if (F >= 0) {
            upwind_cell = owner;
            downwind_cell = neighbour;
        } else {
            upwind_cell = neighbour;
            downwind_cell = owner;
        }

        double phi_U = phi.values[upwind_cell];
        double phi_D = phi.values[downwind_cell];

        double delta_CD = phi_D - phi_U;

        if (std::abs(delta_CD) < 1e-15) {
            continue;
        }

        // r ratio: upwind gradient (projected onto cell-cell vector) vs cell-cell delta
        // d_UD = downwind center - upwind center
        Eigen::VectorXd d_UD = mesh.cells[downwind_cell].center.head(ndim)
                                - mesh.cells[upwind_cell].center.head(ndim);

        double grad_dot = 0.0;
        for (int d = 0; d < ndim; ++d) {
            grad_dot += grad_phi(upwind_cell, d) * d_UD[d];
        }

        double r = 2.0 * grad_dot / delta_CD - 1.0;

        // Apply limiter
        double psi = limiter_func(r);

        // High-order correction: |F| * (phi_MUSCL - phi_UW)
        double phi_MUSCL = phi_U + 0.5 * psi * delta_CD;
        double flux_correction = std::abs(F) * (phi_MUSCL - phi_U);

        // Owner/neighbour contributions (correction is face flux difference)
        if (F >= 0) {
            correction[owner] -= flux_correction;
            correction[neighbour] += flux_correction;
        } else {
            correction[neighbour] -= flux_correction;
            correction[owner] += flux_correction;
        }
    }

    return correction;
}

} // namespace twofluid
