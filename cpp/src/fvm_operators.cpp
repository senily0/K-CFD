#include "twofluid/fvm_operators.hpp"
#include <algorithm>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace twofluid {

// ---------------------------------------------------------------------------
// FVMSystem
// ---------------------------------------------------------------------------

FVMSystem::FVMSystem(int n) : n(n) {
    rhs = Eigen::VectorXd::Zero(n);
    diag = Eigen::VectorXd::Zero(n);
}

void FVMSystem::add_diagonal(int i, double val) {
    rows.push_back(i);
    cols.push_back(i);
    vals.push_back(val);
    diag(i) += val;
}

void FVMSystem::add_off_diagonal(int i, int j, double val) {
    rows.push_back(i);
    cols.push_back(j);
    vals.push_back(val);
}

void FVMSystem::add_source(int i, double val) {
    rhs(i) += val;
}

Eigen::SparseMatrix<double> FVMSystem::to_sparse() const {
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(rows.size());
    for (size_t k = 0; k < rows.size(); ++k) {
        triplets.emplace_back(rows[k], cols[k], vals[k]);
    }
    Eigen::SparseMatrix<double> mat(n, n);
    mat.setFromTriplets(triplets.begin(), triplets.end());
    return mat;
}

void FVMSystem::reset() {
    rows.clear();
    cols.clear();
    vals.clear();
    rhs.setZero();
    diag.setZero();
}

// ---------------------------------------------------------------------------
// Operators
// ---------------------------------------------------------------------------

void diffusion_operator(const FVMesh& mesh, const ScalarField& gamma,
                        FVMSystem& system) {
    for (int fid = 0; fid < mesh.n_faces; ++fid) {
        const Face& face = mesh.faces[fid];
        int owner = face.owner;

        if (face.neighbour >= 0) {
            int neighbour = face.neighbour;
            Eigen::Vector3d xO = mesh.cells[owner].center;
            Eigen::Vector3d xN = mesh.cells[neighbour].center;
            double d_PN = (xN - xO).norm();
            if (d_PN < 1e-30) continue;

            double gamma_O = gamma.values(owner);
            double gamma_N = gamma.values(neighbour);
            double gamma_f = 0.0;
            if (gamma_O + gamma_N > 1e-30) {
                gamma_f = 2.0 * gamma_O * gamma_N / (gamma_O + gamma_N);
            }

            double coeff = gamma_f * face.area / d_PN;
            system.add_diagonal(owner, coeff);
            system.add_diagonal(neighbour, coeff);
            system.add_off_diagonal(owner, neighbour, -coeff);
            system.add_off_diagonal(neighbour, owner, -coeff);
        }
        // Boundary faces handled in apply_boundary_conditions
    }
}

void convection_operator_upwind(const FVMesh& mesh,
                                const Eigen::VectorXd& mass_flux,
                                FVMSystem& system) {
    for (int fid = 0; fid < mesh.n_faces; ++fid) {
        const Face& face = mesh.faces[fid];
        int owner = face.owner;
        double F = mass_flux(fid);

        if (face.neighbour >= 0) {
            int neighbour = face.neighbour;
            system.add_diagonal(owner, std::max(F, 0.0));
            system.add_off_diagonal(owner, neighbour, std::min(F, 0.0));
            system.add_diagonal(neighbour, std::max(-F, 0.0));
            system.add_off_diagonal(neighbour, owner, std::min(-F, 0.0));
        } else {
            if (F >= 0) {
                system.add_diagonal(owner, F);
            }
        }
    }
}

void temporal_operator(const FVMesh& mesh, double rho, double dt,
                       const Eigen::VectorXd& phi_old, FVMSystem& system) {
    int n = mesh.n_cells;
    // Pre-compute coefficients in parallel, then add serially (add_diagonal/add_source are not thread-safe)
    Eigen::VectorXd coeff_vec(n);
#pragma omp parallel for schedule(static)
    for (int ci = 0; ci < n; ++ci) {
        coeff_vec(ci) = rho * mesh.cells[ci].volume / dt;
    }
    for (int ci = 0; ci < n; ++ci) {
        system.add_diagonal(ci, coeff_vec(ci));
        system.add_source(ci, coeff_vec(ci) * phi_old(ci));
    }
}

void source_term(const FVMesh& mesh, const Eigen::VectorXd& source_values,
                 FVMSystem& system) {
    int n = mesh.n_cells;
    Eigen::VectorXd contrib(n);
#pragma omp parallel for schedule(static)
    for (int ci = 0; ci < n; ++ci) {
        contrib(ci) = source_values(ci) * mesh.cells[ci].volume;
    }
    for (int ci = 0; ci < n; ++ci) {
        system.add_source(ci, contrib(ci));
    }
}

void linearized_source(const FVMesh& mesh, const Eigen::VectorXd& Sp,
                       const Eigen::VectorXd& Su, FVMSystem& system) {
    int n = mesh.n_cells;
    Eigen::VectorXd diag_contrib(n), src_contrib(n);
#pragma omp parallel for schedule(static)
    for (int ci = 0; ci < n; ++ci) {
        double vol = mesh.cells[ci].volume;
        diag_contrib(ci) = -Sp(ci) * vol;
        src_contrib(ci)  =  Su(ci) * vol;
    }
    for (int ci = 0; ci < n; ++ci) {
        system.add_diagonal(ci, diag_contrib(ci));
        system.add_source(ci, src_contrib(ci));
    }
}

void apply_boundary_conditions(
    const FVMesh& mesh, const ScalarField& phi, const ScalarField& gamma,
    const Eigen::VectorXd& mass_flux, FVMSystem& system,
    const std::unordered_map<std::string, BoundaryCondition>& bc_types) {

    for (const auto& [bname, fids] : mesh.boundary_patches) {
        BoundaryCondition bc;
        bc.type = "zero_gradient";
        auto bc_it = bc_types.find(bname);
        if (bc_it != bc_types.end()) {
            bc = bc_it->second;
        }

        for (size_t local_idx = 0; local_idx < fids.size(); ++local_idx) {
            int fid = fids[local_idx];
            const Face& face = mesh.faces[fid];
            int owner = face.owner;
            Eigen::Vector3d xO = mesh.cells[owner].center;
            double d_Pf = (face.center - xO).norm();
            if (d_Pf < 1e-30) continue;

            if (bc.type == "dirichlet") {
                auto bv_it = phi.boundary_values.find(bname);
                double phi_b = (bv_it != phi.boundary_values.end())
                    ? bv_it->second(static_cast<int>(local_idx))
                    : 0.0;
                double gamma_f = gamma.values(owner);
                double coeff = gamma_f * face.area / d_Pf;
                system.add_diagonal(owner, coeff);
                system.add_source(owner, coeff * phi_b);

                double F = mass_flux(fid);
                if (F < 0) {
                    // Standard inflow (F < 0 means flow enters domain through this face).
                    // convection_operator_upwind added nothing to diagonal for this face,
                    // so we add the inflow convection contribution here.
                    system.add_source(owner, -F * phi_b);
                } else {
                    // F >= 0: convection_operator_upwind already added F to diagonal
                    // treating this as outflow. For a Dirichlet BC we must override this
                    // and enforce phi_b regardless. Cancel the diagonal outflow term and
                    // add the Dirichlet-enforced convective flux to the source.
                    system.add_diagonal(owner, -F);
                    system.add_source(owner, F * phi_b);
                }
            } else if (bc.type == "neumann") {
                double flux = bc.value * face.area;
                system.add_source(owner, flux);
            } else {  // zero_gradient
                double F = mass_flux(fid);
                if (F < 0) {
                    system.add_diagonal(owner, -F);
                }
            }
        }
    }
}

void temporal_operator_bdf2(const FVMesh& mesh, double rho, double dt,
                             const Eigen::VectorXd& phi_old,
                             const Eigen::VectorXd& phi_old_old,
                             FVMSystem& system) {
    int n = mesh.n_cells;
    Eigen::VectorXd diag_contrib(n), src_contrib(n);
#pragma omp parallel for schedule(static)
    for (int ci = 0; ci < n; ++ci) {
        double coeff = rho * mesh.cells[ci].volume / (2.0 * dt);
        diag_contrib(ci) = 3.0 * coeff;
        src_contrib(ci)  = 4.0 * coeff * phi_old(ci) - coeff * phi_old_old(ci);
    }
    for (int ci = 0; ci < n; ++ci) {
        system.add_diagonal(ci, diag_contrib(ci));
        system.add_source(ci, src_contrib(ci));
    }
}

void diffusion_operator_corrected(const FVMesh& mesh, const ScalarField& gamma,
                                   const ScalarField& phi,
                                   FVMSystem& system,
                                   int n_corrections) {
    for (int fid = 0; fid < mesh.n_faces; ++fid) {
        const Face& face = mesh.faces[fid];
        int owner = face.owner;

        if (face.neighbour >= 0) {
            int neighbour = face.neighbour;
            Eigen::Vector3d xO = mesh.cells[owner].center;
            Eigen::Vector3d xN = mesh.cells[neighbour].center;
            Eigen::Vector3d delta = xN - xO;
            double d_PN = delta.norm();
            if (d_PN < 1e-30) continue;

            double gamma_O = gamma.values(owner);
            double gamma_N = gamma.values(neighbour);
            double gamma_f = 0.0;
            if (gamma_O + gamma_N > 1e-30) {
                gamma_f = 2.0 * gamma_O * gamma_N / (gamma_O + gamma_N);
            }

            // Orthogonal distance: projection of delta onto face normal
            double d_orth = std::abs(delta.dot(face.normal));
            if (d_orth < 1e-30) d_orth = d_PN;

            // Implicit orthogonal part
            double coeff = gamma_f * face.area / d_orth;
            system.add_diagonal(owner, coeff);
            system.add_diagonal(neighbour, coeff);
            system.add_off_diagonal(owner, neighbour, -coeff);
            system.add_off_diagonal(neighbour, owner, -coeff);

            // Explicit non-orthogonal correction on RHS
            if (n_corrections > 0) {
                double phi_diff = phi.values(neighbour) - phi.values(owner);
                // cross-diffusion term: gamma_f * A_f * (phi_N - phi_O) * (1/d_orth - 1/d_PN)
                double correction = gamma_f * face.area * phi_diff
                                    * (1.0 / d_orth - 1.0 / d_PN);
                system.add_source(owner, correction);
                system.add_source(neighbour, -correction);
            }
        }
        // Boundary faces handled in apply_boundary_conditions
    }
}

void under_relax(FVMSystem& system, const ScalarField& phi, double alpha) {
    if (alpha >= 1.0) return;
    int n = system.n;
    Eigen::VectorXd factors(n);
#pragma omp parallel for schedule(static)
    for (int ci = 0; ci < n; ++ci) {
        double ap = system.diag(ci);
        factors(ci) = ap * (1.0 - alpha) / alpha;
    }
    for (int ci = 0; ci < n; ++ci) {
        system.add_diagonal(ci, factors(ci));
        system.add_source(ci, factors(ci) * phi.values(ci));
    }
}

} // namespace twofluid
