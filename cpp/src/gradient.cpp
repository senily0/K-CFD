#include "twofluid/gradient.hpp"
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace twofluid {
namespace detail {

double interpolation_weight(const FVMesh& mesh, int face_id) {
    const Face& face = mesh.faces[face_id];
    const Eigen::Vector3d& xO = mesh.cells[face.owner].center;
    const Eigen::Vector3d& xN = mesh.cells[face.neighbour].center;
    const Eigen::Vector3d& xF = face.center;

    double dO = (xF - xO).norm();
    double dN = (xF - xN).norm();
    double total = dO + dN;
    if (total < 1e-30) return 0.5;
    return dN / total;  // larger weight when face is closer to owner
}

double get_boundary_face_value(const ScalarField& phi, int face_id) {
    const FVMesh& mesh = phi.mesh();
    auto cache_it = mesh.boundary_face_cache.find(face_id);
    if (cache_it != mesh.boundary_face_cache.end()) {
        auto& [bname, li] = cache_it->second;
        auto bit = phi.boundary_values.find(bname);
        if (bit != phi.boundary_values.end()) {
            return bit->second(li);
        }
    }
    const Face& face = mesh.faces[face_id];
    return phi.values(face.owner);
}

} // namespace detail

Eigen::MatrixXd green_gauss_gradient(const ScalarField& phi) {
    const FVMesh& mesh = phi.mesh();
    int ndim = mesh.ndim;
    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(mesh.n_cells, ndim);

    for (int fid = 0; fid < mesh.n_faces; ++fid) {
        const Face& face = mesh.faces[fid];
        int owner = face.owner;
        double phi_f;

        if (face.neighbour >= 0) {
            double gc = detail::interpolation_weight(mesh, fid);
            phi_f = gc * phi.values(owner) +
                    (1.0 - gc) * phi.values(face.neighbour);
        } else {
            phi_f = detail::get_boundary_face_value(phi, fid);
        }

        Eigen::VectorXd flux = phi_f * face.normal.head(ndim) * face.area;
        grad.row(owner) += flux.transpose();
        if (face.neighbour >= 0) {
            grad.row(face.neighbour) -= flux.transpose();
        }
    }

    int n = mesh.n_cells;
#pragma omp parallel for schedule(static)
    for (int ci = 0; ci < n; ++ci) {
        double vol = mesh.cells[ci].volume;
        if (vol > 1e-30) {
            grad.row(ci) /= vol;
        }
    }

    return grad;
}

Eigen::MatrixXd least_squares_gradient(const ScalarField& phi) {
    const FVMesh& mesh = phi.mesh();
    int ndim = mesh.ndim;
    int n_cells = mesh.n_cells;
    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(n_cells, ndim);

#pragma omp parallel for schedule(static)
    for (int ci = 0; ci < n_cells; ++ci) {
        const Cell& cell = mesh.cells[ci];
        Eigen::VectorXd xP = cell.center.head(ndim);

        std::vector<Eigen::VectorXd> dxs;
        std::vector<double> dphis;

        for (int fid : cell.faces) {
            const Face& face = mesh.faces[fid];
            if (face.neighbour >= 0) {
                int nb = (face.owner == ci) ? face.neighbour : face.owner;
                Eigen::VectorXd xN = mesh.cells[nb].center.head(ndim);
                dxs.push_back(xN - xP);
                dphis.push_back(phi.values(nb) - phi.values(ci));
            } else {
                Eigen::VectorXd xF = face.center.head(ndim);
                dxs.push_back(xF - xP);
                double phi_f = detail::get_boundary_face_value(phi, fid);
                dphis.push_back(phi_f - phi.values(ci));
            }
        }

        int n_nb = static_cast<int>(dxs.size());
        if (n_nb < ndim) continue;

        Eigen::MatrixXd A(n_nb, ndim);
        Eigen::VectorXd b(n_nb);
        for (int i = 0; i < n_nb; ++i) {
            A.row(i) = dxs[i].transpose();
            b(i) = dphis[i];
        }

        Eigen::MatrixXd ATA = A.transpose() * A;
        Eigen::VectorXd ATb = A.transpose() * b;

        // Solve using LLT (Cholesky); fall back to zero on failure
        Eigen::LLT<Eigen::MatrixXd> llt(ATA);
        if (llt.info() == Eigen::Success) {
            grad.row(ci) = llt.solve(ATb).transpose();
        }
    }

    return grad;
}

Eigen::VectorXd barth_jespersen_limiter(const FVMesh& mesh,
                                         const ScalarField& phi,
                                         const Eigen::MatrixXd& grad_phi) {
    int n = mesh.n_cells;
    int ndim = mesh.ndim;
    Eigen::VectorXd limiter = Eigen::VectorXd::Ones(n);

    // Compute phi_max and phi_min over each cell and its neighbours
    Eigen::VectorXd phi_max = phi.values;
    Eigen::VectorXd phi_min = phi.values;

    for (int fid = 0; fid < mesh.n_faces; ++fid) {
        const Face& face = mesh.faces[fid];
        int o = face.owner;
        if (face.neighbour >= 0) {
            int nb = face.neighbour;
            phi_max[o]  = std::max(phi_max[o],  phi.values[nb]);
            phi_min[o]  = std::min(phi_min[o],  phi.values[nb]);
            phi_max[nb] = std::max(phi_max[nb], phi.values[o]);
            phi_min[nb] = std::min(phi_min[nb], phi.values[o]);
        }
    }

    // For each cell, compute face reconstructed values and find minimum limiter
    for (int ci = 0; ci < n; ++ci) {
        const Cell& cell = mesh.cells[ci];
        Eigen::VectorXd xP = cell.center.head(ndim);
        double phi_i = phi.values[ci];
        double pmax  = phi_max[ci];
        double pmin  = phi_min[ci];
        double lim_i = 1.0;

        for (int fid : cell.faces) {
            const Face& face = mesh.faces[fid];
            Eigen::VectorXd xF = face.center.head(ndim);
            Eigen::VectorXd dr  = xF - xP;

            double phi_face = phi_i + grad_phi.row(ci).dot(dr);
            double delta    = phi_face - phi_i;

            double lim_f = 1.0;
            if (delta > 1e-30) {
                lim_f = std::min(1.0, (pmax - phi_i) / delta);
            } else if (delta < -1e-30) {
                lim_f = std::min(1.0, (pmin - phi_i) / delta);
            }
            lim_i = std::min(lim_i, lim_f);
        }

        limiter[ci] = std::max(0.0, lim_i);
    }

    return limiter;
}

Eigen::VectorXd venkatakrishnan_limiter(const FVMesh& mesh,
                                          const ScalarField& phi,
                                          const Eigen::MatrixXd& grad_phi,
                                          double epsilon) {
    int n = mesh.n_cells;
    int ndim = mesh.ndim;
    Eigen::VectorXd limiter = Eigen::VectorXd::Ones(n);

    // Compute phi_max and phi_min over each cell and its neighbours
    Eigen::VectorXd phi_max = phi.values;
    Eigen::VectorXd phi_min = phi.values;

    for (int fid = 0; fid < mesh.n_faces; ++fid) {
        const Face& face = mesh.faces[fid];
        int o = face.owner;
        if (face.neighbour >= 0) {
            int nb = face.neighbour;
            phi_max[o]  = std::max(phi_max[o],  phi.values[nb]);
            phi_min[o]  = std::min(phi_min[o],  phi.values[nb]);
            phi_max[nb] = std::max(phi_max[nb], phi.values[o]);
            phi_min[nb] = std::min(phi_min[nb], phi.values[o]);
        }
    }

    double eps2 = epsilon * epsilon;

    for (int ci = 0; ci < n; ++ci) {
        const Cell& cell = mesh.cells[ci];
        Eigen::VectorXd xP = cell.center.head(ndim);
        double phi_i   = phi.values[ci];
        double delta_max = phi_max[ci] - phi_i;
        double delta_min = phi_min[ci] - phi_i;
        double lim_i   = 1.0;

        for (int fid : cell.faces) {
            const Face& face = mesh.faces[fid];
            Eigen::VectorXd xF = face.center.head(ndim);
            double delta_face = grad_phi.row(ci).dot(xF - xP);

            double delta_ref;
            if (delta_face > 0.0) {
                delta_ref = delta_max;
            } else if (delta_face < 0.0) {
                delta_ref = delta_min;
            } else {
                continue;
            }

            double df2  = delta_face * delta_face;
            double dr2  = delta_ref  * delta_ref;
            double num  = dr2 + eps2 + 2.0 * delta_face * delta_ref;
            double den  = dr2 + 2.0 * df2 + delta_face * delta_ref + eps2;
            double lim_f = (std::abs(den) > 1e-30) ? num / den : 1.0;

            lim_i = std::min(lim_i, lim_f);
        }

        limiter[ci] = std::max(0.0, std::min(1.0, lim_i));
    }

    return limiter;
}

Eigen::MatrixXd limit_gradient(const Eigen::MatrixXd& grad_phi,
                                 const Eigen::VectorXd& limiter) {
    // Broadcast limiter (n,) over each column of grad_phi (n, ndim)
    return limiter.asDiagonal() * grad_phi;
}

} // namespace twofluid
