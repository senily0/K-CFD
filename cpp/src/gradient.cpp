#include "twofluid/gradient.hpp"
#include <cmath>

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

    for (int ci = 0; ci < mesh.n_cells; ++ci) {
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
    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(mesh.n_cells, ndim);

    for (int ci = 0; ci < mesh.n_cells; ++ci) {
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

} // namespace twofluid
