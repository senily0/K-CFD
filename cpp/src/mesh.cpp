#include "twofluid/mesh.hpp"
#include <cmath>
#include <iostream>

namespace twofluid {

FVMesh::FVMesh(int ndim) : ndim(ndim) {
    nodes.resize(0, 3);
}

void FVMesh::build_boundary_face_cache() {
    boundary_face_cache.clear();
    for (auto& [bname, fids] : boundary_patches) {
        for (int li = 0; li < (int)fids.size(); ++li) {
            boundary_face_cache[fids[li]] = {bname, li};
        }
    }
}

std::string FVMesh::summary() const {
    std::ostringstream ss;
    ss << "FVMesh(" << ndim << "D): " << n_cells << " cells, "
       << n_faces << " faces (" << n_internal_faces << " internal, "
       << n_boundary_faces << " boundary)\n";
    ss << "  Nodes: " << nodes.rows() << "\n";
    for (const auto& [name, fids] : boundary_patches) {
        ss << "  Boundary '" << name << "': " << fids.size() << " faces\n";
    }
    for (const auto& [name, cids] : cell_zones) {
        ss << "  Zone '" << name << "': " << cids.size() << " cells\n";
    }
    return ss.str();
}

FVMesh::MeshQuality FVMesh::compute_quality() const {
    MeshQuality q{};
    q.max_non_orthogonality = 0.0;
    q.avg_non_orthogonality = 0.0;
    q.max_skewness          = 0.0;
    q.avg_skewness          = 0.0;
    q.max_aspect_ratio      = 0.0;
    q.n_bad_cells           = 0;

    int n_internal = 0;

    // Non-orthogonality and skewness (internal faces only)
    for (int fid = 0; fid < n_faces; ++fid) {
        const Face& face = faces[fid];
        if (face.neighbour < 0) continue;

        const Vec3& xO = cells[face.owner].center;
        const Vec3& xN = cells[face.neighbour].center;
        const Vec3& xF = face.center;

        Vec3 d_vec = xN - xO;
        double d_len = d_vec.norm();
        if (d_len < 1e-30) continue;

        // Non-orthogonality: angle between face normal and owner->neighbour vector
        Vec3 n_unit = face.normal.normalized();
        Vec3 d_unit = d_vec / d_len;
        double cos_theta = std::abs(n_unit.dot(d_unit));
        cos_theta = std::min(1.0, cos_theta);  // clamp for acos safety
        double angle_deg = std::acos(cos_theta) * 180.0 / M_PI;

        q.max_non_orthogonality = std::max(q.max_non_orthogonality, angle_deg);
        q.avg_non_orthogonality += angle_deg;

        // Skewness: distance from face centre to the line O->N intersection,
        // normalised by |O->N|.
        // Intersection point: xO + t*(xN-xO) where t = dot(xF-xO, d_unit)/d_len
        double t = (xF - xO).dot(d_unit) / d_len;
        t = std::max(0.0, std::min(1.0, t));
        Vec3 x_intersect = xO + t * d_vec;
        double skew = (xF - x_intersect).norm() / std::max(d_len, 1e-30);

        q.max_skewness  = std::max(q.max_skewness, skew);
        q.avg_skewness += skew;

        ++n_internal;
    }

    if (n_internal > 0) {
        q.avg_non_orthogonality /= n_internal;
        q.avg_skewness          /= n_internal;
    }

    // Count bad cells (non-orthogonality > 70 degrees) and aspect ratio
    // We need per-cell max non-orthogonality, so iterate faces again.
    std::vector<double> cell_max_nonorth(n_cells, 0.0);
    for (int fid = 0; fid < n_faces; ++fid) {
        const Face& face = faces[fid];
        if (face.neighbour < 0) continue;

        const Vec3& xO = cells[face.owner].center;
        const Vec3& xN = cells[face.neighbour].center;
        Vec3 d_vec = xN - xO;
        double d_len = d_vec.norm();
        if (d_len < 1e-30) continue;

        Vec3 n_unit = face.normal.normalized();
        Vec3 d_unit = d_vec / d_len;
        double cos_theta = std::abs(n_unit.dot(d_unit));
        cos_theta = std::min(1.0, cos_theta);
        double angle_deg = std::acos(cos_theta) * 180.0 / M_PI;

        cell_max_nonorth[face.owner]     = std::max(cell_max_nonorth[face.owner],     angle_deg);
        cell_max_nonorth[face.neighbour] = std::max(cell_max_nonorth[face.neighbour], angle_deg);
    }

    for (int ci = 0; ci < n_cells; ++ci) {
        if (cell_max_nonorth[ci] > 70.0) ++q.n_bad_cells;

        // Aspect ratio from cell node bounding box
        if (cells[ci].nodes.empty()) continue;
        Vec3 bmin = Vec3::Constant(std::numeric_limits<double>::max());
        Vec3 bmax = Vec3::Constant(std::numeric_limits<double>::lowest());
        for (int nid : cells[ci].nodes) {
            Vec3 pt = nodes.row(nid).transpose();
            bmin = bmin.cwiseMin(pt);
            bmax = bmax.cwiseMax(pt);
        }
        Vec3 ext = (bmax - bmin).cwiseAbs();
        double ext_max = ext.maxCoeff();
        double ext_min = ext.head(ndim).minCoeff();
        if (ext_min > 1e-30) {
            double ar = ext_max / ext_min;
            q.max_aspect_ratio = std::max(q.max_aspect_ratio, ar);
        }
    }

    // Warnings
    if (q.max_non_orthogonality > 70.0) {
        std::cout << "[MeshQuality] WARNING: max non-orthogonality = "
                  << q.max_non_orthogonality << " deg (> 70 deg). "
                  << q.n_bad_cells << " bad cells.\n";
    }
    if (q.max_skewness > 0.85) {
        std::cout << "[MeshQuality] WARNING: max skewness = "
                  << q.max_skewness << " (> 0.85).\n";
    }

    return q;
}

} // namespace twofluid
