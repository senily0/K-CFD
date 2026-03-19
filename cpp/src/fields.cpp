#include "twofluid/fields.hpp"
#include <algorithm>

namespace twofluid {

// ---------------------------------------------------------------------------
// ScalarField
// ---------------------------------------------------------------------------

ScalarField::ScalarField(const FVMesh& mesh, const std::string& name,
                         double default_val)
    : mesh_(mesh), name_(name) {
    values = Eigen::VectorXd::Constant(mesh.n_cells, default_val);
    for (const auto& [bname, fids] : mesh.boundary_patches) {
        boundary_values[bname] =
            Eigen::VectorXd::Constant(static_cast<int>(fids.size()), default_val);
    }
}

void ScalarField::set_uniform(double val) {
    values.setConstant(val);
    for (auto& [bname, bvals] : boundary_values) {
        bvals.setConstant(val);
    }
}

void ScalarField::store_old() {
    old_values = values;
}

void ScalarField::set_boundary(const std::string& patch, double val) {
    auto it = boundary_values.find(patch);
    if (it == boundary_values.end()) return;
    it->second.setConstant(val);
}

void ScalarField::set_boundary(const std::string& patch,
                               const Eigen::VectorXd& vals) {
    auto it = boundary_values.find(patch);
    if (it == boundary_values.end()) return;
    it->second = vals;
}

double ScalarField::get_face_value(int face_idx) const {
    const Face& face = mesh_.faces[face_idx];
    if (face.neighbour == -1) {
        for (const auto& [bname, fids] : mesh_.boundary_patches) {
            for (size_t i = 0; i < fids.size(); ++i) {
                if (fids[i] == face_idx) {
                    auto bit = boundary_values.find(bname);
                    if (bit != boundary_values.end()) {
                        return bit->second(static_cast<int>(i));
                    }
                }
            }
        }
    }
    return values(face.owner);
}

ScalarField ScalarField::copy() const {
    ScalarField sf(mesh_, name_);
    sf.values = values;
    for (const auto& [bname, bvals] : boundary_values) {
        sf.boundary_values[bname] = bvals;
    }
    sf.old_values = old_values;
    return sf;
}

double ScalarField::max() const { return values.maxCoeff(); }
double ScalarField::min() const { return values.minCoeff(); }
double ScalarField::mean() const { return values.mean(); }

// ---------------------------------------------------------------------------
// VectorField
// ---------------------------------------------------------------------------

VectorField::VectorField(const FVMesh& mesh, const std::string& name)
    : mesh_(mesh), name_(name) {
    int ndim = mesh.ndim;
    values = Eigen::MatrixXd::Zero(mesh.n_cells, ndim);
    for (const auto& [bname, fids] : mesh.boundary_patches) {
        boundary_values[bname] =
            Eigen::MatrixXd::Zero(static_cast<int>(fids.size()), ndim);
    }
}

void VectorField::set_uniform(const Eigen::VectorXd& val) {
    for (int i = 0; i < values.rows(); ++i) {
        values.row(i) = val.head(mesh_.ndim).transpose();
    }
    for (auto& [bname, bvals] : boundary_values) {
        for (int i = 0; i < bvals.rows(); ++i) {
            bvals.row(i) = val.head(mesh_.ndim).transpose();
        }
    }
}

void VectorField::store_old() {
    old_values = values;
}

void VectorField::set_boundary(const std::string& patch,
                               const Eigen::VectorXd& val) {
    auto it = boundary_values.find(patch);
    if (it == boundary_values.end()) return;
    for (int i = 0; i < it->second.rows(); ++i) {
        it->second.row(i) = val.head(mesh_.ndim).transpose();
    }
}

void VectorField::set_boundary(const std::string& patch,
                               const Eigen::MatrixXd& vals) {
    auto it = boundary_values.find(patch);
    if (it == boundary_values.end()) return;
    it->second = vals;
}

VectorField VectorField::copy() const {
    VectorField vf(mesh_, name_);
    vf.values = values;
    for (const auto& [bname, bvals] : boundary_values) {
        vf.boundary_values[bname] = bvals;
    }
    vf.old_values = old_values;
    return vf;
}

Eigen::VectorXd VectorField::magnitude() const {
    return values.rowwise().norm();
}

} // namespace twofluid
