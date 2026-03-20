#pragma once

#include <string>
#include <unordered_map>
#include <optional>
#include <Eigen/Dense>
#include "twofluid/mesh.hpp"

namespace twofluid {

class ScalarField {
public:
    /// Construct a scalar field on the given mesh.
    ScalarField(const FVMesh& mesh, const std::string& name = "scalar",
                double default_val = 0.0);

    Eigen::VectorXd values;
    std::optional<Eigen::VectorXd> old_values;
    std::optional<Eigen::VectorXd> old_old_values;  // phi^{n-2} for BDF2
    std::unordered_map<std::string, Eigen::VectorXd> boundary_values;

    /// Set all cell and boundary values to a uniform value.
    void set_uniform(double val);

    /// Store current values as old (for unsteady calculations).
    void store_old();

    /// Set boundary patch to a scalar value.
    void set_boundary(const std::string& patch, double val);

    /// Set boundary patch to an array of values.
    void set_boundary(const std::string& patch, const Eigen::VectorXd& vals);

    /// Get the face value (boundary value if boundary face, owner cell value otherwise).
    double get_face_value(int face_idx) const;

    /// Create a deep copy.
    ScalarField copy() const;

    double max() const;
    double min() const;
    double mean() const;

    const std::string& name() const { return name_; }
    const FVMesh& mesh() const { return mesh_; }

private:
    const FVMesh& mesh_;
    std::string name_;
};

class VectorField {
public:
    /// Construct a vector field on the given mesh.
    VectorField(const FVMesh& mesh, const std::string& name = "vector");

    Eigen::MatrixXd values;     // (n_cells, ndim)
    std::optional<Eigen::MatrixXd> old_values;
    std::optional<Eigen::MatrixXd> old_old_values;  // U^{n-2} for BDF2
    std::unordered_map<std::string, Eigen::MatrixXd> boundary_values;

    /// Set all cell and boundary values to a uniform vector.
    void set_uniform(const Eigen::VectorXd& val);

    /// Store current values as old.
    void store_old();

    /// Set boundary patch to a uniform vector value.
    void set_boundary(const std::string& patch, const Eigen::VectorXd& val);

    /// Set boundary patch to a matrix of values (one row per face).
    void set_boundary(const std::string& patch, const Eigen::MatrixXd& vals);

    /// Create a deep copy.
    VectorField copy() const;

    /// Compute magnitude for each cell.
    Eigen::VectorXd magnitude() const;

    const std::string& name() const { return name_; }
    const FVMesh& mesh() const { return mesh_; }

private:
    const FVMesh& mesh_;
    std::string name_;
};

} // namespace twofluid
