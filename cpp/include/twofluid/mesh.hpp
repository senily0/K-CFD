#pragma once

#include <array>
#include <string>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <Eigen/Dense>

namespace twofluid {

// Use DontAlign to avoid alignment issues with std::vector<Face/Cell>
using Vec3 = Eigen::Matrix<double, 3, 1, Eigen::DontAlign>;

struct Face {
    int owner = -1;
    int neighbour = -1;  // -1 for boundary faces
    double area = 0.0;
    Vec3 normal = Vec3::Zero();
    Vec3 center = Vec3::Zero();
    std::vector<int> nodes;
    std::string boundary_tag;  // empty for internal faces
};

struct Cell {
    double volume = 0.0;
    Vec3 center = Vec3::Zero();
    std::vector<int> nodes;
    std::vector<int> faces;
};

class FVMesh {
public:
    int ndim = 2;
    int n_cells = 0;
    int n_faces = 0;
    int n_internal_faces = 0;
    int n_boundary_faces = 0;

    Eigen::MatrixXd nodes;  // (n_nodes, 3)
    std::vector<Cell> cells;
    std::vector<Face> faces;

    std::unordered_map<std::string, std::vector<int>> boundary_patches;
    std::unordered_map<std::string, std::vector<int>> cell_zones;

    /// Build face-to-boundary cache. Call after mesh construction.
    void build_boundary_face_cache();

    /// Lookup: face_id -> (patch_name, local_index). Empty string if internal.
    std::unordered_map<int, std::pair<std::string, int>> boundary_face_cache;

    /// Construct an empty mesh with the given spatial dimension.
    explicit FVMesh(int ndim = 2);

    /// Return a human-readable summary string.
    std::string summary() const;

    struct MeshQuality {
        double max_non_orthogonality;  // degrees
        double avg_non_orthogonality;
        double max_skewness;
        double avg_skewness;
        double max_aspect_ratio;
        int n_bad_cells;  // non-orth > 70 degrees
    };

    MeshQuality compute_quality() const;
};

} // namespace twofluid
