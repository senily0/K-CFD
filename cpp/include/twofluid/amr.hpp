#pragma once

#include <vector>
#include <Eigen/Dense>
#include "twofluid/mesh.hpp"
#include "twofluid/fields.hpp"

namespace twofluid {

struct AMRCell {
    int cell_id;
    int level = 0;
    int parent = -1;
    std::vector<int> children;
    bool is_leaf() const { return children.empty(); }
};

class AMRMesh {
public:
    AMRMesh(const FVMesh& base_mesh, int max_level = 3);

    void refine_cells(const std::vector<int>& cell_ids);
    std::vector<int> get_active_cells() const;
    FVMesh get_active_mesh() const;
    ScalarField transfer_field_to_children(const ScalarField& field,
                                            const FVMesh& active_mesh) const;
    int n_refinements() const { return n_refinements_; }

private:
    FVMesh base_mesh_;
    int max_level_;
    int n_refinements_ = 0;
    std::vector<AMRCell> amr_cells_;
    Eigen::MatrixXd nodes_;
    std::vector<std::vector<int>> cell_node_list_;
};

class GradientJumpEstimator {
public:
    static Eigen::VectorXd estimate(const FVMesh& mesh, const ScalarField& phi);
};

class AMRSolverLoop {
public:
    AMRSolverLoop(AMRMesh& amr_mesh, double refine_fraction = 0.3);
    std::vector<int> mark_cells(const FVMesh& mesh, const ScalarField& phi);

private:
    AMRMesh& amr_mesh_;
    double refine_fraction_;
};

} // namespace twofluid
