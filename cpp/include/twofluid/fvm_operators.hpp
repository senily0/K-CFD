#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "twofluid/mesh.hpp"
#include "twofluid/fields.hpp"

namespace twofluid {

/// FVM discretisation result stored as a linear system Ax = b.
class FVMSystem {
public:
    explicit FVMSystem(int n);

    int n;

    // COO triplet data
    std::vector<int> rows;
    std::vector<int> cols;
    std::vector<double> vals;

    Eigen::VectorXd rhs;
    Eigen::VectorXd diag;  // diagonal coefficients (for SIMPLE a_P access)

    void add_diagonal(int i, double val);
    void add_off_diagonal(int i, int j, double val);
    void add_source(int i, double val);

    /// Build a sparse CSR matrix from accumulated COO data.
    Eigen::SparseMatrix<double> to_sparse() const;

    /// Reset the system to zero.
    void reset();
};

// ----- Discretisation operators -----

/// Diffusion: integral of div(gamma * grad(phi)) dV
void diffusion_operator(const FVMesh& mesh, const ScalarField& gamma,
                        FVMSystem& system);

/// Convection (1st-order Upwind): integral of div(rho*u*phi) dV
void convection_operator_upwind(const FVMesh& mesh,
                                const Eigen::VectorXd& mass_flux,
                                FVMSystem& system);

/// Temporal (Backward Euler): (rho*V/dt)*phi = (rho*V/dt)*phi_old + ...
void temporal_operator(const FVMesh& mesh, double rho, double dt,
                       const Eigen::VectorXd& phi_old, FVMSystem& system);

/// Source term: S_P * V_P added to RHS.
void source_term(const FVMesh& mesh, const Eigen::VectorXd& source_values,
                 FVMSystem& system);

/// Linearised source: S = Su + Sp*phi. Sp < 0 for stability.
void linearized_source(const FVMesh& mesh, const Eigen::VectorXd& Sp,
                       const Eigen::VectorXd& Su, FVMSystem& system);

struct BoundaryCondition {
    std::string type;  // "dirichlet", "neumann", "zero_gradient"
    double value = 0.0;  // for neumann: gradient value
};

/// Apply boundary conditions to the linear system.
void apply_boundary_conditions(
    const FVMesh& mesh, const ScalarField& phi, const ScalarField& gamma,
    const Eigen::VectorXd& mass_flux, FVMSystem& system,
    const std::unordered_map<std::string, BoundaryCondition>& bc_types);

/// Under-relaxation.
void under_relax(FVMSystem& system, const ScalarField& phi, double alpha);

} // namespace twofluid
