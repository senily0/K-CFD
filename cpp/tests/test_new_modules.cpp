/**
 * Tests for all new C++ modules:
 *   - Preconditioner (Jacobi, ILU0)
 *   - AdaptiveTimeControl
 *   - Phase Change Models (Lee, Rohsenow, Zuber, Nusselt, Manager)
 *   - SolidConductionSolver
 *   - P1RadiationModel
 *   - AMR (AMRMesh, GradientJumpEstimator, AMRSolverLoop)
 *   - 3D Mesh Generator
 */

#include <cmath>
#include <iostream>
#include <cassert>

#include "twofluid/mesh.hpp"
#include "twofluid/mesh_generator.hpp"
#include "twofluid/mesh_generator_3d.hpp"
#include "twofluid/fields.hpp"
#include "twofluid/fvm_operators.hpp"
#include "twofluid/linear_solver.hpp"
#include "twofluid/preconditioner.hpp"
#include "twofluid/time_control.hpp"
#include "twofluid/phase_change.hpp"
#include "twofluid/solid_conduction.hpp"
#include "twofluid/radiation.hpp"
#include "twofluid/amr.hpp"

using namespace twofluid;

#define CHECK(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << "\n"; return false; } \
} while(0)

// ========== Preconditioner ==========
bool test_preconditioner() {
    std::cout << "\n=== test_preconditioner ===" << std::endl;

    auto mesh = generate_channel_mesh(1.0, 1.0, 5, 5);
    int n = mesh.n_cells;
    FVMSystem system(n);

    ScalarField gamma(mesh, "gamma");
    gamma.set_uniform(1.0);
    diffusion_operator(mesh, gamma, system);

    // Add diagonal dominance
    for (int i = 0; i < n; ++i) system.add_diagonal(i, 10.0);
    for (int i = 0; i < n; ++i) system.add_source(i, 1.0);

    auto A = system.to_sparse();
    A.makeCompressed();

    // Test "none"
    auto [fn_none, info_none] = create_preconditioner(A, "none");
    CHECK(fn_none == nullptr, "none preconditioner should be null");
    CHECK(info_none.method == "none", "method should be none");

    // Test "jacobi"
    auto [fn_jac, info_jac] = create_preconditioner(A, "jacobi");
    CHECK(fn_jac != nullptr, "jacobi should not be null");
    CHECK(info_jac.method == "jacobi", "method should be jacobi");
    Eigen::VectorXd x = Eigen::VectorXd::Ones(n);
    Eigen::VectorXd y = fn_jac(x);
    CHECK(y.size() == n, "jacobi output size");

    // Test "ilu0"
    auto [fn_ilu, info_ilu] = create_preconditioner(A, "ilu0");
    CHECK(fn_ilu != nullptr, "ilu0 should not be null");
    CHECK(info_ilu.method == "ilu0", "method should be ilu0");
    Eigen::VectorXd z = fn_ilu(x);
    CHECK(z.size() == n, "ilu0 output size");

    std::cout << "  jacobi setup: " << info_jac.setup_time << " s\n";
    std::cout << "  ilu0 setup: " << info_ilu.setup_time << " s\n";
    std::cout << "PASSED" << std::endl;
    return true;
}

// ========== AdaptiveTimeControl ==========
bool test_time_control() {
    std::cout << "\n=== test_time_control ===" << std::endl;

    auto mesh = generate_channel_mesh(1.0, 1.0, 10, 10);

    AdaptiveTimeControl tc(0.01, 1e-8, 1.0, 0.5, 1.0, 0.5);

    Eigen::VectorXd u_mag = Eigen::VectorXd::Constant(mesh.n_cells, 1.0);

    // CFL test
    auto [cfl_max, cfl_arr] = tc.compute_cfl(mesh, u_mag, 0.01);
    CHECK(cfl_max > 0.0, "CFL should be positive");
    CHECK(cfl_arr.size() == mesh.n_cells, "CFL array size");
    std::cout << "  CFL max = " << cfl_max << "\n";

    // Fourier test
    auto [fo_max, fo_arr] = tc.compute_fourier(mesh, 1e-5, 0.01);
    CHECK(fo_max >= 0.0, "Fourier should be non-negative");
    std::cout << "  Fourier max = " << fo_max << "\n";

    // compute_dt
    auto info = tc.compute_dt(mesh, u_mag, 1e-5, true);
    CHECK(info.dt > 0.0, "dt should be positive");
    CHECK(!info.dt_limited_by.empty(), "should have limiting factor");
    std::cout << "  New dt = " << info.dt << " (limited by: " << info.dt_limited_by << ")\n";

    // Divergence shrink
    auto info2 = tc.compute_dt(mesh, u_mag, -1.0, false);
    CHECK(info2.dt_limited_by == "divergence", "should be limited by divergence");
    CHECK(info2.dt < info.dt, "dt should shrink on divergence");
    std::cout << "  After divergence: dt = " << info2.dt << "\n";

    CHECK(tc.dt_history.size() == 2, "should have 2 history entries");

    std::cout << "PASSED" << std::endl;
    return true;
}

// ========== Phase Change Models ==========
bool test_phase_change() {
    std::cout << "\n=== test_phase_change ===" << std::endl;

    // Utility functions
    double T_sat = saturation_temperature(101325.0);  // 1 atm
    CHECK(T_sat > 370.0 && T_sat < 376.0,
          "T_sat at 1 atm should be ~373K");
    std::cout << "  T_sat(1 atm) = " << T_sat << " K\n";

    double h_fg = water_latent_heat(101325.0);
    CHECK(h_fg > 2.0e6 && h_fg < 2.5e6, "h_fg should be ~2.26e6");
    std::cout << "  h_fg(1 atm) = " << h_fg << " J/kg\n";

    auto wp = water_properties(101325.0);
    CHECK(wp.rho_l > 900.0, "rho_l should be ~1000");
    CHECK(wp.rho_g > 0.1, "rho_g should be > 0");

    // Lee model
    auto mesh = generate_channel_mesh(1.0, 1.0, 5, 5);
    LeePhaseChangeModel lee(mesh, 373.15, 0.1, 0.1, 2.26e6, 1000.0, 1.0);

    ScalarField T(mesh, "T");
    ScalarField alpha_l(mesh, "alpha_l");
    T.set_uniform(380.0);  // above T_sat -> evaporation
    alpha_l.set_uniform(0.9);

    auto dot_m = lee.compute_mass_transfer(T, alpha_l);
    CHECK(dot_m.minCoeff() >= 0.0, "evaporation should give positive dot_m");
    std::cout << "  Lee evaporation dot_m = " << dot_m(0) << " kg/(m3.s)\n";

    T.set_uniform(365.0);  // below T_sat -> condensation
    auto dot_m2 = lee.compute_mass_transfer(T, alpha_l);
    CHECK(dot_m2.maxCoeff() <= 0.0, "condensation should give negative dot_m");
    std::cout << "  Lee condensation dot_m = " << dot_m2(0) << " kg/(m3.s)\n";

    auto src = lee.get_source_terms(T, alpha_l);
    CHECK(src.alpha_l.size() == mesh.n_cells, "source size");

    // Rohsenow
    RohsenowBoilingModel rohsenow(373.15, 2.26e6, 958.0, 0.6,
                                   2.82e-4, 4216.0, 0.059, 1.75);
    double q_wall = rohsenow.compute_wall_heat_flux(383.15);  // 10K superheat
    CHECK(q_wall > 0.0, "boiling flux should be positive");
    std::cout << "  Rohsenow q(10K) = " << q_wall << " W/m2\n";

    // Zuber CHF
    ZuberCHFModel zuber(2.26e6, 958.0, 0.6, 0.059);
    double chf = zuber.compute_chf();
    CHECK(chf > 1e5, "CHF should be > 100 kW/m2");
    std::cout << "  Zuber CHF = " << chf << " W/m2\n";
    auto margin = zuber.check_margin(1e5);
    CHECK(margin.safe, "100 kW/m2 should be safe");

    // Nusselt condensation
    NusseltCondensationModel nusselt(373.15, 2.26e6, 958.0, 0.6, 2.82e-4, 0.68);
    double h_cond = nusselt.compute_heat_transfer_coeff(0.1, 10.0);
    CHECK(h_cond > 0.0, "condensation h should be positive");
    std::cout << "  Nusselt h_cond = " << h_cond << " W/(m2.K)\n";

    double dm_cond = nusselt.compute_condensation_rate(0.1, 363.15, 0.01, 0.001);
    CHECK(dm_cond < 0.0, "condensation rate should be negative");

    // PhaseChangeManager
    PhaseChangeManager mgr(mesh, 373.15, 2.26e6, 1000.0, 1.0);
    T.set_uniform(380.0);
    alpha_l.set_uniform(0.9);
    auto total_src = mgr.get_source_terms(T, alpha_l);
    CHECK(total_src.alpha_l.size() == mesh.n_cells, "manager source size");

    std::cout << "PASSED" << std::endl;
    return true;
}

// ========== Solid Conduction ==========
bool test_solid_conduction() {
    std::cout << "\n=== test_solid_conduction ===" << std::endl;

    auto mesh = generate_channel_mesh(1.0, 1.0, 10, 10);
    SolidConductionSolver solver(mesh);
    solver.set_material(8960.0, 385.0, 401.0);  // copper

    // Set BCs: left=400K, right=300K
    solver.bc_T["inlet"] = {"dirichlet", 0.0};
    solver.bc_T["outlet"] = {"dirichlet", 0.0};
    solver.bc_T["wall_bottom"] = {"zero_gradient", 0.0};
    solver.bc_T["wall_top"] = {"zero_gradient", 0.0};
    solver.T.set_boundary("inlet", 400.0);
    solver.T.set_boundary("outlet", 300.0);

    auto result = solver.solve_steady(500, 1e-6);
    std::cout << "  Converged: " << result.converged
              << "  Iterations: " << result.iterations
              << "  T_max: " << result.T_max
              << "  T_min: " << result.T_min << "\n";

    CHECK(result.converged, "should converge");
    CHECK(result.T_max <= 400.0 + 1.0, "T_max should be <= 400");
    CHECK(result.T_min >= 300.0 - 1.0, "T_min should be >= 300");

    std::cout << "PASSED" << std::endl;
    return true;
}

// ========== Radiation ==========
bool test_radiation() {
    std::cout << "\n=== test_radiation ===" << std::endl;

    auto mesh = generate_channel_mesh(1.0, 1.0, 10, 10);
    P1RadiationModel rad(mesh, 1.0);

    // Set BCs
    rad.set_bc("inlet", "marshak", 1000.0);
    rad.set_bc("outlet", "marshak", 500.0);
    rad.set_bc("wall_bottom", "zero_gradient");
    rad.set_bc("wall_top", "zero_gradient");

    // Temperature field
    ScalarField T(mesh, "T");
    T.set_uniform(800.0);

    auto result = rad.solve(T, 200, 1e-4);
    std::cout << "  Converged: " << result.converged
              << "  Iterations: " << result.iterations << "\n";
    if (!result.residuals.empty()) {
        std::cout << "  Final residual: " << result.residuals.back() << "\n";
    }

    CHECK(result.converged, "radiation should converge");

    // Radiative source
    auto q_r = rad.compute_radiative_source(T);
    CHECK(q_r.size() == mesh.n_cells, "source size");
    std::cout << "  q_r range: [" << q_r.minCoeff() << ", " << q_r.maxCoeff() << "]\n";

    std::cout << "PASSED" << std::endl;
    return true;
}

// ========== AMR ==========
bool test_amr() {
    std::cout << "\n=== test_amr ===" << std::endl;

    auto mesh = generate_cavity_mesh(1.0, 4);
    std::cout << "  Base mesh: " << mesh.n_cells << " cells\n";

    AMRMesh amr(mesh, 2);

    // Refine center cells
    auto active = amr.get_active_cells();
    CHECK(static_cast<int>(active.size()) == 16, "initial 4x4 = 16 cells");

    // Count how many of {5,6,9,10} are quad cells
    std::vector<int> refine_ids = {5, 6, 9, 10};
    int n_quad = 0;
    for (int ci : refine_ids) {
        if (mesh.cells[ci].nodes.size() == 4) n_quad++;
    }
    std::cout << "  Quad cells in refine set: " << n_quad << "/" << refine_ids.size() << "\n";

    amr.refine_cells(refine_ids);
    CHECK(amr.n_refinements() == 1, "1 refinement");

    auto active2 = amr.get_active_cells();
    // original - refined + 4*refined_quads
    int expected = 16 - n_quad + 4 * n_quad;
    std::cout << "  After refinement: " << active2.size() << " active cells"
              << " (expected " << expected << ")\n";
    CHECK(static_cast<int>(active2.size()) == expected, "cell count after refine");

    // Get active mesh
    auto refined_mesh = amr.get_active_mesh();
    std::cout << "  Refined mesh: " << refined_mesh.n_cells << " cells, "
              << refined_mesh.n_faces << " faces\n";
    CHECK(refined_mesh.n_cells == expected, "refined mesh cell count");
    CHECK(refined_mesh.n_faces > 0, "should have faces");

    // GradientJumpEstimator
    ScalarField phi(refined_mesh, "phi");
    for (int i = 0; i < refined_mesh.n_cells; ++i) {
        phi.values(i) = refined_mesh.cells[i].center(0);  // linear in x
    }
    auto error = GradientJumpEstimator::estimate(refined_mesh, phi);
    CHECK(error.size() == refined_mesh.n_cells, "error size");

    // AMRSolverLoop
    AMRSolverLoop loop(amr, 0.3);
    auto to_refine = loop.mark_cells(refined_mesh, phi);
    std::cout << "  Cells marked for refinement: " << to_refine.size() << "\n";

    std::cout << "PASSED" << std::endl;
    return true;
}

// ========== 3D Mesh Generator ==========
bool test_3d_mesh() {
    std::cout << "\n=== test_3d_mesh ===" << std::endl;

    // 3D Channel
    auto mesh = generate_3d_channel_mesh(1.0, 0.5, 0.5, 4, 3, 3);
    std::cout << "  3D Channel: " << mesh.n_cells << " cells, "
              << mesh.n_faces << " faces ("
              << mesh.n_internal_faces << " internal, "
              << mesh.n_boundary_faces << " boundary)\n";

    CHECK(mesh.ndim == 3, "should be 3D");
    CHECK(mesh.n_cells == 4 * 3 * 3, "4x3x3 = 36 cells");

    // Expected faces: internal x: 3*3*3=27, y: 4*2*3=24, z: 4*3*2=24 = 75 internal
    // Boundary: x: 2*3*3=18, y: 2*4*3=24, z: 2*4*3=24 = 66 boundary
    CHECK(mesh.n_internal_faces > 0, "should have internal faces");
    CHECK(mesh.n_boundary_faces > 0, "should have boundary faces");

    // Check boundary patches
    CHECK(mesh.boundary_patches.count("inlet") > 0, "should have inlet");
    CHECK(mesh.boundary_patches.count("outlet") > 0, "should have outlet");
    std::cout << "  Boundary patches: ";
    for (auto& [name, fids] : mesh.boundary_patches) {
        std::cout << name << "(" << fids.size() << ") ";
    }
    std::cout << "\n";

    // Cell volumes should be positive
    for (int i = 0; i < mesh.n_cells; ++i) {
        CHECK(mesh.cells[i].volume > 0.0, "cell volume should be positive");
    }

    // Total volume check
    double total_vol = 0.0;
    for (auto& c : mesh.cells) total_vol += c.volume;
    double expected_vol = 1.0 * 0.5 * 0.5;
    CHECK(std::abs(total_vol - expected_vol) < 1e-10,
          "total volume should match domain");
    std::cout << "  Total volume: " << total_vol
              << " (expected " << expected_vol << ")\n";

    // 3D Duct
    auto duct = generate_3d_duct_mesh(2.0, 0.1, 0.1, 5, 3, 3);
    CHECK(duct.n_cells == 45, "5x3x3 = 45 cells");
    std::cout << "  3D Duct: " << duct.n_cells << " cells\n";

    // 3D Cavity
    auto cavity = generate_3d_cavity_mesh(1.0, 1.0, 1.0, 4, 4, 4);
    CHECK(cavity.n_cells == 64, "4x4x4 = 64 cells");
    CHECK(cavity.boundary_patches.count("lid") > 0, "should have lid");
    std::cout << "  3D Cavity: " << cavity.n_cells << " cells\n";

    std::cout << "PASSED" << std::endl;
    return true;
}

int main() {
    std::cout << "Running new module tests...\n";

    int pass = 0, fail = 0;
    auto run = [&](bool (*fn)(), const char* name) {
        try {
            if (fn()) { pass++; }
            else { fail++; std::cerr << "FAILED: " << name << "\n"; }
        } catch (const std::exception& e) {
            fail++;
            std::cerr << "EXCEPTION in " << name << ": " << e.what() << "\n";
        }
    };

    run(test_preconditioner, "preconditioner");
    run(test_time_control, "time_control");
    run(test_phase_change, "phase_change");
    run(test_solid_conduction, "solid_conduction");
    run(test_radiation, "radiation");
    run(test_amr, "amr");
    run(test_3d_mesh, "3d_mesh");

    std::cout << "\n========================================\n";
    std::cout << "Results: " << pass << " passed, " << fail << " failed\n";
    std::cout << "========================================\n";

    return fail > 0 ? 1 : 0;
}
