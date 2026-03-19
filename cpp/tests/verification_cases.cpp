/**
 * C++ Verification Cases - Mirrors Python verification suite
 * Outputs CSV-style results for comparison with Python
 *
 * Cases implemented:
 *  1. Poiseuille flow (analytical comparison)
 *  2. Lid-driven cavity Re=100 (Ghia benchmark)
 *  4. Bubble column two-fluid (transient)
 *  6. MUSCL high-order schemes
 *  9. Phase change (Lee model)
 * 11. P1 Radiation
 * 12. AMR refinement
 * 14. 3D Cavity mesh
 * 16. Preconditioner comparison
 * 17. Adaptive time stepping
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>
#include <string>
#include <sstream>

#include "twofluid/mesh.hpp"
#include "twofluid/mesh_generator.hpp"
#include "twofluid/mesh_generator_3d.hpp"
#include "twofluid/fields.hpp"
#include "twofluid/fvm_operators.hpp"
#include "twofluid/gradient.hpp"
#include "twofluid/interpolation.hpp"
#include "twofluid/linear_solver.hpp"
#include "twofluid/simple_solver.hpp"
#include "twofluid/two_fluid_solver.hpp"
#include "twofluid/turbulence.hpp"
#include "twofluid/closure.hpp"
#include "twofluid/preconditioner.hpp"
#include "twofluid/time_control.hpp"
#include "twofluid/phase_change.hpp"
#include "twofluid/solid_conduction.hpp"
#include "twofluid/radiation.hpp"
#include "twofluid/amr.hpp"
#include "twofluid/vtk_writer.hpp"

using namespace twofluid;

struct CaseResult {
    int case_num;
    std::string name;
    bool converged;
    std::string metric_name;
    double metric_value;
    double wall_time_ms;
    std::string extra;
};

static std::vector<CaseResult> all_results;

void add_result(int num, const std::string& name, bool conv,
                const std::string& mname, double mval, double wt,
                const std::string& extra = "") {
    all_results.push_back({num, name, conv, mname, mval, wt, extra});
}

// ========== Case 1: Poiseuille ==========
void case1_poiseuille() {
    std::cout << "Case 1: Poiseuille flow...\n";
    auto t0 = std::chrono::high_resolution_clock::now();

    double L = 1.0, H = 0.1;
    int nx = 50, ny = 20;
    double rho = 1.0, mu = 0.01, dpdx = -1.0;
    double u_max_analytical = -dpdx * H * H / (8.0 * mu);

    auto mesh = generate_channel_mesh(L, H, nx, ny);
    SIMPLESolver solver(mesh, rho, mu);
    solver.max_iter = 500;
    solver.tol = 1e-5;
    solver.alpha_u = 0.7;
    solver.alpha_p = 0.3;

    // Parabolic inlet
    int n_inlet = static_cast<int>(mesh.boundary_patches["inlet"].size());
    Eigen::MatrixXd inlet_U(n_inlet, 2);
    for (int j = 0; j < n_inlet; ++j) {
        int fid = mesh.boundary_patches["inlet"][j];
        double y = mesh.faces[fid].center[1];
        inlet_U(j, 0) = -dpdx / (2.0 * mu) * y * (H - y);
        inlet_U(j, 1) = 0.0;
    }
    solver.set_inlet("inlet", inlet_U);
    solver.set_outlet("outlet", 0.0);
    solver.set_wall("wall_bottom");
    solver.set_wall("wall_top");

    auto result = solver.solve_steady();

    // Extract velocity at x=0.8L
    double x_target = 0.8;
    double tol_x = L / nx * 1.5;
    std::vector<double> y_num, u_num;
    for (int ci = 0; ci < mesh.n_cells; ++ci) {
        if (std::abs(mesh.cells[ci].center[0] - x_target) < tol_x) {
            y_num.push_back(mesh.cells[ci].center[1]);
            u_num.push_back(solver.velocity().values(ci, 0));
        }
    }

    // L2 error
    double l2_sum = 0.0, anal_max = 0.0;
    for (size_t i = 0; i < y_num.size(); ++i) {
        double u_anal = -dpdx / (2.0 * mu) * y_num[i] * (H - y_num[i]);
        double diff = u_num[i] - u_anal;
        l2_sum += diff * diff;
        anal_max = std::max(anal_max, std::abs(u_anal));
    }
    double L2_error = std::sqrt(l2_sum / y_num.size()) / std::max(anal_max, 1e-15);

    double u_max_num = *std::max_element(u_num.begin(), u_num.end());

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "  u_max_analytical=" << u_max_analytical
              << " u_max_numerical=" << u_max_num
              << " L2_error=" << L2_error << "\n";

    add_result(1, "Poiseuille", result.converged, "L2_error", L2_error, ms,
               "u_max=" + std::to_string(u_max_num));
    add_result(1, "Poiseuille", result.converged, "u_max_numerical", u_max_num, ms);
    add_result(1, "Poiseuille", result.converged, "u_max_analytical", u_max_analytical, ms);
    add_result(1, "Poiseuille", result.converged, "iterations", result.iterations, ms);
}

// ========== Case 2: Lid-Driven Cavity Re=100 ==========
void case2_cavity() {
    std::cout << "Case 2: Lid-driven cavity Re=100...\n";
    auto t0 = std::chrono::high_resolution_clock::now();

    double L = 1.0, U_lid = 1.0, rho = 1.0;
    int Re = 100;
    double mu = rho * U_lid * L / Re;
    int n_grid = 32;

    auto mesh = generate_cavity_mesh(L, n_grid);
    SIMPLESolver solver(mesh, rho, mu);
    solver.max_iter = 3000;
    solver.tol = 1e-4;
    solver.alpha_u = 0.7;
    solver.alpha_p = 0.3;

    int n_lid = static_cast<int>(mesh.boundary_patches["lid"].size());
    Eigen::MatrixXd lid_U(n_lid, 2);
    for (int j = 0; j < n_lid; ++j) {
        lid_U(j, 0) = U_lid;
        lid_U(j, 1) = 0.0;
    }
    solver.set_inlet("lid", lid_U);
    solver.set_wall("wall_bottom");
    solver.set_wall("wall_left");
    solver.set_wall("wall_right");

    auto result = solver.solve_steady();

    // Extract u at x=0.5 (vertical centerline)
    double x_mid = 0.5;
    double tol_x = L / n_grid * 1.5;
    std::vector<double> y_num, u_num;
    for (int ci = 0; ci < mesh.n_cells; ++ci) {
        if (std::abs(mesh.cells[ci].center[0] - x_mid) < tol_x) {
            y_num.push_back(mesh.cells[ci].center[1]);
            u_num.push_back(solver.velocity().values(ci, 0));
        }
    }
    // Sort by y
    std::vector<size_t> idx(y_num.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
        return y_num[a] < y_num[b];
    });

    // Ghia Re=100 data
    std::vector<double> ghia_y = {0.0, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719,
        0.2813, 0.4531, 0.5, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0};
    std::vector<double> ghia_u = {0.0, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150,
        -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151, 0.68717, 0.73722,
        0.78871, 0.84123, 1.0};

    // Interpolate numerical to Ghia y-points
    // Simple linear interpolation
    std::vector<double> y_sorted, u_sorted;
    for (auto i : idx) { y_sorted.push_back(y_num[i]); u_sorted.push_back(u_num[i]); }

    double l2_sum = 0.0;
    int n_ghia = static_cast<int>(ghia_y.size());
    for (int i = 0; i < n_ghia; ++i) {
        double yg = ghia_y[i];
        // Find bracket
        double u_interp = 0.0;
        if (yg <= y_sorted.front()) u_interp = u_sorted.front();
        else if (yg >= y_sorted.back()) u_interp = u_sorted.back();
        else {
            for (size_t j = 0; j + 1 < y_sorted.size(); ++j) {
                if (y_sorted[j] <= yg && y_sorted[j+1] >= yg) {
                    double t = (yg - y_sorted[j]) / (y_sorted[j+1] - y_sorted[j]);
                    u_interp = u_sorted[j] + t * (u_sorted[j+1] - u_sorted[j]);
                    break;
                }
            }
        }
        double diff = u_interp - ghia_u[i];
        l2_sum += diff * diff;
    }
    double L2_ghia = std::sqrt(l2_sum / n_ghia);

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "  Ghia L2 error=" << L2_ghia << " iterations=" << result.iterations << "\n";

    add_result(2, "Cavity_Re100", result.converged, "Ghia_L2_error", L2_ghia, ms);
    add_result(2, "Cavity_Re100", result.converged, "iterations", result.iterations, ms);
}

// ========== Case 4: Bubble Column ==========
void case4_bubble_column() {
    std::cout << "Case 4: Bubble column (two-fluid)...\n";
    auto t0 = std::chrono::high_resolution_clock::now();

    double Lx = 0.15, Ly = 0.45;
    int nx = 8, ny = 20;

    auto mesh = generate_channel_mesh(Lx, Ly, nx, ny);
    TwoFluidSolver tf(mesh);
    tf.rho_l = 998.2; tf.rho_g = 1.225;
    tf.mu_l = 1.003e-3; tf.mu_g = 1.789e-5;
    tf.d_b = 0.005;
    tf.alpha_u = 0.3; tf.alpha_p = 0.2; tf.alpha_alpha = 0.3;
    tf.tol = 1e-3;
    tf.max_outer_iter = 200;
    tf.solve_energy = false;
    tf.solve_momentum = true;

    tf.initialize(0.001);

    Eigen::VectorXd U_l_in(2); U_l_in << 0.0, 0.0;
    Eigen::VectorXd U_g_in(2); U_g_in << 0.0, 0.1;
    tf.set_inlet_bc("inlet", 0.04, U_l_in, U_g_in);
    tf.set_outlet_bc("outlet", 0.0);
    tf.set_wall_bc("wall_bottom");
    tf.set_wall_bc("wall_top");

    auto result = tf.solve_transient(1.0, 0.02, 50);

    double alpha_max = tf.alpha_g_field().max();
    double alpha_mean = tf.alpha_g_field().mean();

    // Check buoyancy: top half should have more gas
    double alpha_top = 0.0, alpha_bot = 0.0;
    int n_top = 0, n_bot = 0;
    double y_mid = Ly / 2.0;
    for (int i = 0; i < mesh.n_cells; ++i) {
        if (mesh.cells[i].center[1] > y_mid) {
            alpha_top += tf.alpha_g_field().values(i); n_top++;
        } else {
            alpha_bot += tf.alpha_g_field().values(i); n_bot++;
        }
    }
    alpha_top /= std::max(n_top, 1);
    alpha_bot /= std::max(n_bot, 1);

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "  alpha_max=" << alpha_max << " alpha_mean=" << alpha_mean
              << " buoyancy=" << (alpha_top > alpha_bot) << "\n";

    add_result(4, "Bubble_Column", true, "alpha_g_max", alpha_max, ms);
    add_result(4, "Bubble_Column", true, "alpha_g_mean", alpha_mean, ms);
    add_result(4, "Bubble_Column", true, "time_steps", result.iterations, ms);
}

// ========== Case 6: MUSCL ==========
void case6_muscl() {
    std::cout << "Case 6: MUSCL high-order schemes...\n";
    auto t0 = std::chrono::high_resolution_clock::now();

    auto mesh = generate_channel_mesh(1.0, 0.1, 50, 10);
    double rho = 1.0, mu = 0.001;
    SIMPLESolver solver(mesh, rho, mu);
    solver.max_iter = 300;
    solver.tol = 1e-4;

    int n_inlet = static_cast<int>(mesh.boundary_patches["inlet"].size());
    Eigen::MatrixXd inlet_U(n_inlet, 2);
    for (int j = 0; j < n_inlet; ++j) {
        inlet_U(j, 0) = 1.0; inlet_U(j, 1) = 0.0;
    }
    solver.set_inlet("inlet", inlet_U);
    solver.set_outlet("outlet", 0.0);
    solver.set_wall("wall_bottom");
    solver.set_wall("wall_top");

    auto result = solver.solve_steady();

    // Test MUSCL deferred correction with all limiters
    ScalarField phi(mesh, "phi");
    for (int i = 0; i < mesh.n_cells; ++i)
        phi.values(i) = mesh.cells[i].center[0];

    auto mass_flux = compute_mass_flux(solver.velocity(), rho, mesh);
    auto grad = green_gauss_gradient(phi);

    std::vector<std::string> limiters = {"van_leer", "minmod", "superbee", "van_albada"};
    for (auto& lim : limiters) {
        auto dc = muscl_deferred_correction(mesh, phi, mass_flux, grad, lim);
        double dc_max = dc.cwiseAbs().maxCoeff();
        std::cout << "  " << lim << " correction max=" << dc_max << "\n";
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    add_result(6, "MUSCL", result.converged, "iterations", result.iterations, ms,
               "4_limiters_tested");
}

// ========== Case 9: Phase Change ==========
void case9_phase_change() {
    std::cout << "Case 9: Phase change (Lee model)...\n";
    auto t0 = std::chrono::high_resolution_clock::now();

    double T_sat_val = saturation_temperature(101325.0);
    double h_fg_val = water_latent_heat(101325.0);
    auto wp = water_properties(101325.0);

    auto mesh = generate_channel_mesh(0.5, 0.1, 20, 10);
    LeePhaseChangeModel lee(mesh, 373.15, 0.1, 0.1, 2.26e6, 1000.0, 1.0);

    ScalarField T(mesh, "T");
    ScalarField alpha_l(mesh, "alpha_l");

    // Test evaporation
    T.set_uniform(383.15); // 10K superheat
    alpha_l.set_uniform(0.9);
    auto dot_m = lee.compute_mass_transfer(T, alpha_l);
    double dot_m_evap = dot_m.mean();

    // Test condensation
    T.set_uniform(363.15); // 10K subcool
    dot_m = lee.compute_mass_transfer(T, alpha_l);
    double dot_m_cond = dot_m.mean();

    // Zuber CHF
    ZuberCHFModel zuber(2.26e6, 958.0, 0.6, 0.059);
    double chf = zuber.compute_chf();

    // Rohsenow
    RohsenowBoilingModel rohsenow(373.15, 2.26e6, 958.0, 0.6,
                                   2.82e-4, 4216.0, 0.059, 1.75);
    double q_rohsenow = rohsenow.compute_wall_heat_flux(383.15);

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "  T_sat=" << T_sat_val << " h_fg=" << h_fg_val
              << " dot_m_evap=" << dot_m_evap << " dot_m_cond=" << dot_m_cond << "\n";

    add_result(9, "Phase_Change", true, "T_sat_1atm", T_sat_val, ms);
    add_result(9, "Phase_Change", true, "h_fg_1atm", h_fg_val, ms);
    add_result(9, "Phase_Change", true, "Lee_evap_rate", dot_m_evap, ms);
    add_result(9, "Phase_Change", true, "Lee_cond_rate", dot_m_cond, ms);
    add_result(9, "Phase_Change", true, "Zuber_CHF", chf, ms);
    add_result(9, "Phase_Change", true, "Rohsenow_q_10K", q_rohsenow, ms);
}

// ========== Case 11: Radiation ==========
void case11_radiation() {
    std::cout << "Case 11: P1 Radiation...\n";
    auto t0 = std::chrono::high_resolution_clock::now();

    auto mesh = generate_channel_mesh(1.0, 1.0, 20, 20);
    P1RadiationModel rad(mesh, 1.0);

    rad.set_bc("inlet", "marshak", 1000.0);
    rad.set_bc("outlet", "marshak", 500.0);
    rad.set_bc("wall_bottom", "zero_gradient");
    rad.set_bc("wall_top", "zero_gradient");

    ScalarField T(mesh, "T");
    T.set_uniform(800.0);

    auto result = rad.solve(T, 200, 1e-6);

    auto q_r = rad.compute_radiative_source(T);

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "  converged=" << result.converged << " iterations=" << result.iterations
              << " q_r_range=[" << q_r.minCoeff() << ", " << q_r.maxCoeff() << "]\n";

    add_result(11, "Radiation_P1", result.converged, "iterations", result.iterations, ms);
    add_result(11, "Radiation_P1", result.converged, "G_max", rad.G.max(), ms);
    add_result(11, "Radiation_P1", result.converged, "G_min", rad.G.min(), ms);
    add_result(11, "Radiation_P1", result.converged, "q_r_max", q_r.maxCoeff(), ms);
}

// ========== Case 12: AMR ==========
void case12_amr() {
    std::cout << "Case 12: AMR...\n";
    auto t0 = std::chrono::high_resolution_clock::now();

    auto mesh = generate_cavity_mesh(1.0, 8);
    AMRMesh amr(mesh, 2);

    // Solve on base mesh to get error indicator
    ScalarField phi(mesh, "phi");
    for (int i = 0; i < mesh.n_cells; ++i) {
        double x = mesh.cells[i].center[0];
        double y = mesh.cells[i].center[1];
        // Sharp gradient in corner
        phi.values(i) = std::exp(-20.0 * ((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)));
    }

    auto error = GradientJumpEstimator::estimate(mesh, phi);
    AMRSolverLoop loop(amr, 0.3);
    auto to_refine = loop.mark_cells(mesh, phi);
    int n_refine = static_cast<int>(to_refine.size());

    amr.refine_cells(to_refine);
    auto refined = amr.get_active_mesh();

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "  base=" << mesh.n_cells << " refined=" << refined.n_cells
              << " cells_refined=" << n_refine << "\n";

    add_result(12, "AMR", true, "base_cells", mesh.n_cells, ms);
    add_result(12, "AMR", true, "refined_cells", refined.n_cells, ms);
    add_result(12, "AMR", true, "cells_marked", n_refine, ms);
}

// ========== Case 14: 3D Cavity ==========
void case14_3d_cavity() {
    std::cout << "Case 14: 3D Cavity mesh...\n";
    auto t0 = std::chrono::high_resolution_clock::now();

    auto mesh = generate_3d_cavity_mesh(1.0, 1.0, 1.0, 8, 8, 8);

    double total_vol = 0.0;
    for (auto& c : mesh.cells) total_vol += c.volume;

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "  cells=" << mesh.n_cells << " faces=" << mesh.n_faces
              << " volume=" << total_vol << "\n";

    add_result(14, "3D_Cavity", true, "n_cells", mesh.n_cells, ms);
    add_result(14, "3D_Cavity", true, "n_faces", mesh.n_faces, ms);
    add_result(14, "3D_Cavity", true, "total_volume", total_vol, ms);
}

// ========== Case 16: Preconditioner ==========
void case16_preconditioner() {
    std::cout << "Case 16: Preconditioner comparison...\n";

    auto mesh = generate_channel_mesh(1.0, 0.1, 30, 15);
    int n = mesh.n_cells;
    FVMSystem system(n);
    ScalarField gamma(mesh, "g");
    gamma.set_uniform(1.0);
    diffusion_operator(mesh, gamma, system);
    for (int i = 0; i < n; ++i) system.add_diagonal(i, 10.0);
    for (int i = 0; i < n; ++i) system.add_source(i, 1.0);

    auto A = system.to_sparse();
    A.makeCompressed();

    auto test_precond = [&](const std::string& method) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto [fn, info] = create_preconditioner(A, method);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "  " << method << " setup=" << info.setup_time << "s\n";
        add_result(16, "Preconditioner", true, method + "_setup_ms", ms, ms);
    };

    test_precond("none");
    test_precond("jacobi");
    test_precond("ilu0");
}

// ========== Case 17: Adaptive Time Control ==========
void case17_adaptive_dt() {
    std::cout << "Case 17: Adaptive time control...\n";
    auto t0 = std::chrono::high_resolution_clock::now();

    auto mesh = generate_channel_mesh(1.0, 0.1, 20, 10);
    AdaptiveTimeControl tc(0.01, 1e-8, 1.0, 0.5, 1.0, 0.5, 1.2, 0.5, 0.9);

    Eigen::VectorXd u_mag = Eigen::VectorXd::Constant(mesh.n_cells, 1.0);

    // Simulate 5 steps
    for (int step = 0; step < 5; ++step) {
        auto info = tc.compute_dt(mesh, u_mag, 1e-4, true);
    }

    // One divergence step
    auto info_div = tc.compute_dt(mesh, u_mag, -1.0, false);

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "  dt_history: ";
    for (auto dt : tc.dt_history) std::cout << dt << " ";
    std::cout << "\n";

    add_result(17, "Adaptive_dt", true, "final_dt", tc.dt(), ms);
    add_result(17, "Adaptive_dt", true, "n_steps", tc.dt_history.size(), ms);
    add_result(17, "Adaptive_dt", true, "dt_after_divergence", info_div.dt, ms);
}

// ========== Main ==========
int main() {
    std::cout << "================================================================\n";
    std::cout << "  Two-Fluid FVM C++ Verification Cases\n";
    std::cout << "================================================================\n\n";

    case1_poiseuille();
    case2_cavity();
    case4_bubble_column();
    case6_muscl();
    case9_phase_change();
    case11_radiation();
    case12_amr();
    case14_3d_cavity();
    case16_preconditioner();
    case17_adaptive_dt();

    // Print results table
    std::cout << "\n================================================================\n";
    std::cout << "  RESULTS TABLE (for Python comparison)\n";
    std::cout << "================================================================\n";
    std::cout << std::left << std::setw(6) << "Case"
              << std::setw(20) << "Name"
              << std::setw(6) << "Conv"
              << std::setw(25) << "Metric"
              << std::setw(18) << "Value"
              << std::setw(12) << "Time(ms)" << "\n";
    std::cout << std::string(87, '-') << "\n";

    for (auto& r : all_results) {
        std::cout << std::left << std::setw(6) << r.case_num
                  << std::setw(20) << r.name
                  << std::setw(6) << (r.converged ? "Y" : "N")
                  << std::setw(25) << r.metric_name
                  << std::setw(18) << std::setprecision(6) << r.metric_value
                  << std::setw(12) << std::setprecision(1) << std::fixed << r.wall_time_ms
                  << "\n";
    }

    // CSV output
    std::cout << "\n--- CSV ---\n";
    std::cout << "case,name,converged,metric,value\n";
    for (auto& r : all_results) {
        std::cout << r.case_num << "," << r.name << ","
                  << (r.converged ? "Y" : "N") << ","
                  << r.metric_name << ","
                  << std::setprecision(10) << std::scientific << r.metric_value << "\n";
    }

    return 0;
}
