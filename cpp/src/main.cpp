/**
 * Standalone Two-Fluid FVM Solver
 *
 * Usage:
 *   twofluid_solver --case channel|cavity|bfs [options]
 *
 * Options:
 *   --nx N          Cells in x-direction (default: 50)
 *   --ny N          Cells in y-direction (default: same as nx)
 *   --Re N          Reynolds number (default: 100)
 *   --turbulence    Enable k-epsilon turbulence model
 *   --max-iter N    Maximum SIMPLE iterations (default: 500)
 *   --tol F         Convergence tolerance (default: 1e-5)
 *   --output FILE   Output VTU file (default: result.vtu)
 */

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#include "twofluid/mesh.hpp"
#include "twofluid/mesh_generator.hpp"
#include "twofluid/fields.hpp"
#include "twofluid/simple_solver.hpp"
#include "twofluid/two_fluid_solver.hpp"
#include "twofluid/vtk_writer.hpp"

using namespace twofluid;

struct Config {
    std::string case_type = "channel";
    int nx = 50;
    int ny = -1;  // -1 means use default per case
    double Re = 100.0;
    bool turbulence = false;
    int max_iter = 300;
    double tol = 1e-5;
    std::string output = "result.vtu";
};

static void print_usage() {
    std::cout <<
        "Usage: twofluid_solver --case channel|cavity|bfs|bubble [options]\n"
        "\n"
        "Options:\n"
        "  --case TYPE       Case type: channel, cavity, bfs, bubble (required)\n"
        "  --nx N            Cells in x-direction (default: 50)\n"
        "  --ny N            Cells in y-direction (default: depends on case)\n"
        "  --Re N            Reynolds number (default: 100)\n"
        "  --turbulence      Enable k-epsilon turbulence model\n"
        "  --max-iter N      Maximum SIMPLE iterations (default: 500)\n"
        "  --tol F           Convergence tolerance (default: 1e-5)\n"
        "  --output FILE     Output VTU file (default: result.vtu)\n"
        "  --help            Show this help message\n"
        "\n"
        "Examples:\n"
        "  twofluid_solver --case channel --nx 100 --ny 50 --output channel.vtu\n"
        "  twofluid_solver --case cavity --nx 50 --Re 1000 --output cavity.vtu\n"
        "  twofluid_solver --case bfs --nx 120 --ny 40 --Re 5100 --turbulence --output bfs.vtu\n";
}

static Config parse_args(int argc, char* argv[]) {
    Config cfg;
    bool has_case = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage();
            std::exit(0);
        } else if (arg == "--case" && i + 1 < argc) {
            cfg.case_type = argv[++i];
            has_case = true;
        } else if (arg == "--nx" && i + 1 < argc) {
            cfg.nx = std::atoi(argv[++i]);
        } else if (arg == "--ny" && i + 1 < argc) {
            cfg.ny = std::atoi(argv[++i]);
        } else if (arg == "--Re" && i + 1 < argc) {
            cfg.Re = std::atof(argv[++i]);
        } else if (arg == "--turbulence") {
            cfg.turbulence = true;
        } else if (arg == "--max-iter" && i + 1 < argc) {
            cfg.max_iter = std::atoi(argv[++i]);
        } else if (arg == "--tol" && i + 1 < argc) {
            cfg.tol = std::atof(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            cfg.output = argv[++i];
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage();
            std::exit(1);
        }
    }

    if (!has_case) {
        std::cerr << "Error: --case is required.\n\n";
        print_usage();
        std::exit(1);
    }

    return cfg;
}

// ---------------------------------------------------------------------------
// Case setup: each returns mesh + solver pair
// ---------------------------------------------------------------------------

struct CaseData {
    std::unique_ptr<FVMesh> mesh;
    std::unique_ptr<SIMPLESolver> solver;
    std::unique_ptr<TwoFluidSolver> tf_solver;  // for two-fluid cases
};

static CaseData setup_channel(const Config& cfg) {
    double Ly = 1.0;
    int ny = (cfg.ny > 0) ? cfg.ny : cfg.nx / 2;
    double Lx = 2.0 * Ly;
    double U_mean = 1.0;
    double mu = U_mean * Ly / cfg.Re;

    auto mesh = std::make_unique<FVMesh>(generate_channel_mesh(Lx, Ly, cfg.nx, ny));
    auto solver = std::make_unique<SIMPLESolver>(*mesh, 1.0, mu);

    // Parabolic inlet velocity profile
    int n_inlet = static_cast<int>(mesh->boundary_patches["inlet"].size());
    Eigen::MatrixXd inlet_U(n_inlet, 2);
    for (int j = 0; j < n_inlet; ++j) {
        int fid = mesh->boundary_patches["inlet"][j];
        double y = mesh->faces[fid].center[1];
        double u_para = 6.0 * U_mean * y * (Ly - y) / (Ly * Ly);
        inlet_U(j, 0) = u_para;
        inlet_U(j, 1) = 0.0;
    }
    solver->set_inlet("inlet", inlet_U);
    solver->set_outlet("outlet", 0.0);
    solver->set_wall("wall_bottom");
    solver->set_wall("wall_top");

    std::cout << "Channel flow: Lx=" << Lx << " Ly=" << Ly
              << " nx=" << cfg.nx << " ny=" << ny
              << " Re=" << cfg.Re << " mu=" << mu << "\n";

    return {std::move(mesh), std::move(solver)};
}

static CaseData setup_cavity(const Config& cfg) {
    double L = 1.0;
    double U_lid = 1.0;
    double mu = U_lid * L / cfg.Re;

    auto mesh = std::make_unique<FVMesh>(generate_cavity_mesh(L, cfg.nx));
    auto solver = std::make_unique<SIMPLESolver>(*mesh, 1.0, mu);

    // Lid: moving wall with U=(1,0)
    int n_lid = static_cast<int>(mesh->boundary_patches["lid"].size());
    Eigen::MatrixXd lid_U(n_lid, 2);
    for (int j = 0; j < n_lid; ++j) {
        lid_U(j, 0) = U_lid;
        lid_U(j, 1) = 0.0;
    }
    solver->set_inlet("lid", lid_U);  // Dirichlet velocity = (1,0)

    // Other walls
    solver->set_wall("wall_bottom");
    solver->set_wall("wall_left");
    solver->set_wall("wall_right");

    std::cout << "Lid-driven cavity: L=" << L << " n=" << cfg.nx
              << " Re=" << cfg.Re << " mu=" << mu << "\n";

    return {std::move(mesh), std::move(solver)};
}

static CaseData setup_bfs(const Config& cfg) {
    double step_height = 1.0;
    double expansion_ratio = 2.0;
    double H = step_height * expansion_ratio;
    double H_in = H - step_height;
    double L_up = 5.0;
    double L_down = 30.0;
    int ny = (cfg.ny > 0) ? cfg.ny : 40;
    int nx_up = cfg.nx / 4;
    int nx_down = cfg.nx - nx_up;

    double U_mean = 1.0;
    double mu = U_mean * H_in / cfg.Re;

    auto mesh = std::make_unique<FVMesh>(generate_bfs_mesh(
        step_height, expansion_ratio, L_up, L_down, nx_up, nx_down, ny));
    auto solver = std::make_unique<SIMPLESolver>(*mesh, 1.0, mu);

    // Uniform inlet velocity above the step
    int n_inlet = static_cast<int>(mesh->boundary_patches["inlet"].size());
    Eigen::MatrixXd inlet_U(n_inlet, 2);
    for (int j = 0; j < n_inlet; ++j) {
        inlet_U(j, 0) = U_mean;
        inlet_U(j, 1) = 0.0;
    }
    solver->set_inlet("inlet", inlet_U);
    solver->set_outlet("outlet", 0.0);
    solver->set_wall("wall_bottom");
    solver->set_wall("wall_top");
    solver->set_wall("wall_step_top");
    solver->set_wall("wall_step_inlet");

    std::cout << "Backward-facing step: H=" << H << " step=" << step_height
              << " nx=" << (nx_up + nx_down) << " ny=" << ny
              << " Re=" << cfg.Re << " mu=" << mu << "\n";

    return {std::move(mesh), std::move(solver)};
}

// ---------------------------------------------------------------------------
// Bubble column (Two-Fluid)
// ---------------------------------------------------------------------------

static CaseData setup_bubble(const Config& cfg) {
    double Lx = 0.15;
    double Ly = 0.45;
    int ny = (cfg.ny > 0) ? cfg.ny : 20;

    auto mesh = std::make_unique<FVMesh>(generate_channel_mesh(Lx, Ly, cfg.nx, ny));
    auto tf = std::make_unique<TwoFluidSolver>(*mesh);

    // Physical properties (water-air at atmospheric)
    tf->rho_l = 998.2;
    tf->rho_g = 1.225;
    tf->mu_l = 1.003e-3;
    tf->mu_g = 1.789e-5;
    tf->d_b = 0.005;

    // Solver parameters
    tf->alpha_u = 0.3;
    tf->alpha_p = 0.2;
    tf->alpha_alpha = 0.3;
    tf->tol = 1e-3;
    tf->max_outer_iter = cfg.max_iter;
    tf->solve_energy = false;
    tf->solve_momentum = true;

    // Initialize
    tf->initialize(0.01);

    // Inlet: bottom, gas enters with alpha_g=0.05, liquid stagnant
    Eigen::VectorXd U_l_in(2); U_l_in << 0.0, 0.01;
    Eigen::VectorXd U_g_in(2); U_g_in << 0.0, 0.1;
    tf->set_inlet_bc("inlet", 0.05, U_l_in, U_g_in);

    // Outlet: top
    tf->set_outlet_bc("outlet", 0.0);

    // Walls
    tf->set_wall_bc("wall_bottom", 0.0);
    tf->set_wall_bc("wall_top", 0.0);

    std::cout << "Bubble column: Lx=" << Lx << " Ly=" << Ly
              << " nx=" << cfg.nx << " ny=" << ny << "\n";

    return {std::move(mesh), nullptr, std::move(tf)};
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    Config cfg = parse_args(argc, argv);

    std::cout << "=== Two-Fluid FVM Solver ===\n\n";

    CaseData data;
    if (cfg.case_type == "channel") {
        data = setup_channel(cfg);
    } else if (cfg.case_type == "cavity") {
        data = setup_cavity(cfg);
    } else if (cfg.case_type == "bfs") {
        data = setup_bfs(cfg);
    } else if (cfg.case_type == "bubble" || cfg.case_type == "two_fluid") {
        data = setup_bubble(cfg);
    } else {
        std::cerr << "Unknown case type: " << cfg.case_type << "\n";
        return 1;
    }

    FVMesh& mesh = *data.mesh;

    std::cout << "Mesh: " << mesh.n_cells << " cells, "
              << mesh.n_faces << " faces ("
              << mesh.n_internal_faces << " internal, "
              << mesh.n_boundary_faces << " boundary)\n\n";

    // Two-fluid solver path
    if (data.tf_solver) {
        TwoFluidSolver& tf = *data.tf_solver;

        std::cout << "Solving (Two-Fluid transient)...\n";
        double t_end_sim = 0.05;  // short simulation for testing
        double dt_sim = 0.001;
        int report = 10;
        auto result_ptr = std::make_unique<SolveResult>(
            tf.solve_transient(t_end_sim, dt_sim, report));
        SolveResult& result = *result_ptr;

        std::cout << "\n=== Results ===" << std::endl;
        std::cout << "  Converged:    " << (result.converged ? "yes" : "no") << std::endl;
        std::cout << "  Time steps:   " << result.iterations << std::endl;
        std::cout << "  Wall time:    " << result.wall_time << " s" << std::endl;
        if (!result.residuals.empty()) {
            std::cout << "  Final residual: " << result.residuals.back() << std::endl;
        }

        std::cout << "  alpha_g range: [" << tf.alpha_g_field().min()
                  << ", " << tf.alpha_g_field().max() << "]" << std::endl;

        double ul_max = tf.U_l_field().magnitude().maxCoeff();
        double ug_max = tf.U_g_field().magnitude().maxCoeff();
        std::cout << "  |U_l| max:    " << ul_max << std::endl;
        std::cout << "  |U_g| max:    " << ug_max << std::endl;
        std::cout << "  p range:      [" << tf.pressure().min()
                  << ", " << tf.pressure().max() << "]" << std::endl;

        // Write VTU
        std::unordered_map<std::string, Eigen::VectorXd> scalar_data;
        std::unordered_map<std::string, Eigen::MatrixXd> vector_data;

        scalar_data["pressure"] = tf.pressure().values;
        scalar_data["alpha_gas"] = tf.alpha_g_field().values;
        scalar_data["alpha_liquid"] = tf.alpha_l_field().values;
        vector_data["velocity_liquid"] = tf.U_l_field().values;
        vector_data["velocity_gas"] = tf.U_g_field().values;

        if (tf.solve_energy) {
            scalar_data["T_liquid"] = tf.T_l_field().values;
            scalar_data["T_gas"] = tf.T_g_field().values;
        }

        std::cout << "  Writing VTU... cells=" << mesh.n_cells
                  << " cells_vec=" << mesh.cells.size()
                  << " nodes=" << mesh.nodes.rows() << "x" << mesh.nodes.cols()
                  << std::endl;
        if (!mesh.cells.empty()) {
            std::cout << "  cell[0].nodes=" << mesh.cells[0].nodes.size()
                      << " max_node_id=";
            int mx = 0;
            for (auto& c : mesh.cells) for (auto ni : c.nodes) mx = std::max(mx, ni);
            std::cout << mx << " (n_points=" << mesh.nodes.rows() << ")" << std::endl;
        }
        try {
            write_vtu(cfg.output, mesh, scalar_data, vector_data);
            std::cout << "\n  Output written to: " << cfg.output << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "  VTU write failed: " << e.what() << std::endl;
        }

        return result.converged ? 0 : 1;
    }

    // Single-phase SIMPLE solver path
    SIMPLESolver& solver = *data.solver;

    // Configure solver
    solver.max_iter = cfg.max_iter;
    solver.tol = cfg.tol;

    // Enable turbulence if requested
    if (cfg.turbulence) {
        solver.enable_turbulence(0.001, 0.01);
        std::cout << "Turbulence model: k-epsilon enabled\n\n";
    }

    // Solve
    std::cout << "Solving...\n";
    auto t_start = std::chrono::high_resolution_clock::now();
    SolveResult result = solver.solve_steady();
    auto t_end_clock = std::chrono::high_resolution_clock::now();
    double total_wall = std::chrono::duration<double>(t_end_clock - t_start).count();

    // Print results
    std::cout << "\n=== Results ===\n";
    std::cout << "  Converged:    " << (result.converged ? "yes" : "no") << "\n";
    std::cout << "  Iterations:   " << result.iterations << "\n";
    std::cout << "  Wall time:    " << total_wall << " s\n";
    if (!result.residuals.empty()) {
        std::cout << "  Final residual: " << result.residuals.back() << "\n";
    }

    // Velocity statistics
    VectorField& U = solver.velocity();
    ScalarField& p = solver.pressure();
    Eigen::VectorXd U_mag = U.magnitude();
    std::cout << "  |U| max:      " << U_mag.maxCoeff() << "\n";
    std::cout << "  |U| mean:     " << U_mag.mean() << "\n";
    std::cout << "  p range:      [" << p.min() << ", " << p.max() << "]\n";

    if (cfg.turbulence && solver.turbulence_model()) {
        auto* turb = solver.turbulence_model();
        std::cout << "  k range:      [" << turb->k().min() << ", "
                  << turb->k().max() << "]\n";
        std::cout << "  epsilon range: [" << turb->epsilon().min() << ", "
                  << turb->epsilon().max() << "]\n";
    }

    // Write VTU output
    std::unordered_map<std::string, Eigen::VectorXd> scalar_data;
    std::unordered_map<std::string, Eigen::MatrixXd> vector_data;

    scalar_data["pressure"] = p.values;
    scalar_data["velocity_magnitude"] = U_mag;
    vector_data["velocity"] = U.values;

    if (cfg.turbulence && solver.turbulence_model()) {
        auto* turb = solver.turbulence_model();
        scalar_data["k"] = turb->k().values;
        scalar_data["epsilon"] = turb->epsilon().values;
        scalar_data["mu_t"] = turb->get_mu_t();
    }

    write_vtu(cfg.output, mesh, scalar_data, vector_data);
    std::cout << "\n  Output written to: " << cfg.output << "\n";

    return result.converged ? 0 : 1;
}
