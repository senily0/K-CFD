/**
 * MPI Performance Benchmark -- Domain-Decomposed Solve
 *
 * Large-scale 3D channel flow for scaling test.
 * Mesh: configurable (default 40x20x20 = 16K cells)
 *
 * Usage:
 *   Serial:   benchmark_mpi [nx] [ny] [nz] [max_iter]
 *   Parallel: mpiexec -n 4 benchmark_mpi [nx] [ny] [nz] [max_iter]
 *
 * Parallel mode uses real domain decomposition:
 *   - Rank 0 generates full mesh and partitions with RCB
 *   - Partition IDs are broadcast to all ranks
 *   - Each rank extracts its LOCAL submesh
 *   - Outer Schwarz loop: local SIMPLE iterations + MPI ghost exchange
 *   - mpi_interface BCs are Dirichlet, updated with received neighbor values
 *   - Global convergence via MPI_Allreduce
 *
 * Hardware target: Intel Core Ultra 5 125H (14C/18T), 32GB RAM
 */

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <set>
#include <numeric>

#ifdef USE_MPI
#include <mpi.h>
#endif

#include "twofluid/mesh.hpp"
#include "twofluid/mesh_generator_3d.hpp"
#include "twofluid/fields.hpp"
#include "twofluid/simple_solver.hpp"
#include "twofluid/vtk_writer.hpp"
#include "twofluid/mpi_comm.hpp"
#include "twofluid/partitioning.hpp"

using namespace twofluid;

// ---------------------------------------------------------------------------
// Ghost exchange data structure for mpi_interface boundary faces.
//
// For each mpi_interface face on the local clean_mesh, we record:
//   - local_face_id: face index in clean_mesh
//   - local_owner_cell: local cell that owns this face
//   - remote_rank: the MPI rank that owns the cell on the other side
//   - remote_global_cell: the global cell ID of the remote neighbor
//   - patch_local_idx: index of this face within the "mpi_interface" patch
// ---------------------------------------------------------------------------
struct MPIGhostFace {
    int local_face_id;
    int local_owner_cell;
    int remote_rank;
    int remote_global_cell;
    int patch_local_idx;
};

// ---------------------------------------------------------------------------
// Per-neighbor exchange buffer structure.
// After setup, for each neighbor rank we know:
//   - which local cells to pack and send (their values are needed by neighbor)
//   - which patch face indices to write received values into
// ---------------------------------------------------------------------------
struct NeighborExchange {
    int remote_rank;
    // Cells whose values we send TO this neighbor (local cell indices).
    // Order matches what the neighbor expects to receive.
    std::vector<int> send_cells;
    // For each value we RECEIVE from this neighbor, the patch-local index
    // in the "mpi_interface" boundary_values array where we write it.
    std::vector<int> recv_patch_indices;
    // For each value we receive, the local owner cell (for blending).
    std::vector<int> recv_owner_cells;
};

// ---------------------------------------------------------------------------
// Helper: set BCs on local solver based on which patches exist.
// mpi_interface is set as Dirichlet (inlet) for velocity so that the
// solver uses the boundary_values we provide (updated each outer iteration).
// Pressure at mpi_interface uses zero-gradient (Neumann) -- the pressure
// boundary_values are updated directly on the field each outer iteration.
// ---------------------------------------------------------------------------
static void set_local_bcs(SIMPLESolver& solver, FVMesh& local_mesh,
                          double Ly, double Lz, double U_mean) {
    auto& patches = local_mesh.boundary_patches;

    if (patches.count("inlet") && !patches["inlet"].empty()) {
        int n_inlet = static_cast<int>(patches["inlet"].size());
        Eigen::MatrixXd inlet_U(n_inlet, 3);
        for (int j = 0; j < n_inlet; ++j) {
            int fid = patches["inlet"][j];
            double y = local_mesh.faces[fid].center[1];
            double z = local_mesh.faces[fid].center[2];
            double u_prof = 2.25 * U_mean * 4.0 * (y/Ly) * (1.0 - y/Ly)
                            * 4.0 * (z/Lz) * (1.0 - z/Lz);
            inlet_U(j, 0) = u_prof;
            inlet_U(j, 1) = 0.0;
            inlet_U(j, 2) = 0.0;
        }
        solver.set_inlet("inlet", inlet_U);
    }

    if (patches.count("outlet") && !patches["outlet"].empty()) {
        solver.set_outlet("outlet", 0.0);
    }

    if (patches.count("wall_bottom") && !patches["wall_bottom"].empty())
        solver.set_wall("wall_bottom");
    if (patches.count("wall_top") && !patches["wall_top"].empty())
        solver.set_wall("wall_top");
    if (patches.count("wall_front") && !patches["wall_front"].empty())
        solver.set_wall("wall_front");
    if (patches.count("wall_back") && !patches["wall_back"].empty())
        solver.set_wall("wall_back");

    // mpi_interface: Dirichlet velocity (values updated from neighbor each
    // outer iteration) and zero-gradient pressure (set_inlet gives exactly
    // this: bc_u_=dirichlet, bc_p_=zero_gradient).
    if (patches.count("mpi_interface") && !patches["mpi_interface"].empty()) {
        int n_mpi = static_cast<int>(patches["mpi_interface"].size());
        Eigen::MatrixXd mpi_U = Eigen::MatrixXd::Zero(n_mpi, 3);
        solver.set_inlet("mpi_interface", mpi_U);
    }
}

// ---------------------------------------------------------------------------
// Build the per-face ghost mapping during clean_mesh construction.
// Returns a vector of MPIGhostFace, one per mpi_interface face.
// ---------------------------------------------------------------------------
struct CleanMeshResult {
    FVMesh mesh;
    std::vector<MPIGhostFace> ghost_faces;
    std::unordered_map<int, int> gcell_to_local;  // global cell -> local cell
};

static CleanMeshResult build_clean_mesh(
    const FVMesh& global_mesh,
    const std::vector<int>& part_ids,
    int my_rank)
{
    CleanMeshResult result;
    FVMesh& clean_mesh = result.mesh;
    clean_mesh = FVMesh(3);

    // Collect owned global cell IDs
    std::vector<int> owned_global;
    for (int ci = 0; ci < global_mesh.n_cells; ++ci) {
        if (part_ids[ci] == my_rank) owned_global.push_back(ci);
    }

    // Collect used nodes
    std::set<int> used_nodes;
    for (int ci : owned_global) {
        for (int nid : global_mesh.cells[ci].nodes) used_nodes.insert(nid);
    }

    // Remap nodes
    std::unordered_map<int, int> node_map;
    int new_nid = 0;
    clean_mesh.nodes.resize(used_nodes.size(), 3);
    for (int old_nid : used_nodes) {
        node_map[old_nid] = new_nid;
        clean_mesh.nodes.row(new_nid) = global_mesh.nodes.row(old_nid);
        new_nid++;
    }

    // Create cells
    clean_mesh.n_cells = static_cast<int>(owned_global.size());
    clean_mesh.cells.resize(clean_mesh.n_cells);
    auto& gcell_to_local = result.gcell_to_local;
    for (int li = 0; li < clean_mesh.n_cells; ++li) {
        int gi = owned_global[li];
        gcell_to_local[gi] = li;
        auto& cell = clean_mesh.cells[li];
        cell.center = global_mesh.cells[gi].center;
        cell.volume = global_mesh.cells[gi].volume;
        for (int nid : global_mesh.cells[gi].nodes) {
            cell.nodes.push_back(node_map[nid]);
        }
    }

    // Helper to add a face to clean_mesh and track mpi_interface ghost info.
    // global_face_idx: the index in global_mesh.faces this came from.
    // remote_global_cell: the global cell ID on the other side (if mpi_interface).
    // remote_rank_id: the rank owning the remote cell (if mpi_interface).
    auto add_face = [&](const Face& gf, int local_owner, int local_neighbour,
                        Vec3 normal, const std::string& boundary_tag,
                        int remote_global_cell, int remote_rank_id) {
        int fid = static_cast<int>(clean_mesh.faces.size());
        Face lf;
        lf.owner = local_owner;
        lf.neighbour = local_neighbour;
        lf.area = gf.area;
        lf.normal = normal;
        lf.center = gf.center;
        lf.boundary_tag = boundary_tag;
        for (int nid : gf.nodes) {
            auto nit = node_map.find(nid);
            if (nit != node_map.end()) lf.nodes.push_back(nit->second);
        }
        clean_mesh.cells[local_owner].faces.push_back(fid);
        if (local_neighbour >= 0) {
            clean_mesh.cells[local_neighbour].faces.push_back(fid);
        }
        clean_mesh.faces.push_back(lf);

        // If this is an mpi_interface face, record ghost mapping.
        // patch_local_idx will be set later after we build boundary_patches.
        if (boundary_tag == "mpi_interface") {
            MPIGhostFace gf_info;
            gf_info.local_face_id = fid;
            gf_info.local_owner_cell = local_owner;
            gf_info.remote_rank = remote_rank_id;
            gf_info.remote_global_cell = remote_global_cell;
            gf_info.patch_local_idx = -1;  // set later
            result.ghost_faces.push_back(gf_info);
        }
    };

    // Pass 1: faces where owner is owned
    for (int fi = 0; fi < global_mesh.n_faces; ++fi) {
        const Face& gf = global_mesh.faces[fi];
        auto it_o = gcell_to_local.find(gf.owner);
        if (it_o == gcell_to_local.end()) continue;

        auto it_n = gcell_to_local.find(gf.neighbour);
        if (gf.neighbour >= 0 && it_n != gcell_to_local.end()) {
            // Internal face (both cells owned)
            add_face(gf, it_o->second, it_n->second, gf.normal,
                     gf.boundary_tag, -1, -1);
        } else if (gf.neighbour >= 0 && it_n == gcell_to_local.end()) {
            // Cross-partition face: neighbour not owned -> mpi_interface
            int remote_rank = part_ids[gf.neighbour];
            add_face(gf, it_o->second, -1, gf.normal,
                     "mpi_interface", gf.neighbour, remote_rank);
        } else {
            // Original boundary face (neighbour == -1)
            add_face(gf, it_o->second, -1, gf.normal,
                     gf.boundary_tag, -1, -1);
        }
    }

    // Pass 2: faces where neighbour is owned but owner is not
    for (int fi = 0; fi < global_mesh.n_faces; ++fi) {
        const Face& gf = global_mesh.faces[fi];
        if (gf.neighbour < 0) continue;
        auto it_o = gcell_to_local.find(gf.owner);
        auto it_n = gcell_to_local.find(gf.neighbour);
        if (it_o == gcell_to_local.end() && it_n != gcell_to_local.end()) {
            // Flip normal since we are making the neighbour the owner
            Vec3 flipped_normal(-gf.normal[0], -gf.normal[1], -gf.normal[2]);
            int remote_rank = part_ids[gf.owner];
            add_face(gf, it_n->second, -1, flipped_normal,
                     "mpi_interface", gf.owner, remote_rank);
        }
    }

    // Finalize mesh counts
    clean_mesh.n_faces = static_cast<int>(clean_mesh.faces.size());
    clean_mesh.n_internal_faces = 0;
    clean_mesh.n_boundary_faces = 0;
    for (auto& f : clean_mesh.faces) {
        if (f.neighbour >= 0) clean_mesh.n_internal_faces++;
        else clean_mesh.n_boundary_faces++;
    }

    // Build boundary patches from face tags
    for (int fi = 0; fi < clean_mesh.n_faces; ++fi) {
        auto& f = clean_mesh.faces[fi];
        if (f.neighbour < 0 && !f.boundary_tag.empty()) {
            clean_mesh.boundary_patches[f.boundary_tag].push_back(fi);
        }
    }
    clean_mesh.build_boundary_face_cache();

    // Now set patch_local_idx on each ghost face.
    // For each mpi_interface face in ghost_faces, find its index within the
    // "mpi_interface" boundary patch.
    if (clean_mesh.boundary_patches.count("mpi_interface")) {
        const auto& mpi_fids = clean_mesh.boundary_patches["mpi_interface"];
        // Build reverse lookup: face_id -> patch local index
        std::unordered_map<int, int> face_to_patch_idx;
        for (int i = 0; i < static_cast<int>(mpi_fids.size()); ++i) {
            face_to_patch_idx[mpi_fids[i]] = i;
        }
        for (auto& gf : result.ghost_faces) {
            auto it = face_to_patch_idx.find(gf.local_face_id);
            if (it != face_to_patch_idx.end()) {
                gf.patch_local_idx = it->second;
            }
        }
    }

    return result;
}

// ---------------------------------------------------------------------------
// Build the per-neighbor exchange structures using MPI communication.
//
// Each rank has a list of MPIGhostFace entries saying "I need the value of
// global cell X from rank R". We need to tell rank R which of its local
// cells to pack. We do this by:
//   1. Each rank groups its ghost faces by remote_rank.
//   2. For each neighbor, exchange the list of global cell IDs needed.
//   3. Each rank looks up the local cell index for each requested global cell
//      and builds a send_cells list in the order the remote expects.
// ---------------------------------------------------------------------------
static std::vector<NeighborExchange> build_exchange_maps(
    MPIComm& comm,
    const std::vector<MPIGhostFace>& ghost_faces,
    const std::unordered_map<int, int>& gcell_to_local)
{
    // Group ghost faces by remote rank
    std::map<int, std::vector<const MPIGhostFace*>> by_rank;
    for (const auto& gf : ghost_faces) {
        by_rank[gf.remote_rank].push_back(&gf);
    }

    std::vector<NeighborExchange> exchanges;

#ifdef USE_MPI
    for (auto& [remote_rank, faces] : by_rank) {
        NeighborExchange ex;
        ex.remote_rank = remote_rank;

        int n_need = static_cast<int>(faces.size());

        // Pack the global cell IDs we need from this neighbor
        std::vector<int> need_global_cells(n_need);
        ex.recv_patch_indices.resize(n_need);
        ex.recv_owner_cells.resize(n_need);
        for (int i = 0; i < n_need; ++i) {
            need_global_cells[i] = faces[i]->remote_global_cell;
            ex.recv_patch_indices[i] = faces[i]->patch_local_idx;
            ex.recv_owner_cells[i] = faces[i]->local_owner_cell;
        }

        // Exchange counts first
        int n_they_need = 0;
        MPI_Sendrecv(&n_need, 1, MPI_INT, remote_rank, 400,
                     &n_they_need, 1, MPI_INT, remote_rank, 400,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Exchange the global cell ID lists
        std::vector<int> they_need_global_cells(n_they_need);
        MPI_Sendrecv(need_global_cells.data(), n_need, MPI_INT, remote_rank, 401,
                     they_need_global_cells.data(), n_they_need, MPI_INT, remote_rank, 401,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Build send_cells: for each global cell the neighbor needs, look up
        // the local cell index on THIS rank.
        ex.send_cells.resize(n_they_need);
        for (int i = 0; i < n_they_need; ++i) {
            auto it = gcell_to_local.find(they_need_global_cells[i]);
            if (it != gcell_to_local.end()) {
                ex.send_cells[i] = it->second;
            } else {
                // This should not happen if partitioning is correct
                ex.send_cells[i] = 0;
            }
        }

        exchanges.push_back(std::move(ex));
    }
#else
    (void)comm; (void)ghost_faces; (void)gcell_to_local;
#endif

    return exchanges;
}

// ---------------------------------------------------------------------------
// Perform the actual ghost value exchange and update mpi_interface BCs.
//
// For each neighbor:
//   1. Pack this rank's cell values (u,v,w,p) at send_cells
//   2. MPI_Sendrecv with the neighbor
//   3. Unpack received values into the velocity and pressure boundary_values
//      at the mpi_interface patch indices
// ---------------------------------------------------------------------------
static void exchange_interface_values(
    MPIComm& comm,
    SIMPLESolver& solver,
    const std::vector<NeighborExchange>& exchanges)
{
#ifdef USE_MPI
    for (const auto& ex : exchanges) {
        const int n_send = static_cast<int>(ex.send_cells.size());
        const int n_recv = static_cast<int>(ex.recv_patch_indices.size());

        // Pack send buffer: 4 doubles per cell (u, v, w, p)
        std::vector<double> send_buf(n_send * 4);
        for (int i = 0; i < n_send; ++i) {
            int ci = ex.send_cells[i];
            send_buf[i * 4 + 0] = solver.velocity().values(ci, 0);
            send_buf[i * 4 + 1] = solver.velocity().values(ci, 1);
            send_buf[i * 4 + 2] = solver.velocity().values(ci, 2);
            send_buf[i * 4 + 3] = solver.pressure().values(ci);
        }

        std::vector<double> recv_buf(n_recv * 4);

        // Use rank ordering to avoid deadlock: lower rank sends first
        if (comm.rank() < ex.remote_rank) {
            MPI_Send(send_buf.data(), n_send * 4, MPI_DOUBLE,
                     ex.remote_rank, 500, MPI_COMM_WORLD);
            MPI_Recv(recv_buf.data(), n_recv * 4, MPI_DOUBLE,
                     ex.remote_rank, 500, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            MPI_Recv(recv_buf.data(), n_recv * 4, MPI_DOUBLE,
                     ex.remote_rank, 500, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(send_buf.data(), n_send * 4, MPI_DOUBLE,
                     ex.remote_rank, 500, MPI_COMM_WORLD);
        }

        // Unpack into mpi_interface boundary_values with under-relaxation.
        // Velocity BC is Dirichlet (set_inlet), so the solver uses these
        // boundary_values as the face value in the momentum equation.
        // Pressure BC is zero-gradient (set_inlet gives bc_p_=zero_gradient),
        // so the solver uses the owner cell value -- we do NOT update pressure
        // boundary_values to avoid destabilizing the pressure equation.
        auto u_bv_it = solver.velocity().boundary_values.find("mpi_interface");
        const double relax = 0.3;

        for (int i = 0; i < n_recv; ++i) {
            int pi = ex.recv_patch_indices[i];
            if (pi < 0) continue;

            double recv_u = recv_buf[i * 4 + 0];
            double recv_v = recv_buf[i * 4 + 1];
            double recv_w = recv_buf[i * 4 + 2];

            if (u_bv_it != solver.velocity().boundary_values.end()) {
                auto& bv = u_bv_it->second;
                bv(pi, 0) = (1.0 - relax) * bv(pi, 0) + relax * recv_u;
                bv(pi, 1) = (1.0 - relax) * bv(pi, 1) + relax * recv_v;
                bv(pi, 2) = (1.0 - relax) * bv(pi, 2) + relax * recv_w;
            }
        }
    }
#else
    (void)comm; (void)solver; (void)exchanges;
#endif
}

// ===================================================================
// MAIN
// ===================================================================

int main(int argc, char* argv[]) {
    // Parse arguments
    int nx = (argc > 1) ? std::atoi(argv[1]) : 40;
    int ny = (argc > 2) ? std::atoi(argv[2]) : 20;
    int nz = (argc > 3) ? std::atoi(argv[3]) : 20;
    int max_iter = (argc > 4) ? std::atoi(argv[4]) : 50;

    MPIComm comm;
    int n_cells_total = nx * ny * nz;

    if (comm.is_root()) {
        std::cout << "================================================================\n";
        std::cout << "  K-CFD MPI Performance Benchmark (Domain Decomposition)\n";
        std::cout << "================================================================\n";
        std::cout << "  MPI ranks:    " << comm.size() << "\n";
        std::cout << "  Mesh:         " << nx << " x " << ny << " x " << nz
                  << " = " << n_cells_total << " cells\n";
        std::cout << "  Max iter:     " << max_iter << "\n";
        std::cout << "  Est. memory:  ~" << (n_cells_total * 300) / (1024*1024) << " MB\n";
        std::cout << "================================================================\n\n";
    }

    // ============================================================
    // Phase 1: Mesh generation (all ranks generate the same mesh)
    // ============================================================
    auto t_mesh_start = std::chrono::high_resolution_clock::now();

    double Lx = 10.0, Ly = 1.0, Lz = 1.0;
    double Re = 100.0;
    double U_mean = 1.0;
    double rho = 1.0;
    double mu = rho * U_mean * Ly / Re;

    FVMesh mesh(3);
    if (comm.is_root()) {
        std::cout << "Generating 3D mesh..." << std::flush;
    }
    mesh = generate_3d_channel_mesh(Lx, Ly, Lz, nx, ny, nz);

    auto t_mesh_end = std::chrono::high_resolution_clock::now();
    double mesh_time = std::chrono::duration<double>(t_mesh_end - t_mesh_start).count();

    if (comm.is_root()) {
        std::cout << " done (" << std::fixed << std::setprecision(2)
                  << mesh_time << " s)\n";
        std::cout << "  Cells: " << mesh.n_cells
                  << "  Faces: " << mesh.n_faces
                  << " (" << mesh.n_internal_faces << " internal)\n";

        auto quality = mesh.compute_quality();
        std::cout << "  Quality: non-orth=" << std::setprecision(1)
                  << quality.max_non_orthogonality << "\xC2\xB0"
                  << "  skewness=" << std::setprecision(3)
                  << quality.max_skewness
                  << "  aspect=" << std::setprecision(1)
                  << quality.max_aspect_ratio << "\n\n";
    }

    // ============================================================
    // Phase 2: Serial solve (baseline, only on root)
    // ============================================================
    double serial_solve_time = 0.0;
    double serial_final_res = 1.0;
    SolveResult serial_result;
    serial_result.converged = false;
    serial_result.iterations = 0;

    if (comm.is_root()) {
        std::cout << "--- SERIAL BASELINE ---\n";
        std::cout << "Solving full mesh (" << max_iter << " iterations)..." << std::flush;

        auto t_serial_start = std::chrono::high_resolution_clock::now();

        SIMPLESolver serial_solver(mesh, rho, mu);
        serial_solver.max_iter = max_iter;
        serial_solver.tol = 1e-8;  // tight tol so we run all iterations
        serial_solver.alpha_u = 0.7;
        serial_solver.alpha_p = 0.3;
        serial_solver.linear_solver_type = "bicgstab";

        // Set BCs
        int n_inlet = static_cast<int>(mesh.boundary_patches["inlet"].size());
        Eigen::MatrixXd inlet_U(n_inlet, 3);
        for (int j = 0; j < n_inlet; ++j) {
            int fid = mesh.boundary_patches["inlet"][j];
            double y = mesh.faces[fid].center[1];
            double z = mesh.faces[fid].center[2];
            double u_prof = 2.25 * U_mean * 4.0 * (y/Ly) * (1.0 - y/Ly)
                            * 4.0 * (z/Lz) * (1.0 - z/Lz);
            inlet_U(j, 0) = u_prof;
            inlet_U(j, 1) = 0.0;
            inlet_U(j, 2) = 0.0;
        }
        serial_solver.set_inlet("inlet", inlet_U);
        serial_solver.set_outlet("outlet", 0.0);
        serial_solver.set_wall("wall_bottom");
        serial_solver.set_wall("wall_top");
        serial_solver.set_wall("wall_front");
        serial_solver.set_wall("wall_back");

        serial_result = serial_solver.solve_steady();

        auto t_serial_end = std::chrono::high_resolution_clock::now();
        serial_solve_time = std::chrono::duration<double>(t_serial_end - t_serial_start).count();
        serial_final_res = serial_result.residuals.empty() ? 1.0 : serial_result.residuals.back();

        std::cout << " done\n";
        std::cout << "  Converged:      " << (serial_result.converged ? "yes" : "no") << "\n";
        std::cout << "  Iterations:     " << serial_result.iterations << "\n";
        std::cout << "  Final residual: " << std::scientific << std::setprecision(3)
                  << serial_final_res << "\n";

        double u_max = serial_solver.velocity().magnitude().maxCoeff();
        double p_range = serial_solver.pressure().max() - serial_solver.pressure().min();
        std::cout << "  |U| max:        " << std::fixed << std::setprecision(4) << u_max << "\n";
        std::cout << "  p range:        " << std::setprecision(4) << p_range << "\n";
        std::cout << "  Serial time:    " << std::setprecision(3) << serial_solve_time << " s\n";
        std::cout << "  Time/iter:      " << std::setprecision(4)
                  << serial_solve_time / std::max(1, serial_result.iterations) << " s\n\n";
    }

    // Broadcast serial_solve_time and final residual to all ranks
#ifdef USE_MPI
    MPI_Bcast(&serial_solve_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&serial_final_res, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif

    // ============================================================
    // Phase 3: Parallel domain-decomposed solve
    // ============================================================
    if (comm.is_parallel()) {
        comm.barrier();

        if (comm.is_root()) {
            std::cout << "--- PARALLEL DOMAIN DECOMPOSITION (" << comm.size() << " ranks) ---\n";
        }

        // ----------------------------------------------------------
        // 3a. Partition on root, broadcast to all ranks
        // ----------------------------------------------------------
        auto t_part_start = std::chrono::high_resolution_clock::now();

        std::vector<int> part_ids(mesh.n_cells, 0);
        if (comm.is_root()) {
            std::cout << "Partitioning into " << comm.size() << " domains..." << std::flush;
            part_ids = RCBPartitioner::partition(mesh, comm.size());
        }

#ifdef USE_MPI
        MPI_Bcast(part_ids.data(), mesh.n_cells, MPI_INT, 0, MPI_COMM_WORLD);
#endif

        auto t_part_end = std::chrono::high_resolution_clock::now();
        double part_time = std::chrono::duration<double>(t_part_end - t_part_start).count();

        if (comm.is_root()) {
            std::vector<int> counts(comm.size(), 0);
            for (int pid : part_ids) counts[pid]++;
            std::cout << " done (" << std::setprecision(3) << part_time << " s)\n";
            std::cout << "  Cells per rank: ";
            for (int r = 0; r < comm.size(); ++r) {
                std::cout << counts[r];
                if (r < comm.size() - 1) std::cout << " / ";
            }
            std::cout << "\n";
        }

        // ----------------------------------------------------------
        // 3b. Build clean local mesh with ghost face mapping
        // ----------------------------------------------------------
        auto t_extract_start = std::chrono::high_resolution_clock::now();

        CleanMeshResult cm_result = build_clean_mesh(mesh, part_ids, comm.rank());
        FVMesh& clean_mesh = cm_result.mesh;

        auto t_extract_end = std::chrono::high_resolution_clock::now();
        double extract_time = std::chrono::duration<double>(t_extract_end - t_extract_start).count();

        if (comm.is_root()) {
            std::cout << "  Extract time:   " << std::setprecision(3) << extract_time << " s\n";
        }

        // Print per-rank info
        for (int r = 0; r < comm.size(); ++r) {
            if (r == comm.rank()) {
                std::cout << "  Rank " << r << ": " << clean_mesh.n_cells << " cells, "
                          << clean_mesh.n_internal_faces << " int, "
                          << clean_mesh.n_boundary_faces << " bnd, patches:";
                for (auto& [name, fids] : clean_mesh.boundary_patches) {
                    std::cout << " " << name << "(" << fids.size() << ")";
                }
                std::cout << ", ghost_faces=" << cm_result.ghost_faces.size()
                          << "\n" << std::flush;
            }
            comm.barrier();
        }

        // ----------------------------------------------------------
        // 3c. Build exchange maps (MPI communication of index lists)
        // ----------------------------------------------------------
        auto exchanges = build_exchange_maps(
            comm, cm_result.ghost_faces, cm_result.gcell_to_local);

        if (comm.is_root()) {
            int total_exchange_faces = 0;
            for (const auto& ex : exchanges) {
                total_exchange_faces += static_cast<int>(ex.recv_patch_indices.size());
            }
            std::cout << "  Exchange setup: " << exchanges.size() << " neighbors, "
                      << total_exchange_faces << " interface faces\n";
        }

        // ----------------------------------------------------------
        // 3d. Create local solver and set BCs
        // ----------------------------------------------------------
        SIMPLESolver local_solver(clean_mesh, rho, mu);
        local_solver.alpha_u = 0.7;
        local_solver.alpha_p = 0.3;
        local_solver.linear_solver_type = "bicgstab";

        set_local_bcs(local_solver, clean_mesh, Ly, Lz, U_mean);

        // ----------------------------------------------------------
        // 3e. Outer Schwarz iteration loop with REAL ghost exchange
        // ----------------------------------------------------------
        const int inner_iters = 5;   // SIMPLE iterations per exchange
        const int outer_iters = (max_iter + inner_iters - 1) / inner_iters;
        const double tol = 1e-8;

        comm.barrier();

        if (comm.is_root()) {
            std::cout << "\nSolving: " << outer_iters << " outer x "
                      << inner_iters << " inner iterations...\n" << std::flush;
        }

        auto t_solve_start = std::chrono::high_resolution_clock::now();
        int total_iters = 0;
        bool converged = false;
        double final_residual = 1.0;

        for (int outer = 0; outer < outer_iters; ++outer) {
            // Exchange ghost values BEFORE inner solve so that the
            // mpi_interface Dirichlet BCs have current neighbor data.
            exchange_interface_values(comm, local_solver, exchanges);

            // Inner SIMPLE iterations on local mesh
            local_solver.max_iter = inner_iters;
            local_solver.tol = tol * 0.1;  // tight inner tol
            SolveResult inner_result = local_solver.solve_steady();
            total_iters += inner_result.iterations;

            // Global convergence check
            double local_res = inner_result.residuals.empty()
                             ? 0.0 : inner_result.residuals.back();
            double global_res = comm.all_reduce_max(local_res);
            final_residual = global_res;

            if (comm.is_root() && (outer % 5 == 0 || outer == outer_iters - 1)) {
                std::cout << "  Outer " << std::setw(3) << outer + 1 << "/"
                          << outer_iters << "  inner_iters=" << inner_result.iterations
                          << "  global_res=" << std::scientific << std::setprecision(3)
                          << global_res << std::fixed << "\n" << std::flush;
            }

            if (global_res < tol) {
                converged = true;
                break;
            }
        }

        auto t_solve_end = std::chrono::high_resolution_clock::now();
        double parallel_solve_time = std::chrono::duration<double>(
            t_solve_end - t_solve_start).count();

        comm.barrier();

        // ----------------------------------------------------------
        // 3f. Report results
        // ----------------------------------------------------------
        double u_max_local = local_solver.velocity().magnitude().maxCoeff();
        double u_max_global = comm.all_reduce_max(u_max_local);

        if (comm.is_root()) {
            std::cout << "\n================================================================\n";
            std::cout << "  RESULTS\n";
            std::cout << "================================================================\n";
            std::cout << "  Converged:        " << (converged ? "yes" : "no") << "\n";
            std::cout << "  Total iterations: " << total_iters << "\n";
            std::cout << "  Final residual:   " << std::scientific << std::setprecision(3)
                      << final_residual << "\n";
            std::cout << "  |U| max (global): " << std::fixed << std::setprecision(4)
                      << u_max_global << "\n";
            std::cout << "  Serial final res: " << std::scientific << std::setprecision(3)
                      << serial_final_res << "\n\n";

            std::cout << "================================================================\n";
            std::cout << "  PERFORMANCE\n";
            std::cout << "================================================================\n";
            std::cout << "  Mesh generation:  " << std::fixed << std::setprecision(3)
                      << mesh_time << " s\n";
            std::cout << "  Partitioning:     " << part_time << " s\n";
            std::cout << "  Mesh extraction:  " << extract_time << " s\n";
            std::cout << "  Serial time:      " << serial_solve_time << " s"
                      << " (" << max_iter << " iters on " << n_cells_total << " cells)\n";
            std::cout << "  Parallel time:    " << parallel_solve_time << " s"
                      << " (" << total_iters << " total iters, "
                      << comm.size() << " ranks)\n";

            double speedup = (serial_solve_time > 0.0)
                           ? serial_solve_time / parallel_solve_time : 0.0;
            double efficiency = speedup / comm.size() * 100.0;

            std::cout << "  Speedup:          " << std::setprecision(2) << speedup << "x\n";
            std::cout << "  Efficiency:       " << std::setprecision(1) << efficiency << "%\n";

            double serial_per_iter = serial_solve_time / std::max(1, max_iter);
            double parallel_per_iter = parallel_solve_time / std::max(1, total_iters);
            std::cout << "  Serial time/iter: " << std::setprecision(4)
                      << serial_per_iter << " s\n";
            std::cout << "  Par. time/iter:   " << parallel_per_iter << " s\n";

            double cells_per_sec = static_cast<double>(n_cells_total) * total_iters
                                 / parallel_solve_time;
            std::cout << "  Throughput:       " << std::scientific << std::setprecision(2)
                      << cells_per_sec << " cell-iters/s\n";

            std::cout << "================================================================\n";

            // Scaling model
            double n_ranks = static_cast<double>(comm.size());
            double expected = std::pow(n_ranks, 0.3);
            std::cout << "\n  Theoretical speedup (O(n^1.3) solver): "
                      << std::fixed << std::setprecision(2) << expected << "x\n";
            std::cout << "  Actual speedup:                         "
                      << speedup << "x\n";
        }
    } else {
        // ============================================================
        // Single rank: serial only (no parallel comparison)
        // ============================================================
        if (comm.is_root()) {
            std::cout << "================================================================\n";
            std::cout << "  RESULTS (serial only, run with mpiexec -n N for parallel)\n";
            std::cout << "================================================================\n";
            std::cout << "  Converged:      " << (serial_result.converged ? "yes" : "no") << "\n";
            std::cout << "  Iterations:     " << serial_result.iterations << "\n";
            std::cout << "  Final residual: " << std::scientific << std::setprecision(3);
            if (!serial_result.residuals.empty()) std::cout << serial_result.residuals.back();
            else std::cout << "N/A";
            std::cout << "\n";
            std::cout << "  Solve time:     " << std::fixed << std::setprecision(3)
                      << serial_solve_time << " s\n";

            double cells_per_sec = static_cast<double>(n_cells_total)
                                 * serial_result.iterations / serial_solve_time;
            std::cout << "  Time/iteration: " << std::setprecision(4)
                      << serial_solve_time / std::max(1, serial_result.iterations) << " s\n";
            std::cout << "  Throughput:     " << std::scientific << std::setprecision(2)
                      << cells_per_sec << " cell-iters/s\n";
            std::cout << "================================================================\n";

            std::cout << "\n  Projected scaling:\n";
            for (int np : {2, 4, 8, 14}) {
                double projected = serial_solve_time / std::pow(np, 0.3);
                std::cout << "    " << np << " ranks: ~"
                          << std::fixed << std::setprecision(2) << projected << " s "
                          << "(speedup ~" << std::setprecision(1)
                          << std::pow(np, 0.3) << "x)\n";
            }
        }
    }

    return 0;
}
