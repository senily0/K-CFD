#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "twofluid/mesh.hpp"
#include "twofluid/fields.hpp"
#include "twofluid/gradient.hpp"
#include "twofluid/fvm_operators.hpp"
#include "twofluid/linear_solver.hpp"
#include "twofluid/interpolation.hpp"
#include "twofluid/simple_solver.hpp"
#include "twofluid/two_fluid_solver.hpp"
#include "twofluid/closure.hpp"
#include "twofluid/vtk_writer.hpp"
#include "twofluid/mesh_generator.hpp"
#include "twofluid/preconditioner.hpp"
#include "twofluid/time_control.hpp"
#include "twofluid/phase_change.hpp"
#include "twofluid/solid_conduction.hpp"
#include "twofluid/conjugate_ht.hpp"
#include "twofluid/radiation.hpp"
#include "twofluid/amr.hpp"
#include "twofluid/mesh_generator_3d.hpp"

namespace py = pybind11;
using namespace twofluid;

PYBIND11_MODULE(twofluid_cpp, m) {
    m.doc() = "Two-Fluid FVM solver C++ core";

    // ---- Face ----
    py::class_<Face>(m, "Face")
        .def(py::init<>())
        .def_readwrite("owner", &Face::owner)
        .def_readwrite("neighbour", &Face::neighbour)
        .def_readwrite("area", &Face::area)
        .def_readwrite("normal", &Face::normal)
        .def_readwrite("center", &Face::center)
        .def_readwrite("nodes", &Face::nodes)
        .def_readwrite("boundary_tag", &Face::boundary_tag);

    // ---- Cell ----
    py::class_<Cell>(m, "Cell")
        .def(py::init<>())
        .def_readwrite("volume", &Cell::volume)
        .def_readwrite("center", &Cell::center)
        .def_readwrite("nodes", &Cell::nodes)
        .def_readwrite("faces", &Cell::faces);

    // ---- FVMesh ----
    py::class_<FVMesh>(m, "FVMesh")
        .def(py::init<int>(), py::arg("ndim") = 2)
        .def_readwrite("ndim", &FVMesh::ndim)
        .def_readwrite("n_cells", &FVMesh::n_cells)
        .def_readwrite("n_faces", &FVMesh::n_faces)
        .def_readwrite("n_internal_faces", &FVMesh::n_internal_faces)
        .def_readwrite("n_boundary_faces", &FVMesh::n_boundary_faces)
        .def_readwrite("nodes", &FVMesh::nodes)
        .def_readwrite("boundary_patches", &FVMesh::boundary_patches)
        .def_readwrite("cell_zones", &FVMesh::cell_zones)
        // Expose faces/cells via methods to avoid pybind11 stl copy issue
        .def("add_face", [](FVMesh& self, const Face& f) { self.faces.push_back(f); })
        .def("add_cell", [](FVMesh& self, const Cell& c) { self.cells.push_back(c); })
        .def("num_faces", [](const FVMesh& self) -> int { return (int)self.faces.size(); })
        .def("num_cells", [](const FVMesh& self) -> int { return (int)self.cells.size(); })
        .def("get_face", [](const FVMesh& self, int i) -> const Face& { return self.faces[i]; },
             py::return_value_policy::reference_internal)
        .def("get_cell", [](const FVMesh& self, int i) -> const Cell& { return self.cells[i]; },
             py::return_value_policy::reference_internal)
        .def("summary", &FVMesh::summary);

    // ---- ScalarField ----
    py::class_<ScalarField>(m, "ScalarField")
        .def(py::init<const FVMesh&, const std::string&, double>(),
             py::arg("mesh"), py::arg("name") = "scalar",
             py::arg("default_val") = 0.0)
        .def_readwrite("values", &ScalarField::values)
        .def_readwrite("old_values", &ScalarField::old_values)
        .def_readwrite("boundary_values", &ScalarField::boundary_values)
        .def("set_uniform", &ScalarField::set_uniform)
        .def("store_old", &ScalarField::store_old)
        .def("set_boundary",
             py::overload_cast<const std::string&, double>(
                 &ScalarField::set_boundary))
        .def("set_boundary",
             py::overload_cast<const std::string&, const Eigen::VectorXd&>(
                 &ScalarField::set_boundary))
        .def("get_face_value", &ScalarField::get_face_value)
        .def("copy", &ScalarField::copy)
        .def("max", &ScalarField::max)
        .def("min", &ScalarField::min)
        .def("mean", &ScalarField::mean)
        .def_property_readonly("name", &ScalarField::name)
        .def_property_readonly("mesh", &ScalarField::mesh,
                               py::return_value_policy::reference);

    // ---- VectorField ----
    py::class_<VectorField>(m, "VectorField")
        .def(py::init<const FVMesh&, const std::string&>(),
             py::arg("mesh"), py::arg("name") = "vector")
        .def_readwrite("values", &VectorField::values)
        .def_readwrite("old_values", &VectorField::old_values)
        .def_readwrite("boundary_values", &VectorField::boundary_values)
        .def("set_uniform", &VectorField::set_uniform)
        .def("store_old", &VectorField::store_old)
        .def("set_boundary",
             py::overload_cast<const std::string&, const Eigen::VectorXd&>(
                 &VectorField::set_boundary))
        .def("set_boundary",
             py::overload_cast<const std::string&, const Eigen::MatrixXd&>(
                 &VectorField::set_boundary))
        .def("copy", &VectorField::copy)
        .def("magnitude", &VectorField::magnitude)
        .def_property_readonly("name", &VectorField::name)
        .def_property_readonly("mesh", &VectorField::mesh,
                               py::return_value_policy::reference);

    // ---- FVMSystem ----
    py::class_<FVMSystem>(m, "FVMSystem")
        .def(py::init<int>())
        .def_readonly("n", &FVMSystem::n)
        .def_readwrite("rhs", &FVMSystem::rhs)
        .def_readwrite("diag", &FVMSystem::diag)
        .def("add_diagonal", &FVMSystem::add_diagonal)
        .def("add_off_diagonal", &FVMSystem::add_off_diagonal)
        .def("add_source", &FVMSystem::add_source)
        .def("reset", &FVMSystem::reset);

    // ---- Gradient functions ----
    m.def("green_gauss_gradient", &green_gauss_gradient,
          "Green-Gauss gradient reconstruction");
    m.def("least_squares_gradient", &least_squares_gradient,
          "Least Squares gradient reconstruction");

    // ---- Operator functions ----
    m.def("diffusion_operator", &diffusion_operator);
    m.def("convection_operator_upwind", &convection_operator_upwind);
    m.def("temporal_operator", &temporal_operator);
    m.def("source_term", &source_term);
    m.def("linearized_source", &linearized_source);
    m.def("under_relax", &under_relax);

    // ---- Linear solver ----
    m.def("solve_linear_system", &solve_linear_system,
          py::arg("system"),
          py::arg("x0") = Eigen::VectorXd(),
          py::arg("method") = "direct",
          py::arg("tol") = 1e-6,
          py::arg("maxiter") = 1000,
          "Solve the linear system Ax=b from an FVMSystem");

    // ---- Interpolation ----
    m.def("compute_mass_flux", &compute_mass_flux,
          "Compute face mass flux from velocity field");
    m.def("limiter_van_leer", &limiter_van_leer);
    m.def("limiter_minmod", &limiter_minmod);
    m.def("limiter_superbee", &limiter_superbee);
    m.def("limiter_van_albada", &limiter_van_albada);
    m.def("muscl_deferred_correction", &muscl_deferred_correction,
          py::arg("mesh"), py::arg("phi"), py::arg("mass_flux"),
          py::arg("grad_phi"), py::arg("limiter_name") = "van_leer",
          "MUSCL deferred correction source term");

    // ---- SolveResult ----
    py::class_<SolveResult>(m, "SolveResult")
        .def(py::init<>())
        .def_readwrite("converged", &SolveResult::converged)
        .def_readwrite("iterations", &SolveResult::iterations)
        .def_readwrite("residuals", &SolveResult::residuals)
        .def_readwrite("wall_time", &SolveResult::wall_time);

    // ---- SIMPLESolver ----
    py::class_<SIMPLESolver>(m, "SIMPLESolver")
        .def(py::init<FVMesh&, double, double>(),
             py::keep_alive<1, 2>())  // keep mesh alive
        .def_readwrite("alpha_u", &SIMPLESolver::alpha_u)
        .def_readwrite("alpha_p", &SIMPLESolver::alpha_p)
        .def_readwrite("max_iter", &SIMPLESolver::max_iter)
        .def_readwrite("tol", &SIMPLESolver::tol)
        .def("set_inlet", &SIMPLESolver::set_inlet)
        .def("set_wall", &SIMPLESolver::set_wall)
        .def("set_outlet", &SIMPLESolver::set_outlet,
             py::arg("patch"), py::arg("p_val") = 0.0)
        .def("solve_steady", &SIMPLESolver::solve_steady)
        .def("velocity", &SIMPLESolver::velocity,
             py::return_value_policy::reference_internal)
        .def("pressure", &SIMPLESolver::pressure,
             py::return_value_policy::reference_internal);

    // ---- VTK Writer ----
    m.def("write_vtu", &write_vtu,
          py::arg("filename"),
          py::arg("mesh"),
          py::arg("cell_scalar_data") = std::unordered_map<std::string, Eigen::VectorXd>{},
          py::arg("cell_vector_data") = std::unordered_map<std::string, Eigen::MatrixXd>{},
          "Write mesh and field data to VTU format");

    // ---- Mesh Generators ----
    m.def("generate_channel_mesh", &generate_channel_mesh,
          py::arg("Lx"), py::arg("Ly"), py::arg("nx"), py::arg("ny"),
          "Generate a structured quad channel mesh");
    m.def("generate_cavity_mesh", &generate_cavity_mesh,
          py::arg("L"), py::arg("n"),
          "Generate a lid-driven cavity mesh");
    m.def("generate_bfs_mesh", &generate_bfs_mesh,
          py::arg("step_height") = 1.0,
          py::arg("expansion_ratio") = 2.0,
          py::arg("L_up") = 5.0,
          py::arg("L_down") = 30.0,
          py::arg("nx_up") = 50,
          py::arg("nx_down") = 250,
          py::arg("ny") = 80,
          "Generate a backward-facing step mesh");

    // ---- Closure models ----
    m.def("drag_coefficient_implicit", &drag_coefficient_implicit,
          py::arg("alpha_g"), py::arg("rho_l"),
          py::arg("U_g"), py::arg("U_l"),
          py::arg("d_b"), py::arg("mu_l"),
          "Implicit drag coefficient K_drag (Schiller-Naumann)");
    m.def("ranz_marshall_nusselt",
          &ranz_marshall_nusselt,
          py::arg("rho_l"), py::arg("mu_l"), py::arg("cp_l"), py::arg("k_l"),
          py::arg("U_g"), py::arg("U_l"), py::arg("d_b"),
          "Ranz-Marshall Nusselt number");
    m.def("interfacial_heat_transfer",
          &interfacial_heat_transfer,
          py::arg("alpha_g"), py::arg("rho_l"), py::arg("mu_l"),
          py::arg("cp_l"), py::arg("k_l"),
          py::arg("U_g"), py::arg("U_l"), py::arg("d_b"),
          "Interfacial heat transfer coefficients (h_i, a_i)");
    m.def("sato_bubble_induced_turbulence",
          &sato_bubble_induced_turbulence,
          py::arg("alpha_g"), py::arg("rho_l"),
          py::arg("U_g"), py::arg("U_l"), py::arg("d_b"),
          "Sato bubble-induced turbulence viscosity");

    // ---- Preconditioner ----
    py::class_<PreconditionerInfo>(m, "PreconditionerInfo")
        .def_readwrite("method", &PreconditionerInfo::method)
        .def_readwrite("setup_time", &PreconditionerInfo::setup_time);
    m.def("create_preconditioner", &create_preconditioner,
          py::arg("A"), py::arg("method") = "none");

    // ---- AdaptiveTimeControl ----
    py::class_<TimeStepInfo>(m, "TimeStepInfo")
        .def_readwrite("cfl_max", &TimeStepInfo::cfl_max)
        .def_readwrite("fourier_max", &TimeStepInfo::fourier_max)
        .def_readwrite("dt_limited_by", &TimeStepInfo::dt_limited_by)
        .def_readwrite("dt", &TimeStepInfo::dt);

    py::class_<AdaptiveTimeControl>(m, "AdaptiveTimeControl")
        .def(py::init<double, double, double, double, double, double, double, double, double>(),
             py::arg("dt_init"), py::arg("dt_min") = 1e-8,
             py::arg("dt_max") = 1.0, py::arg("cfl_target") = 0.5,
             py::arg("cfl_max") = 1.0, py::arg("fourier_max") = 0.5,
             py::arg("growth_factor") = 1.2, py::arg("shrink_factor") = 0.5,
             py::arg("safety_factor") = 0.9)
        .def("dt", &AdaptiveTimeControl::dt)
        .def("compute_cfl", &AdaptiveTimeControl::compute_cfl)
        .def("compute_fourier", &AdaptiveTimeControl::compute_fourier)
        .def("compute_dt", &AdaptiveTimeControl::compute_dt,
             py::arg("mesh"), py::arg("u_mag"),
             py::arg("alpha_diff") = -1.0, py::arg("converged") = true)
        .def_readwrite("dt_history", &AdaptiveTimeControl::dt_history)
        .def_readwrite("cfl_history", &AdaptiveTimeControl::cfl_history)
        .def_readwrite("fourier_history", &AdaptiveTimeControl::fourier_history);

    // ---- Phase Change Models ----
    m.def("saturation_temperature", &saturation_temperature);
    m.def("water_latent_heat", &water_latent_heat);

    py::class_<WaterProperties>(m, "WaterProperties")
        .def_readwrite("T_sat", &WaterProperties::T_sat)
        .def_readwrite("h_fg", &WaterProperties::h_fg)
        .def_readwrite("rho_l", &WaterProperties::rho_l)
        .def_readwrite("rho_g", &WaterProperties::rho_g)
        .def_readwrite("cp_l", &WaterProperties::cp_l)
        .def_readwrite("mu_l", &WaterProperties::mu_l)
        .def_readwrite("k_l", &WaterProperties::k_l);
    m.def("water_properties", &water_properties);

    py::class_<LeePhaseChangeModel>(m, "LeePhaseChangeModel")
        .def(py::init<const FVMesh&, double, double, double, double, double, double>(),
             py::arg("mesh"), py::arg("T_sat") = 373.15,
             py::arg("r_evap") = 0.1, py::arg("r_cond") = 0.1,
             py::arg("L_latent") = 2.26e6,
             py::arg("rho_l") = 1000.0, py::arg("rho_g") = 1.0)
        .def("compute_mass_transfer", &LeePhaseChangeModel::compute_mass_transfer)
        .def("get_source_terms", &LeePhaseChangeModel::get_source_terms);

    py::class_<RohsenowBoilingModel>(m, "RohsenowBoilingModel")
        .def(py::init<double, double, double, double, double, double, double, double, double, double, double>(),
             py::arg("T_sat"), py::arg("h_fg"),
             py::arg("rho_l"), py::arg("rho_g"),
             py::arg("mu_l"), py::arg("cp_l"),
             py::arg("sigma"), py::arg("Pr_l"),
             py::arg("C_sf") = 0.013, py::arg("n") = 1.0,
             py::arg("g") = 9.81)
        .def("compute_wall_heat_flux", &RohsenowBoilingModel::compute_wall_heat_flux)
        .def("compute_mass_transfer_wall", &RohsenowBoilingModel::compute_mass_transfer_wall);

    py::class_<CHFMargin>(m, "CHFMargin")
        .def_readwrite("chf", &CHFMargin::chf)
        .def_readwrite("ratio", &CHFMargin::ratio)
        .def_readwrite("safe", &CHFMargin::safe);

    py::class_<ZuberCHFModel>(m, "ZuberCHFModel")
        .def(py::init<double, double, double, double, double>(),
             py::arg("h_fg"), py::arg("rho_l"), py::arg("rho_g"),
             py::arg("sigma"), py::arg("g") = 9.81)
        .def("compute_chf", &ZuberCHFModel::compute_chf)
        .def("check_margin", &ZuberCHFModel::check_margin);

    py::class_<NusseltCondensationModel>(m, "NusseltCondensationModel")
        .def(py::init<double, double, double, double, double, double, double>(),
             py::arg("T_sat"), py::arg("h_fg"),
             py::arg("rho_l"), py::arg("rho_g"),
             py::arg("mu_l"), py::arg("k_l"),
             py::arg("g") = 9.81)
        .def("compute_heat_transfer_coeff", &NusseltCondensationModel::compute_heat_transfer_coeff)
        .def("compute_condensation_rate", &NusseltCondensationModel::compute_condensation_rate);

    // ---- Solid Conduction ----
    py::class_<SolidSolveResult>(m, "SolidSolveResult")
        .def_readwrite("converged", &SolidSolveResult::converged)
        .def_readwrite("iterations", &SolidSolveResult::iterations)
        .def_readwrite("T_max", &SolidSolveResult::T_max)
        .def_readwrite("T_min", &SolidSolveResult::T_min);

    py::class_<SolidConductionSolver>(m, "SolidConductionSolver")
        .def(py::init<FVMesh&, const std::vector<int>&>(),
             py::arg("mesh"), py::arg("cell_ids") = std::vector<int>{},
             py::keep_alive<1, 2>())
        .def_readwrite("rho", &SolidConductionSolver::rho)
        .def_readwrite("cp", &SolidConductionSolver::cp)
        .def_readwrite("k_s", &SolidConductionSolver::k_s)
        .def_readwrite("alpha_T", &SolidConductionSolver::alpha_T)
        .def("set_material", &SolidConductionSolver::set_material)
        .def("set_heat_source", &SolidConductionSolver::set_heat_source,
             py::arg("q"), py::arg("cells") = std::vector<int>{})
        .def("solve_steady", &SolidConductionSolver::solve_steady,
             py::arg("max_iter") = 200, py::arg("tol") = 1e-6)
        .def("solve_one_step", &SolidConductionSolver::solve_one_step,
             py::arg("dt") = -1.0);

    // ---- Conjugate Heat Transfer ----
    py::class_<CHTResult>(m, "CHTResult")
        .def_readwrite("converged", &CHTResult::converged)
        .def_readwrite("iterations", &CHTResult::iterations)
        .def_readwrite("T_interface_avg", &CHTResult::T_interface_avg)
        .def_readwrite("heat_flux", &CHTResult::heat_flux);

    py::class_<CHTCoupling>(m, "CHTCoupling")
        .def(py::init<FVMesh&, const std::vector<int>&, const std::vector<int>&>(),
             py::keep_alive<1, 2>())
        .def_readwrite("rho_f", &CHTCoupling::rho_f)
        .def_readwrite("cp_f", &CHTCoupling::cp_f)
        .def_readwrite("k_f", &CHTCoupling::k_f)
        .def_readwrite("max_cht_iter", &CHTCoupling::max_cht_iter)
        .def_readwrite("tol_cht", &CHTCoupling::tol_cht)
        .def_readwrite("alpha_cht", &CHTCoupling::alpha_cht)
        .def("solve_steady", &CHTCoupling::solve_steady);

    // ---- P1 Radiation Model ----
    py::class_<RadiationResult>(m, "RadiationResult")
        .def_readwrite("converged", &RadiationResult::converged)
        .def_readwrite("iterations", &RadiationResult::iterations)
        .def_readwrite("residuals", &RadiationResult::residuals);

    py::class_<P1RadiationModel>(m, "P1RadiationModel")
        .def(py::init<FVMesh&, double>(),
             py::arg("mesh"), py::arg("kappa") = 1.0,
             py::keep_alive<1, 2>())
        .def_readwrite("kappa", &P1RadiationModel::kappa)
        .def_readwrite("alpha_G", &P1RadiationModel::alpha_G)
        .def("solve", &P1RadiationModel::solve,
             py::arg("T"), py::arg("max_iter") = 100, py::arg("tol") = 1e-6)
        .def("compute_radiative_source", &P1RadiationModel::compute_radiative_source)
        .def("set_bc", &P1RadiationModel::set_bc,
             py::arg("patch"), py::arg("bc_type"), py::arg("T_wall") = 0.0);

    // ---- AMR ----
    py::class_<AMRCell>(m, "AMRCell")
        .def_readwrite("cell_id", &AMRCell::cell_id)
        .def_readwrite("level", &AMRCell::level)
        .def_readwrite("parent", &AMRCell::parent)
        .def_readwrite("children", &AMRCell::children)
        .def("is_leaf", &AMRCell::is_leaf);

    py::class_<AMRMesh>(m, "AMRMesh")
        .def(py::init<const FVMesh&, int>(),
             py::arg("base_mesh"), py::arg("max_level") = 3)
        .def("refine_cells", &AMRMesh::refine_cells)
        .def("get_active_cells", &AMRMesh::get_active_cells)
        .def("get_active_mesh", &AMRMesh::get_active_mesh)
        .def("transfer_field_to_children", &AMRMesh::transfer_field_to_children)
        .def("n_refinements", &AMRMesh::n_refinements);

    m.def("gradient_jump_estimate", &GradientJumpEstimator::estimate);

    py::class_<AMRSolverLoop>(m, "AMRSolverLoop")
        .def(py::init<AMRMesh&, double>(),
             py::arg("amr_mesh"), py::arg("refine_fraction") = 0.3,
             py::keep_alive<1, 2>())
        .def("mark_cells", &AMRSolverLoop::mark_cells);

    // ---- 3D Mesh Generators ----
    m.def("generate_3d_channel_mesh", &generate_3d_channel_mesh,
          py::arg("Lx") = 1.0, py::arg("Ly") = 0.1, py::arg("Lz") = 0.1,
          py::arg("nx") = 20, py::arg("ny") = 10, py::arg("nz") = 10,
          py::arg("boundary_names") = std::unordered_map<std::string, std::string>{});
    m.def("generate_3d_duct_mesh", &generate_3d_duct_mesh,
          py::arg("Lx") = 2.0, py::arg("Ly") = 0.1, py::arg("Lz") = 0.1,
          py::arg("nx") = 20, py::arg("ny") = 10, py::arg("nz") = 10);
    m.def("generate_3d_cavity_mesh", &generate_3d_cavity_mesh,
          py::arg("Lx") = 1.0, py::arg("Ly") = 1.0, py::arg("Lz") = 1.0,
          py::arg("nx") = 16, py::arg("ny") = 16, py::arg("nz") = 16);

    // ---- TwoFluidSolver ----
    py::class_<TwoFluidSolver>(m, "TwoFluidSolver")
        .def(py::init<FVMesh&>(), py::keep_alive<1, 2>())
        .def_readwrite("rho_l", &TwoFluidSolver::rho_l)
        .def_readwrite("rho_g", &TwoFluidSolver::rho_g)
        .def_readwrite("mu_l", &TwoFluidSolver::mu_l)
        .def_readwrite("mu_g", &TwoFluidSolver::mu_g)
        .def_readwrite("cp_l", &TwoFluidSolver::cp_l)
        .def_readwrite("cp_g", &TwoFluidSolver::cp_g)
        .def_readwrite("k_l", &TwoFluidSolver::k_l)
        .def_readwrite("k_g", &TwoFluidSolver::k_g)
        .def_readwrite("d_b", &TwoFluidSolver::d_b)
        .def_readwrite("h_fg", &TwoFluidSolver::h_fg)
        .def_readwrite("T_sat", &TwoFluidSolver::T_sat)
        .def_readwrite("r_phase_change", &TwoFluidSolver::r_phase_change)
        .def_readwrite("alpha_u", &TwoFluidSolver::alpha_u)
        .def_readwrite("alpha_p", &TwoFluidSolver::alpha_p)
        .def_readwrite("alpha_alpha", &TwoFluidSolver::alpha_alpha)
        .def_readwrite("alpha_T", &TwoFluidSolver::alpha_T)
        .def_readwrite("tol", &TwoFluidSolver::tol)
        .def_readwrite("max_outer_iter", &TwoFluidSolver::max_outer_iter)
        .def_readwrite("solve_energy", &TwoFluidSolver::solve_energy)
        .def_readwrite("solve_momentum", &TwoFluidSolver::solve_momentum)
        .def_readwrite("g", &TwoFluidSolver::g)
        .def_readwrite("dt", &TwoFluidSolver::dt)
        .def("initialize", &TwoFluidSolver::initialize,
             py::arg("alpha_g_init") = 0.05)
        .def("set_wall_bc", &TwoFluidSolver::set_wall_bc,
             py::arg("patch"), py::arg("q_wall") = 0.0)
        .def("set_inlet_bc", &TwoFluidSolver::set_inlet_bc,
             py::arg("patch"), py::arg("alpha_g"),
             py::arg("U_l"), py::arg("U_g"),
             py::arg("T_l") = 0.0, py::arg("T_g") = 0.0)
        .def("set_outlet_bc", &TwoFluidSolver::set_outlet_bc,
             py::arg("patch"), py::arg("p_val") = 0.0)
        .def("solve_transient", &TwoFluidSolver::solve_transient,
             py::arg("t_end"), py::arg("dt"), py::arg("report_interval") = 100)
        .def("alpha_g_field", &TwoFluidSolver::alpha_g_field,
             py::return_value_policy::reference_internal)
        .def("alpha_l_field", &TwoFluidSolver::alpha_l_field,
             py::return_value_policy::reference_internal)
        .def("U_l_field", &TwoFluidSolver::U_l_field,
             py::return_value_policy::reference_internal)
        .def("U_g_field", &TwoFluidSolver::U_g_field,
             py::return_value_policy::reference_internal)
        .def("pressure", &TwoFluidSolver::pressure,
             py::return_value_policy::reference_internal)
        .def("T_l_field", &TwoFluidSolver::T_l_field,
             py::return_value_policy::reference_internal)
        .def("T_g_field", &TwoFluidSolver::T_g_field,
             py::return_value_policy::reference_internal);
}
