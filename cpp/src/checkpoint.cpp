#include "twofluid/checkpoint.hpp"

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

namespace twofluid {

// ---------------------------------------------------------------------------
// Binary I/O helpers
// ---------------------------------------------------------------------------

static void write_i32(std::ofstream& f, int32_t v) {
    f.write(reinterpret_cast<const char*>(&v), sizeof(v));
}

static void write_f64(std::ofstream& f, double v) {
    f.write(reinterpret_cast<const char*>(&v), sizeof(v));
}

static void write_str(std::ofstream& f, const std::string& s) {
    int32_t len = static_cast<int32_t>(s.size());
    write_i32(f, len);
    f.write(s.data(), len);
}

static int32_t read_i32(std::ifstream& f) {
    int32_t v;
    f.read(reinterpret_cast<char*>(&v), sizeof(v));
    return v;
}

static double read_f64(std::ifstream& f) {
    double v;
    f.read(reinterpret_cast<char*>(&v), sizeof(v));
    return v;
}

static std::string read_str(std::ifstream& f) {
    int32_t len = read_i32(f);
    std::string s(len, '\0');
    f.read(&s[0], len);
    return s;
}

// ---------------------------------------------------------------------------
// write_checkpoint
// ---------------------------------------------------------------------------

void write_checkpoint(const std::string& filename,
                       const FVMesh& mesh,
                       const std::vector<const ScalarField*>& scalar_fields,
                       const std::vector<const VectorField*>& vector_fields,
                       double time, int step)
{
    std::ofstream f(filename, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("write_checkpoint: cannot open file: " + filename);
    }

    // Magic + version
    f.write("KCFD", 4);
    write_i32(f, 1);  // version

    int32_t n_cells = static_cast<int32_t>(mesh.n_cells);
    int32_t n_scalar = static_cast<int32_t>(scalar_fields.size());
    int32_t n_vector = static_cast<int32_t>(vector_fields.size());
    int32_t ndim    = static_cast<int32_t>(mesh.ndim);

    write_i32(f, n_cells);
    write_i32(f, n_scalar);
    write_i32(f, n_vector);
    write_f64(f, time);
    write_i32(f, static_cast<int32_t>(step));
    write_i32(f, ndim);

    // Scalar fields
    for (const ScalarField* sf : scalar_fields) {
        write_str(f, sf->name());
        for (int ci = 0; ci < n_cells; ++ci) {
            write_f64(f, sf->values[ci]);
        }
    }

    // Vector fields
    for (const VectorField* vf : vector_fields) {
        write_str(f, vf->name());
        for (int ci = 0; ci < n_cells; ++ci) {
            for (int d = 0; d < ndim; ++d) {
                write_f64(f, vf->values(ci, d));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// read_checkpoint
// ---------------------------------------------------------------------------

std::pair<double, int> read_checkpoint(
    const std::string& filename,
    const FVMesh& mesh,
    std::vector<ScalarField*>& scalar_fields,
    std::vector<VectorField*>& vector_fields)
{
    std::ifstream f(filename, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("read_checkpoint: cannot open file: " + filename);
    }

    // Magic
    char magic[4];
    f.read(magic, 4);
    if (std::strncmp(magic, "KCFD", 4) != 0) {
        throw std::runtime_error("read_checkpoint: invalid magic in file: " + filename);
    }

    int32_t version   = read_i32(f);
    if (version != 1) {
        throw std::runtime_error("read_checkpoint: unsupported version " +
                                 std::to_string(version));
    }

    int32_t n_cells  = read_i32(f);
    int32_t n_scalar = read_i32(f);
    int32_t n_vector = read_i32(f);
    double  time     = read_f64(f);
    int     step     = static_cast<int>(read_i32(f));
    int32_t ndim     = read_i32(f);

    if (n_cells != static_cast<int32_t>(mesh.n_cells)) {
        throw std::runtime_error("read_checkpoint: mesh size mismatch");
    }

    // Scalar fields
    for (int si = 0; si < n_scalar; ++si) {
        std::string name = read_str(f);
        if (si < static_cast<int>(scalar_fields.size())) {
            ScalarField* sf = scalar_fields[si];
            for (int ci = 0; ci < n_cells; ++ci) {
                sf->values[ci] = read_f64(f);
            }
        } else {
            // skip
            for (int ci = 0; ci < n_cells; ++ci) read_f64(f);
        }
        (void)name;
    }

    // Vector fields
    for (int vi = 0; vi < n_vector; ++vi) {
        std::string name = read_str(f);
        if (vi < static_cast<int>(vector_fields.size())) {
            VectorField* vf = vector_fields[vi];
            for (int ci = 0; ci < n_cells; ++ci) {
                for (int d = 0; d < ndim; ++d) {
                    vf->values(ci, d) = read_f64(f);
                }
            }
        } else {
            // skip
            for (int ci = 0; ci < n_cells; ++ci) {
                for (int d = 0; d < ndim; ++d) read_f64(f);
            }
        }
        (void)name;
    }

    return {time, step};
}

// ---------------------------------------------------------------------------
// Convenience wrappers for TwoFluidSolver
// ---------------------------------------------------------------------------

void write_two_fluid_checkpoint(const std::string& filename,
                                  TwoFluidSolver& solver,
                                  double time, int step)
{
    // Scalar fields: p, alpha_g, alpha_l, T_l, T_g
    std::vector<const ScalarField*> scalars = {
        &solver.pressure(),
        &solver.alpha_g_field(),
        &solver.alpha_l_field(),
        &solver.T_l_field(),
        &solver.T_g_field()
    };

    // Vector fields: U_l, U_g
    std::vector<const VectorField*> vectors = {
        &solver.U_l_field(),
        &solver.U_g_field()
    };

    write_checkpoint(filename, solver.pressure().mesh(), scalars, vectors, time, step);
}

std::pair<double, int> read_two_fluid_checkpoint(
    const std::string& filename, TwoFluidSolver& solver)
{
    std::vector<ScalarField*> scalars = {
        &solver.pressure(),
        &solver.alpha_g_field(),
        &solver.alpha_l_field(),
        &solver.T_l_field(),
        &solver.T_g_field()
    };

    std::vector<VectorField*> vectors = {
        &solver.U_l_field(),
        &solver.U_g_field()
    };

    return read_checkpoint(filename, solver.pressure().mesh(), scalars, vectors);
}

} // namespace twofluid
