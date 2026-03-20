#pragma once
#ifdef USE_MPI
#include <mpi.h>
#endif
#include <vector>
#include <Eigen/Dense>
#include "twofluid/mesh.hpp"
#include "twofluid/fields.hpp"

namespace twofluid {

/// MPI communicator wrapper. All MPI calls go through this class.
/// When compiled without USE_MPI, provides serial stubs.
class MPIComm {
public:
    MPIComm();
    ~MPIComm();

    int rank() const { return rank_; }
    int size() const { return size_; }
    bool is_root() const { return rank_ == 0; }
    bool is_parallel() const { return size_ > 1; }

    void barrier();

    // Collective operations
    double all_reduce_sum(double val);
    double all_reduce_max(double val);
    double all_reduce_min(double val);

    // Point-to-point for ghost cell exchange
    void send_scalar(const Eigen::VectorXd& data, const std::vector<int>& indices, int dest, int tag);
    Eigen::VectorXd recv_scalar(int count, int src, int tag);
    void send_vector(const Eigen::MatrixXd& data, const std::vector<int>& indices, int dest, int tag);
    Eigen::MatrixXd recv_vector(int count, int ndim, int src, int tag);

private:
    int rank_ = 0;
    int size_ = 1;
#ifdef USE_MPI
    bool owns_mpi_ = false;  // true if we called MPI_Init
#endif
};

} // namespace twofluid
