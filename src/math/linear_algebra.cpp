#include <vector>

#include <Eigen/Dense>
#include <Kokkos_Core.hpp>

#include "kokkos_abstraction.hpp"
#include "math/linear_algebra.hpp"
#include "kokkos_types.hpp"
#include "loop_layout.hpp"
#include "utils/error.hpp"

#include <KokkosBatched_LU_Decl.hpp>
#include <KokkosBatched_Trsm_Decl.hpp>
#include <KokkosBatched_Gemm_Decl.hpp>


namespace athelas::math::linalg {

auto newton_norm_l2(
    AthelasArray2D<double> du,
    AthelasArray1D<double> wgts,
    const double scale_e,
    const double scale_f) -> double {
  assert(scale_e > 0.0 && "newton_norm_l2 :: scale_e must be positive definite!");
  assert(scale_f > 0.0 && "newton_norm_l2 :: scale_f must be positive definite!");

  auto idx = [&](const int q, const int v) { return q * 4 + v; };

  Kokkos::Array<const double, 2> inv_scale = {1.0 / scale_e, 1.0 / scale_f};

  static const int nq = static_cast<int>(wgts.size());
  static const int nx = static_cast<int>(du.extent(0));
  static const IndexRange ib(nx);
  static const IndexRange qb(nq);

  double norm_sq = 0.0;
  athelas::par_reduce(
      DEFAULT_LOOP_PATTERN, "norm_l2", DevExecSpace(),
      ib.s, ib.e, qb.s, qb.e,
      KOKKOS_LAMBDA(const int i, const int q, double& lnorm_sq) {
        const double w = wgts(q);
        for (int v = 0; v < 2; ++v) {
          const double r = du(i, idx(q, v)) * inv_scale[v];
          lnorm_sq += w * r * r;
        }
      },
      Kokkos::Sum<double>(norm_sq));

  return Kokkos::sqrt(norm_sq);
}

void block_thomas_solve(const int N, const int m,
                        BlockStore A, BlockStore B, BlockStore C,
                        VecStore d, const ThomasScratch& scratch) {
    Kokkos::Profiling::pushRegion("BlockThomas");

    auto W     = scratch.W;
    auto Y     = scratch.Y;
    auto Bi_lu = scratch.Bi_lu;

    using LU     = KokkosBatched::SerialLU<
                       KokkosBatched::Algo::LU::Unblocked>;

    using TrsmLL = KokkosBatched::SerialTrsm<
                       KokkosBatched::Side::Left,
                       KokkosBatched::Uplo::Lower,
                       KokkosBatched::Trans::NoTranspose,
                       KokkosBatched::Diag::Unit,
                       KokkosBatched::Algo::Trsm::Unblocked>;

    using TrsmLU = KokkosBatched::SerialTrsm<
                       KokkosBatched::Side::Left,
                       KokkosBatched::Uplo::Upper,
                       KokkosBatched::Trans::NoTranspose,
                       KokkosBatched::Diag::NonUnit,
                       KokkosBatched::Algo::Trsm::Unblocked>;

    using Gemm   = KokkosBatched::SerialGemm<
                       KokkosBatched::Trans::NoTranspose,
                       KokkosBatched::Trans::NoTranspose,
                       KokkosBatched::Algo::Gemm::Unblocked>;

    // Copy mxm matrix src -> dst
    auto mat_copy = [&](auto src, auto dst) {
        for (int r = 0; r < m; ++r) {
            for (int c = 0; c < m; ++c) {
                dst(r, c) = src(r, c);
            }
        }
    };

    // This is a 0D loop, it simply creates a "parallel" region
    athelas::par_for(DEFAULT_FLAT_LOOP_PATTERN, "block_thomas",
        DevExecSpace(), 0, 0,
        KOKKOS_LAMBDA(const int) {


            // Apply already-factored Bi_lu to a 1D vector in-place.
            // Forward substitution (unit lower), then back substitution (non-unit upper).
            auto trsv = [&](auto vec) {
                for (int r = 0; r < m; ++r) {
                    for (int k = 0; k < r; ++k) {
                        vec(r) = Kokkos::fma(-Bi_lu(r, k), vec(k), vec(r));
                    }
                    // unit diagonal - no division
                }
                for (int r = m - 1; r >= 0; --r) {
                    for (int k = r + 1; k < m; ++k) {
                        vec(r) = Kokkos::fma(-Bi_lu(r, k), vec(k), vec(r));
                    }
                    vec(r) /= Bi_lu(r, r);
                }
            };

            // out_vec -= mat * rhs_vec
            auto gemv_sub = [&](auto mat, auto rhs_vec, auto out_vec) {
                for (int r = 0; r < m; ++r) {
                    double acc = 0;
                    for (int k = 0; k < m; ++k) {
                        acc = Kokkos::fma(mat(r, k), rhs_vec(k), acc);
                    }
                    out_vec(r) -= acc;
                }
            };

            // Forward sweep
            for (int i = 0; i < N - 1; ++i) {
                auto Bi  = Kokkos::subview(B, i, Kokkos::ALL, Kokkos::ALL);
                auto Ci  = Kokkos::subview(C, i, Kokkos::ALL, Kokkos::ALL);
                auto di  = Kokkos::subview(d, i, Kokkos::ALL);
                auto Ai1 = Kokkos::subview(A, i, Kokkos::ALL, Kokkos::ALL);
                auto Bi1 = Kokkos::subview(B, i + 1, Kokkos::ALL, Kokkos::ALL);
                auto di1 = Kokkos::subview(d, i + 1, Kokkos::ALL);
                auto Wi  = Kokkos::subview(W, i, Kokkos::ALL, Kokkos::ALL);
                auto Yi  = Kokkos::subview(Y, i,  Kokkos::ALL);

                // Factor B(i) into Bi_lu, leaving B untouched
                mat_copy(Bi, Bi_lu);
                LU::invoke(Bi_lu);

                // W(i) = B(i)^{-1} C(i)
                mat_copy(Ci, Wi);
                TrsmLL::invoke(1.0, Bi_lu, Wi);
                TrsmLU::invoke(1.0, Bi_lu, Wi);

                // Y(i) = B(i)^{-1} d(i)
                for (int r = 0; r < m; ++r) {
                   Yi(r) = di(r);
                }
                trsv(Yi);

                // B(i+1) -= A(i+1) * W(i)
                Gemm::invoke(-1.0, Ai1, Wi, 1.0, Bi1);

                // d(i+1) -= A(i+1) * Y(i)
                gemv_sub(Ai1, Yi, di1);
            }

            // Terminal solve: B(N-1) * x(N-1) = d(N-1)
            auto BN = Kokkos::subview(B, N - 1, Kokkos::ALL, Kokkos::ALL);
            auto dN = Kokkos::subview(d, N - 1, Kokkos::ALL);

            mat_copy(BN, Bi_lu);
            LU::invoke(Bi_lu);
            trsv(dN);  // dN overwritten with x(N-1)

            // Back substitution: x(i) = Y(i) - W(i) * x(i+1)
            // Solution written into d in-place.
            for (int i = N - 2; i >= 0; --i) {
                auto Wi  = Kokkos::subview(W, i, Kokkos::ALL, Kokkos::ALL);
                auto Yi  = Kokkos::subview(Y, i, Kokkos::ALL);
                auto xi  = Kokkos::subview(d, i, Kokkos::ALL);
                auto xi1 = Kokkos::subview(d, i + 1, Kokkos::ALL);

                for (int r = 0; r < m; ++r) {
                  xi(r) = Yi(r);
                }
                gemv_sub(Wi, xi1, xi);
            }
        });

    Kokkos::Profiling::popRegion();
}

/**
 * @brief Diagonalizes a symmetric tridiagonal matrix using Eigen.
 */
void tri_sym_diag(int n, std::vector<double> &d, std::vector<double> &e,
                  std::vector<double> &array) {
  assert(n > 0 && "Matrix dim must be > 0");

  // trivial
  if (n == 1) {
    return;
  }

  // Create tridiagonal matrix
  Eigen::MatrixXd tri_matrix = Eigen::MatrixXd::Zero(n, n);

  // Fill diagonal
  for (int i = 0; i < n; i++) {
    tri_matrix(i, i) = d[i];
  }

  // Fill super/sub diagonals
  for (int i = 0; i < n - 1; i++) {
    tri_matrix(i, i + 1) = e[i];
    tri_matrix(i + 1, i) = e[i];
  }

  // Compute eigendecomposition
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(tri_matrix);

  if (solver.info() != Eigen::Success) {
    throw_athelas_error("Eigendecomposition failed in tri_sym_diag");
  }

  // Get eigenvalues and eigenvectors
  const Eigen::VectorXd &eigenvalues = solver.eigenvalues();
  const Eigen::MatrixXd &eigenvectors = solver.eigenvectors();

  // Update d with eigenvalues (Eigen returns them in ascending order)
  for (int i = 0; i < n; i++) {
    d[i] = eigenvalues(i);
  }

  // Matrix multiply eigenvectors' * array. Only array[0] is nonzero initially.
  double const k = array[0];
  for (int i = 0; i < n; i++) {
    // First row of eigenvectors matrix (corresponding to first element of
    // array)
    array[i] = k * eigenvectors(0, i);
  }
}
/**
 * @brief Use Eigen to invert a matrix M using LU factorization.
 **/
void invert_matrix(std::vector<double> &M, int n) {
  // Map the std::vector to an Eigen matrix (column-major order)
  Eigen::Map<Eigen::MatrixXd> matrix(M.data(), n, n);

  // Compute the inverse using LU decomposition
  Eigen::MatrixXd inverse = matrix.inverse();

  // Check if the matrix is invertible by verifying determinant is non-zero
  if (std::abs(matrix.determinant()) < std::numeric_limits<double>::epsilon()) {
    throw_athelas_error(" ! Issue occurred in matrix inversion.");
  }

  // Copy the result back to the original vector
  matrix = inverse;
}

} // namespace athelas::math::linalg
