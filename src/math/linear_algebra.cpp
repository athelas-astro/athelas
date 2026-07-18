#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <Eigen/Dense>
#include <Kokkos_Core.hpp>

#include "kokkos_abstraction.hpp"
#include "kokkos_types.hpp"
#include "loop_layout.hpp"
#include "math/linear_algebra.hpp"
#include "utils/error.hpp"

#include <KokkosBatched_Gemm_Decl.hpp>
#include <KokkosBatched_Getrf.hpp>
#include <KokkosBatched_Getrs.hpp>

namespace athelas::math::linalg {

auto newton_norm_l2(AthelasArray2D<double> du, AthelasArray2D<double> sqrt_gm,
                    AthelasArray1D<double> dr, AthelasArray1D<double> wgts)
    -> double {
  auto idx = [&](const int q, const int v) { return 4 * q + v; };

  static const int nq = static_cast<int>(wgts.size());
  static const int nx = static_cast<int>(du.extent(0));
  static const IndexRange ib(nx);
  static const IndexRange qb(nq);

  // Scale before squaring. Radiation residuals can be large enough that
  // r*r overflows even though the weighted L2 norm is representable.
  double max_abs = 0.0;
  athelas::par_reduce(
      DEFAULT_LOOP_PATTERN, "norm_l2_max", DevExecSpace(), ib.s, ib.e, qb.s,
      qb.e,
      KOKKOS_LAMBDA(const int i, const int q, double &lmax) {
        for (int v = 0; v < 4; ++v) {
          const double r = du(i, idx(q, v));
          lmax = std::max(lmax, std::isfinite(r)
                                    ? std::abs(r)
                                    : std::numeric_limits<double>::infinity());
        }
      },
      Kokkos::Max<double>(max_abs));

  if (max_abs == 0.0 || !std::isfinite(max_abs)) {
    return max_abs;
  }

  double norm_sq_scaled = 0.0;
  athelas::par_reduce(
      DEFAULT_LOOP_PATTERN, "norm_l2", DevExecSpace(), ib.s, ib.e, qb.s, qb.e,
      KOKKOS_LAMBDA(const int i, const int q, double &lnorm_sq_scaled) {
        const double w = wgts(q);
        const double gm = sqrt_gm(i, q + 1);
        for (int v = 0; v < 4; ++v) {
          const double r_scaled = du(i, idx(q, v)) / max_abs;
          lnorm_sq_scaled += w * r_scaled * r_scaled * gm * dr(i);
        }
      },
      Kokkos::Sum<double>(norm_sq_scaled));

  return max_abs * Kokkos::sqrt(norm_sq_scaled);
}

void block_thomas_solve(const int N, const int m, BlockStore A, BlockStore B,
                        BlockStore C, VecStore d,
                        const ThomasScratch &scratch) {
  Kokkos::Profiling::pushRegion("BlockThomas");

  auto W = scratch.W;
  auto Y = scratch.Y;
  auto Bi_lu = scratch.Bi_lu;
  auto piv = scratch.piv;
  auto factor_info = scratch.factor_info;
  auto min_relative_pivot = scratch.min_relative_pivot;
  auto row_scale = scratch.row_scale;
  auto col_scale = scratch.col_scale;
  const bool collect_diagnostics =
      factor_info.extent(0) == 1 && min_relative_pivot.extent(0) == 1;
  const bool equilibrate =
      row_scale.extent_int(0) == N && row_scale.extent_int(1) == m &&
      col_scale.extent_int(0) == N && col_scale.extent_int(1) == m;

  if (collect_diagnostics) {
    Kokkos::deep_copy(factor_info, 0);
    Kokkos::deep_copy(min_relative_pivot,
                      std::numeric_limits<double>::infinity());
  }

  if (equilibrate) {
    // Row then column max-norm equilibration. With R and C diagonal, solve
    // R A C y = R b and return x = C y. Bounds passed to par_for are
    // inclusive in this codebase, hence N - 1 and m - 1 below.
    athelas::par_for(
        DEFAULT_LOOP_PATTERN, "block_thomas_row_scale", DevExecSpace(), 0,
        N - 1, 0, m - 1, KOKKOS_LAMBDA(const int i, const int r) {
          double row_max = 0.0;
          for (int c = 0; c < m; ++c) {
            row_max = std::max(row_max, Kokkos::abs(B(i, r, c)));
            if (i > 0) {
              row_max = std::max(row_max, Kokkos::abs(A(i - 1, r, c)));
            }
            if (i < N - 1) {
              row_max = std::max(row_max, Kokkos::abs(C(i, r, c)));
            }
          }
          row_scale(i, r) =
              row_max > 0.0 && std::isfinite(row_max) ? 1.0 / row_max : 1.0;
        });
    athelas::par_for(
        DEFAULT_LOOP_PATTERN, "block_thomas_apply_row_scale", DevExecSpace(), 0,
        N - 1, 0, m - 1, KOKKOS_LAMBDA(const int i, const int r) {
          const double scale = row_scale(i, r);
          d(i, r) *= scale;
          for (int c = 0; c < m; ++c) {
            B(i, r, c) *= scale;
            if (i > 0) {
              A(i - 1, r, c) *= scale;
            }
            if (i < N - 1) {
              C(i, r, c) *= scale;
            }
          }
        });

    athelas::par_for(
        DEFAULT_LOOP_PATTERN, "block_thomas_col_scale", DevExecSpace(), 0,
        N - 1, 0, m - 1, KOKKOS_LAMBDA(const int i, const int c) {
          double col_max = 0.0;
          for (int r = 0; r < m; ++r) {
            col_max = std::max(col_max, Kokkos::abs(B(i, r, c)));
            if (i < N - 1) {
              col_max = std::max(col_max, Kokkos::abs(A(i, r, c)));
            }
            if (i > 0) {
              col_max = std::max(col_max, Kokkos::abs(C(i - 1, r, c)));
            }
          }
          col_scale(i, c) =
              col_max > 0.0 && std::isfinite(col_max) ? 1.0 / col_max : 1.0;
        });
    athelas::par_for(
        DEFAULT_LOOP_PATTERN, "block_thomas_apply_col_scale", DevExecSpace(), 0,
        N - 1, 0, m - 1, KOKKOS_LAMBDA(const int i, const int c) {
          const double scale = col_scale(i, c);
          for (int r = 0; r < m; ++r) {
            B(i, r, c) *= scale;
            if (i < N - 1) {
              A(i, r, c) *= scale;
            }
            if (i > 0) {
              C(i - 1, r, c) *= scale;
            }
          }
        });
  }

  using Getrf =
      KokkosBatched::SerialGetrf<KokkosBatched::Algo::Getrf::Unblocked>;
  using Getrs =
      KokkosBatched::SerialGetrs<KokkosBatched::Trans::NoTranspose,
                                 KokkosBatched::Algo::Getrs::Unblocked>;

  using Gemm = KokkosBatched::SerialGemm<KokkosBatched::Trans::NoTranspose,
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
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "block_thomas", DevExecSpace(), 0, 0,
      KOKKOS_LAMBDA(const int) {
        int first_factor_info = 0;
        double min_pivot_ratio = std::numeric_limits<double>::infinity();

        auto factor = [&](auto mat) {
          double max_entry = 0.0;
          for (int r = 0; r < m; ++r) {
            for (int c = 0; c < m; ++c) {
              max_entry = std::max(max_entry, Kokkos::abs(mat(r, c)));
            }
          }

          const int info = Getrf::invoke(mat, piv);
          if (first_factor_info == 0 && info != 0) {
            first_factor_info = info;
          }
          if (max_entry > 0.0) {
            for (int r = 0; r < m; ++r) {
              min_pivot_ratio =
                  std::min(min_pivot_ratio, Kokkos::abs(mat(r, r)) / max_entry);
            }
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
          auto Bi = Kokkos::subview(B, i, Kokkos::ALL, Kokkos::ALL);
          auto Ci = Kokkos::subview(C, i, Kokkos::ALL, Kokkos::ALL);
          auto di = Kokkos::subview(d, i, Kokkos::ALL);
          auto Ai1 = Kokkos::subview(A, i, Kokkos::ALL, Kokkos::ALL);
          auto Bi1 = Kokkos::subview(B, i + 1, Kokkos::ALL, Kokkos::ALL);
          auto di1 = Kokkos::subview(d, i + 1, Kokkos::ALL);
          auto Wi = Kokkos::subview(W, i, Kokkos::ALL, Kokkos::ALL);
          auto Yi = Kokkos::subview(Y, i, Kokkos::ALL);

          // Factor B(i) into Bi_lu, leaving B untouched
          mat_copy(Bi, Bi_lu);
          factor(Bi_lu);

          // W(i) = B(i)^{-1} C(i)
          mat_copy(Ci, Wi);
          for (int col = 0; col < m; ++col) {
            auto W_col = Kokkos::subview(Wi, Kokkos::ALL, col);
            Getrs::invoke(Bi_lu, piv, W_col);
          }

          // Y(i) = B(i)^{-1} d(i)
          for (int r = 0; r < m; ++r) {
            Yi(r) = di(r);
          }
          Getrs::invoke(Bi_lu, piv, Yi);

          // B(i+1) -= A(i+1) * W(i)
          Gemm::invoke(-1.0, Ai1, Wi, 1.0, Bi1);

          // d(i+1) -= A(i+1) * Y(i)
          gemv_sub(Ai1, Yi, di1);
        }

        // Terminal solve: B(N-1) * x(N-1) = d(N-1)
        auto BN = Kokkos::subview(B, N - 1, Kokkos::ALL, Kokkos::ALL);
        auto dN = Kokkos::subview(d, N - 1, Kokkos::ALL);

        mat_copy(BN, Bi_lu);
        factor(Bi_lu);
        Getrs::invoke(Bi_lu, piv, dN); // dN overwritten with x(N-1)

        // Back substitution: x(i) = Y(i) - W(i) * x(i+1)
        // Solution written into d in-place.
        for (int i = N - 2; i >= 0; --i) {
          auto Wi = Kokkos::subview(W, i, Kokkos::ALL, Kokkos::ALL);
          auto Yi = Kokkos::subview(Y, i, Kokkos::ALL);
          auto xi = Kokkos::subview(d, i, Kokkos::ALL);
          auto xi1 = Kokkos::subview(d, i + 1, Kokkos::ALL);

          for (int r = 0; r < m; ++r) {
            xi(r) = Yi(r);
          }
          gemv_sub(Wi, xi1, xi);
        }

        if (collect_diagnostics) {
          factor_info(0) = first_factor_info;
          min_relative_pivot(0) = min_pivot_ratio;
        }
      });

  if (equilibrate) {
    athelas::par_for(
        DEFAULT_LOOP_PATTERN, "block_thomas_restore_col_scale", DevExecSpace(),
        0, N - 1, 0, m - 1, KOKKOS_LAMBDA(const int i, const int c) {
          d(i, c) *= col_scale(i, c);
        });
  }

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
