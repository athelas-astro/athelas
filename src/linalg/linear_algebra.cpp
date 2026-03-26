/**
 * @file linear_algebra.cpp
 * --------------
 *
 * @author Brandon L. Barker
 * @brief Basic linear algebra functions.
 *
 * @details Linear algebra routines for quadrature and limiters.
 *          - tri_sym_diag
 *          - invert_matrix
 */

#include <cstddef>
#include <vector>

#include <Eigen/Dense>
#include <Kokkos_Core.hpp>

#include "linalg/linear_algebra.hpp"
#include "utils/error.hpp"

#include <KokkosLapack_gesv.hpp>
#include "KokkosBlas.hpp"

namespace athelas {

// =============================================================================
// Block Thomas Algorithm for Block Tridiagonal Systems
//
// Solves the N×N block tridiagonal system:
//
//   [ B(:,:,0)  C(:,:,0)                          ] [ x(:,0)   ]   [ d(:,0)   ]
//   [ A(:,:,1)  B(:,:,1)  C(:,:,1)                ] [ x(:,1)   ]   [ d(:,1)   ]
//   [           A(:,:,2)  B(:,:,2)  C(:,:,2)      ] [ x(:,2)   ] = [ d(:,2)   ]
//   [                      ...                    ] [  ...     ]   [  ...     ]
//   [                     A(:,:,N-1) B(:,:,N-1)   ] [ x(:,N-1) ]   [ d(:,N-1) ]
//
// each block is m×m dense; A(:,:,0) and C(:,:,N-1) are unused.
//
// Algorithm:
//   Forward sweep (i = 0 .. N-2):
//     Solve B(i) * [W(i) | Y(i)] = [C(i) | d(i)]   ← augmented gesv
//     B(i+1) -= A(i+1) * W(i)                        ← gemm
//     d(i+1) -= A(i+1) * Y(i)                        ← gemv
//
//   Back substitution:
//     Solve B(N-1) * x(N-1) = d(N-1)                 ← gesv
//     x(i) = Y(i) - W(i) * x(i+1)  for i = N-2..0   ← gemv
//
// =============================================================================



// ---------------------------------------------------------------------------
// Storage layout choice: [m, m, N] with LayoutLeft
//
// With LayoutLeft the *first* index strides fastest. Storing blocks as
// [row, col, block_index] means subview(..., ALL, ALL, i) produces a 2D
// [m×m] view whose strides are (1, m) — i.e. contiguous column-major
// storage as expected by LAPACK-backed routines.
//
// The more intuitive [N, m, m] layout would give non-unit row-stride in
// the block subview, which can cause issues with gesv depending on the
// backend. Using [m, m, N] avoids that problem entirely.
// ---------------------------------------------------------------------------

// ===========================================================================
// block_thomas_solve
//
// Parameters:
//   N     — number of block rows/columns
//   m     — block size
//   A     — lower diagonal blocks [m,m,N-1], A(:,:,0) unused,  read-only
//   B     — main  diagonal blocks [m,m,N], overwritten during solve
//   C     — upper diagonal blocks [m,m,N-1], C(:,:,N-1) unused, read-only
//   d     — RHS [m,N], overwritten with solution x on output
//   W     — workspace [m,m,N-1], B(i)^{-1} * C(i)
//   Y     — workspace [m,N-1],   B(i)^{-1} * d(i)
//   ipiv  — workspace [m],       pivot array for gesv
// ===========================================================================
void block_thomas_solve(int N, int m, BlockStore A, BlockStore B,
                        BlockStore C, VecStore d, BlockStore W,
                        VecStore Y, PivotStore ipiv) {
    Kokkos::Profiling::pushRegion("BlockThomas");
    // -------------------------------------------------------------------------
    // Forward sweep
    // -------------------------------------------------------------------------
    for (int i = 0; i < N - 1; ++i) {
        auto Bi  = Kokkos::subview(B, Kokkos::ALL, Kokkos::ALL, i);
        auto Ci  = Kokkos::subview(C, Kokkos::ALL, Kokkos::ALL, i);
        auto di  = Kokkos::subview(d, Kokkos::ALL, i);
        auto Wi  = Kokkos::subview(W, Kokkos::ALL, Kokkos::ALL, i);
        auto Yi  = Kokkos::subview(Y, Kokkos::ALL, i);
        auto Ai1 = Kokkos::subview(A, Kokkos::ALL, Kokkos::ALL, i);
        auto Bi1 = Kokkos::subview(B, Kokkos::ALL, Kokkos::ALL, i + 1);
        auto di1 = Kokkos::subview(d, Kokkos::ALL, i + 1);
 
        // Augmented solve:  B(i) * [W(i) | Y(i)] = [C(i) | d(i)]
        Kokkos::View<Scalar**, Layout, MemSpace> Bi_lu("Bi_lu", m, m);
        Kokkos::View<Scalar**, Layout, MemSpace> aug("aug",     m, m + 1);
        Kokkos::deep_copy(Bi_lu, Bi);
 
        Kokkos::parallel_for("fill_aug",
            Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {m, m + 1}),
            KOKKOS_LAMBDA(int r, int c) {
                aug(r, c) = (c < m) ? Ci(r, c) : di(r);
            });
        Kokkos::fence();
 
        KokkosLapack::gesv(Bi_lu, aug, ipiv);
        Kokkos::fence();
 
        Kokkos::parallel_for("unpack_aug",
            Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {m, m + 1}),
            KOKKOS_LAMBDA(int r, int c) {
                if (c < m) {
                Wi(r, c) = aug(r, c);
                } else{
                Yi(r)    = aug(r, c);
                }
            });
        Kokkos::fence();
 
        KokkosBlas::gemm("N", "N", Scalar(-1), Ai1, Wi, Scalar(1), Bi1);
        KokkosBlas::gemv("N",      Scalar(-1), Ai1, Yi, Scalar(1), di1);
        Kokkos::fence();
    }
 
    // -------------------------------------------------------------------------
    // Terminal block solve:  B(N-1) * x(N-1) = d(N-1)
    // -------------------------------------------------------------------------
    {
        auto BN = Kokkos::subview(B, Kokkos::ALL, Kokkos::ALL, N - 1);
        auto dN = Kokkos::subview(d, Kokkos::ALL, N - 1);
 
        Kokkos::View<Scalar**, Layout, MemSpace> BN_lu("BN_lu", m, m);
        Kokkos::View<Scalar**, Layout, MemSpace> rhs("rhs",     m, 1);
        Kokkos::deep_copy(BN_lu, BN);
 
        Kokkos::parallel_for("wrap_rhs", m,
            KOKKOS_LAMBDA(int r) { rhs(r, 0) = dN(r); });
        Kokkos::fence();
 
        KokkosLapack::gesv(BN_lu, rhs, ipiv);
        Kokkos::fence();
 
        Kokkos::parallel_for("unwrap_sol", m,
            KOKKOS_LAMBDA(int r) { dN(r) = rhs(r, 0); });
        Kokkos::fence();
    }
 
    // -------------------------------------------------------------------------
    // Back substitution:  x(i) = Y(i) - W(i) * x(i+1)
    // -------------------------------------------------------------------------
    for (int i = N - 2; i >= 0; --i) {
        auto Wi  = Kokkos::subview(W, Kokkos::ALL, Kokkos::ALL, i);
        auto Yi  = Kokkos::subview(Y, Kokkos::ALL, i);
        auto xi  = Kokkos::subview(d, Kokkos::ALL, i);
        auto xi1 = Kokkos::subview(d, Kokkos::ALL, i + 1);
 
        Kokkos::deep_copy(xi, Yi);
        KokkosBlas::gemv("N", Scalar(-1), Wi, xi1, Scalar(1), xi);
        Kokkos::fence();
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

} // namespace athelas
