#pragma once

#include <cassert>
#include <cmath>
#include <vector>

#include "Kokkos_Macros.hpp"
#include "kokkos_types.hpp"
#include <Kokkos_Core.hpp>

namespace athelas::math::linalg {

// Fallback modes for matrix inversion when determinant is poor.
// None: leave the caller to handle inf/NaN from a singular matrix.
// Identity: return identity matrices, used for characteristic limiting where
// the safest fallback is component-wise limiting.
enum class InversionFallback {
  None,
  Identity,
};

using Scalar = double;
using ExecSpace = Kokkos::DefaultExecutionSpace;
using MemSpace = ExecSpace::memory_space;

using Layout = Kokkos::LayoutRight;
using BlockStore =
    Kokkos::View<Scalar ***, Layout, MemSpace>; // [N, m, m] or [N-1, m, m]
using VecStore = Kokkos::View<Scalar **, Layout, MemSpace>; // [N,m] or [N-1, m]
using PivotStore = Kokkos::View<int *, Layout, MemSpace>; // [m]

/**
 * @brief Computes a quadrature-weighted L2 error norm for a Newton-Raphson
 * iteration.
 *
 * @param du      2D Kokkos view of shape (nx, 4*nNodes) containing the Newton
 * update vector
 * @param wgts    1D Kokkos view of length nNodes containing quadrature weights
 * @return Dimensionless L2 norm of the scaled update vector
 */
auto newton_norm_l2(AthelasArray2D<double> du, AthelasArray2D<double> sqrt_gm,
                    AthelasArray1D<double> dr, AthelasArray1D<double> wgts)
    -> double;

/**
 * @struct ThomasScratch
 * @brief Scratch storage for block Thomas solve.
 */
struct ThomasScratch {
  // Workspaces that persist across solves to avoid allocations.
  BlockStore W; // [N - 1, m, m]
  VecStore Y; // [N - 1, m]

  // Per-block solve scratch (reused for each block).
  AthelasArray2D<double> Bi_lu; // [m, m]
};

/**
 * @ brief Block Thomas Algorithm for block tridiagonal systems
 *
 *  Solves the N×N block tridiagonal system:
 *
 *   [ B(0,:,:)  C(0,:,:)                          ] [ x(0,:)   ]   [ d(0,:)   ]
 *   [ A(1,:,:)  B(1,:,:)  C(1,:,:)                ] [ x(1,:)   ]   [ d(1,:)   ]
 *   [           A(2,:,:)  B(2,:,:)  C(2,:,:)      ] [ x(2,:)   ] = [ d(2,:)   ]
 *   [                      ...                    ] [  ...     ]   [  ...     ]
 *   [                     A(N-1:,:) B(N-1:,:)     ] [ x(N-1,:) ]   [ d(N-1,:) ]
 *
 * each block is m×m dense; A(:,:,0) and C(:,:,N-1) are unused.
 *
 * Algorithm:
 *   Forward sweep (i = 0 .. N-2):
 *     Solve B(i) * [W(i) | Y(i)] = [C(i) | d(i)]
 *     B(i+1) -= A(i+1) * W(i)
 *     d(i+1) -= A(i+1) * Y(i)
 *
 *   Back substitution:
 *     Solve B(N-1) * x(N-1) = d(N-1)
 *     x(i) = Y(i) - W(i) * x(i+1)  for i = N-2..0
 *
 *  NOTE: Can I remove Bi_lu and factor into B?
 */
void block_thomas_solve(int N, int m, BlockStore A, BlockStore B, BlockStore C,
                        VecStore d, const ThomasScratch &scratch);

template <class T>
KOKKOS_INLINE_FUNCTION void fill_identity(T A, const int n) {
  for (int r = 0; r < n; ++r) {
    for (int c = 0; c < n; ++c) {
      A(r, c) = (r == c) ? 1.0 : 0.0;
    }
  }
}

template <InversionFallback Fallback = InversionFallback::None, class Matrix,
          class Inverse>
KOKKOS_INLINE_FUNCTION void invert_3x3(Matrix A, Inverse A_inv) {
  assert(A.extent(0) == 3 && A.extent(1) == 3 &&
         "invert_3x3 input matrix must be 3x3");
  assert(A_inv.extent(0) == 3 && A_inv.extent(1) == 3 &&
         "invert_3x3 output matrix must be 3x3");

  const double a00 = A(0, 0);
  const double a01 = A(0, 1);
  const double a02 = A(0, 2);
  const double a10 = A(1, 0);
  const double a11 = A(1, 1);
  const double a12 = A(1, 2);
  const double a20 = A(2, 0);
  const double a21 = A(2, 1);
  const double a22 = A(2, 2);

  const double c00 = a11 * a22 - a12 * a21;
  const double c01 = -(a10 * a22 - a12 * a20);
  const double c02 = a10 * a21 - a11 * a20;
  const double c10 = -(a01 * a22 - a02 * a21);
  const double c11 = a00 * a22 - a02 * a20;
  const double c12 = -(a00 * a21 - a01 * a20);
  const double c20 = a01 * a12 - a02 * a11;
  const double c21 = -(a00 * a12 - a02 * a10);
  const double c22 = a00 * a11 - a01 * a10;

  const double det = a00 * c00 + a01 * c01 + a02 * c02;
  if constexpr (Fallback == InversionFallback::Identity) {
    // Safety. If the determinant blows up, set identity matrices.
    // Useful in characteristic limiting eigen matrices.
    if (!std::isfinite(det) || std::abs(det) <= 1.0e-300) {
      fill_identity(A, 3);
      fill_identity(A_inv, 3);
      return;
    }
  }

  const double inv_det = 1.0 / det;
  A_inv(0, 0) = c00 * inv_det;
  A_inv(0, 1) = c10 * inv_det;
  A_inv(0, 2) = c20 * inv_det;
  A_inv(1, 0) = c01 * inv_det;
  A_inv(1, 1) = c11 * inv_det;
  A_inv(1, 2) = c21 * inv_det;
  A_inv(2, 0) = c02 * inv_det;
  A_inv(2, 1) = c12 * inv_det;
  A_inv(2, 2) = c22 * inv_det;
}

// Fill identity matrix
template <class T>
KOKKOS_INLINE_FUNCTION constexpr void identity_matrix(T Mat, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (i == j) {
        Mat(i, j) = 1.0;
      } else {
        Mat(i, j) = 0.0;
      }
    }
  }
}

/**
 * @brief Matrix vector multiplication
 **/
template <int N, class M, class V>
KOKKOS_INLINE_FUNCTION constexpr void mat_mul(double alpha, M A, V x,
                                              double beta, V y) {
  static_assert(M::rank == 2 && V::rank == 1,
                "Input types must be rank 2 and rank 1 views.");
  // Calculate A*x=y
  for (int i = 0; i < N; i++) {
    double sum = 0.0;
    for (int j = 0; j < N; j++) {
      sum += A(i, j) * x(j);
    }
    y(i) = alpha * sum + beta * y(i);
  }
}
KOKKOS_FUNCTION
void tri_sym_diag(int n, std::vector<double> &d, std::vector<double> &e,
                  std::vector<double> &array);
KOKKOS_FUNCTION
void invert_matrix(std::vector<double> &M, int n);
} // namespace athelas::math::linalg
