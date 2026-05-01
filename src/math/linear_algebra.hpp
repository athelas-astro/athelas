#pragma once

#include <vector>

#include "kokkos_types.hpp"
#include "Kokkos_Macros.hpp"
#include <Kokkos_Core.hpp>

namespace athelas::math::linalg {


using Scalar    = double;
using ExecSpace = Kokkos::DefaultExecutionSpace;
using MemSpace  = ExecSpace::memory_space;

using Layout     = Kokkos::LayoutRight;
using BlockStore = Kokkos::View<Scalar***, Layout, MemSpace>; // [N, m, m] or [N-1, m, m]
using VecStore   = Kokkos::View<Scalar**,  Layout, MemSpace>; // [N,m] or [N-1, m]
using PivotStore = Kokkos::View<int*,      Layout, MemSpace>; // [m]

/**
 * @brief Computes a quadrature-weighted L2 error norm for a Newton-Raphson iteration.
 *
 * @param du      2D Kokkos view of shape (nx, 2*nNodes) containing the Newton update vector
 * @param wgts    1D Kokkos view of length nNodes containing quadrature weights
 * @param scale_e  Characteristic scale for the first quantity
 * @param scale_f  Characteristic scale for the second quantity
 * @return Dimensionless L2 norm of the scaled update vector
 */
auto newton_norm_l2(
    AthelasArray2D<double> du,
    AthelasArray1D<double> wgts,
    double scale_e,
    double scale_f) -> double;

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
void block_thomas_solve(int N, int m, BlockStore A, BlockStore B,
                        BlockStore C, VecStore d,
                        const ThomasScratch &scratch);

// Fill identity matrix
template <class T>
KOKKOS_INLINE_FUNCTION constexpr void IDENTITY_MATRIX(T Mat, int n) {
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
KOKKOS_INLINE_FUNCTION constexpr void MAT_MUL(double alpha, M A, V x,
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
