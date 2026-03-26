/**
 * @file linear_algebra.hpp
 * --------------
 *
 * @brief Basic linear algebra functions.
 *
 * @details Linear algebra routines for quadrature and limiters.
 */

#pragma once

#include <vector>

#include "Kokkos_Macros.hpp"
#include <Kokkos_Core.hpp>

namespace athelas {

using Scalar    = double;
using ExecSpace = Kokkos::DefaultExecutionSpace;
using MemSpace  = ExecSpace::memory_space;

using Layout     = Kokkos::LayoutLeft;
using BlockStore = Kokkos::View<Scalar***, Layout, MemSpace>; // [m, m, N] or [m, m, N-1]
using VecStore   = Kokkos::View<Scalar**,  Layout, MemSpace>; // [m, N] or [m, N-1]
using PivotStore = Kokkos::View<int*,      Layout, MemSpace>; // [m]

void block_thomas_solve(int N, int m, BlockStore A, BlockStore B,
                        BlockStore C, VecStore d, BlockStore W,
                        VecStore Y, PivotStore ipiv);

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
} // namespace athelas
