#pragma once
/**
 * Utilities for testing
 * Contains:
 * SoftEqual
 **/

#include <cmath>
#include <iostream>
#include <print>

#include "Kokkos_Core.hpp"

#include "io/parser.hpp"

using athelas::io::Parser;

/**
 * Test for near machine precision
 **/
inline bool soft_equal(const double &val, const double &ref,
                       const double tol = 1.0e-8) {
  if (std::abs(val - ref) < tol * std::abs(ref) + tol) {
    return true;
  } else {
    return false;
  }
}

// Utility function to print parse results
inline void print_parser_data(const Parser::ParseResult &result) {
  // Print headers
  std::print("Headers: ");
  for (size_t i = 0; i < result.headers.size(); ++i) {
    std::cout << std::format("\"{}\"", result.headers[i]);
    if (i < result.headers.size() - 1) {
      std::print(", ");
    }
  }
  std::print("\n\n");

  // Print rows
  for (size_t row_idx = 0; row_idx < result.rows.size(); ++row_idx) {
    const auto &row = result.rows[row_idx];
    std::cout << std::format("Row {}: ", row_idx + 1);

    for (size_t col_idx = 0; col_idx < row.size(); ++col_idx) {
      std::cout << std::format("\"{}\"", row[col_idx]);
      if (col_idx < row.size() - 1) {
        std::print(", ");
      }
    }
    std::print("\n\n");
  }
}
// =============================================================================
// test_helpers.hpp
// Utility functions shared across all test translation units.
// =============================================================================

using Scalar     = double;
using ExecSpace  = Kokkos::DefaultExecutionSpace;
using MemSpace   = ExecSpace::memory_space;
using Layout     = Kokkos::LayoutRight;
using BlockStore = Kokkos::View<Scalar***, Layout, MemSpace>; // [m, m, N]
using VecStore   = Kokkos::View<Scalar**,  Layout, MemSpace>; // [m, N]
using PivotStore = Kokkos::View<int*,      Layout, MemSpace>; // [m]

// ---------------------------------------------------------------------------
// Populate one block slice from a row-major std::vector (m*m elements).
// LayoutLeft is column-major, so we index as (row, col) which matches how
// the solver and LAPACK expect the data.
// ---------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION void set_block(BlockStore& store, int blk_idx, int m,
                      const std::vector<double>& vals)
{
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, store);
    for (int r = 0; r < m; ++r)
        for (int c = 0; c < m; ++c)
            h(blk_idx, r, c) = vals[static_cast<std::size_t>(r * m + c)];
    Kokkos::deep_copy(store, h);
}

// ---------------------------------------------------------------------------
// Populate one vector slice from a std::vector (m elements).
// ---------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION void set_vec(VecStore& store, int blk_idx, int m,
                    const std::vector<double>& vals)
{
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, store);
    for (int r = 0; r < m; ++r)
        h(blk_idx, r) = vals[static_cast<std::size_t>(r)];
    Kokkos::deep_copy(store, h);
}

// ---------------------------------------------------------------------------
// Extract one vector slice into a std::vector.
// ---------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION std::vector<double> get_vec(const VecStore& store, int blk_idx, int m)
{
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, store);
    std::vector<double> out(static_cast<std::size_t>(m));
    for (int r = 0; r < m; ++r)
        out[static_cast<std::size_t>(r)] = h(blk_idx, r);
    return out;
}

// ---------------------------------------------------------------------------
// Max absolute error between d (solution on device) and a flat x_exact
// vector ordered as x_exact[i*m + j].
// ---------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION double max_error(const VecStore& d, int N, int m,
                        const std::vector<double>& x_exact)
{
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, d);
    double err = 0.0;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < m; ++j)
            err = std::max(err,
                           std::fabs(h(i, j) -
                                     x_exact[static_cast<std::size_t>(i * m + j)]));
    return err;
}

// ---------------------------------------------------------------------------
// Build the canonical  -I / (diag_base+i)*I / -I  block tridiagonal problem
// whose exact solution is x = ones everywhere.
// ---------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION void build_identity_block_problem(int N, int m,
                                         BlockStore& A, BlockStore& B,
                                         BlockStore& C, VecStore&   d,
                                         double diag_base = 4.0)
{
    auto A_h = Kokkos::create_mirror_view(A);
    auto B_h = Kokkos::create_mirror_view(B);
    auto C_h = Kokkos::create_mirror_view(C);
    auto d_h = Kokkos::create_mirror_view(d);
    Kokkos::deep_copy(A_h, Scalar(0));
    Kokkos::deep_copy(B_h, Scalar(0));
    Kokkos::deep_copy(C_h, Scalar(0));
    Kokkos::deep_copy(d_h, Scalar(0));
 
    for (int i = 0; i < N; ++i) {
        const double diag = diag_base + i;
        for (int j = 0; j < m; ++j) {
            B_h(i, j, j) = diag;
            if (i > 0)   A_h(i - 1, j, j) = -1.0;
            if (i < N-1) C_h(i, j, j)     = -1.0;
        }
        const double rhs = (i == 0)   ?        diag - 1.0
                         : (i == N-1) ? -1.0 + diag
                                      : -1.0 + diag - 1.0;
        for (int j = 0; j < m; ++j) d_h(i, j) = rhs;
    }
    Kokkos::deep_copy(A, A_h);
    Kokkos::deep_copy(B, B_h);
    Kokkos::deep_copy(C, C_h);
    Kokkos::deep_copy(d, d_h);
}
