#pragma once

#include "Kokkos_Macros.hpp"
#include "geometry/grid.hpp"
#include "limiters/slope_limiter.hpp"
#include "utils/utilities.hpp"

namespace athelas {

using namespace utilities;

void conservative_correction(AthelasArray3D<double> u_k,
                             AthelasArray3D<double> ucf,
                             const GridStructure &grid, int nv);

auto initialize_slope_limiter(std::string field, const GridStructure *grid,
                              const ProblemIn *pin, IndexRange vars)
    -> SlopeLimiter;

// Standard MINMOD function
template <typename T>
constexpr auto MINMOD(T a, T b, T c) -> T {
  if (SGN(a) == SGN(b) && SGN(b) == SGN(c)) {
    return SGN(a) * std::min({std::abs(a), std::abs(b), std::abs(c)});
  }
  return T(0);
}

// TVB MINMOD function
template <typename T>
constexpr auto MINMOD_B(T a, T b, T c, T dx, T M) -> T {
  if (std::abs(a) > M * dx * dx) {
    return MINMOD(a, b, c);
  }
  return a;
}

auto barth_jespersen(double U_v_L, double U_v_R, double U_c_L, double U_c_T,
                     double U_c_R, double alpha) -> double;

void detect_troubled_cells(const AthelasArray3D<double> U,
                           AthelasArray1D<double> D, const GridStructure &grid,
                           const basis::NodalBasis &basis,
                           const IndexRange &vb);

/**
 * Return the cell average of a field q on cell ix.
 * This version uses the naive ubar = \sum_q w_q u_q / sum_w
 * The parameter `int extrapolate` designates how the cell average is
 *computed.
 *  0  : Return standard cell average on ix
 * -1 : Extrapolate left, e.g.,  polynomial from ix+1 into ix
 * +1 : Extrapolate right, e.g.,  polynomial from ix-1 into ix
 **/
KOKKOS_INLINE_FUNCTION
auto cell_average(AthelasArray3D<double> U, AthelasArray1D<double> weights,
                  const double dr, const int v, const int i,
                  const int extrapolate = 0) -> double {
  assert((extrapolate == -1 || extrapolate == 0 || extrapolate == 1) &&
         "cell_average:: extrapolate must be -1, 0, 1");
  static const int nNodes = static_cast<int>(weights.size());

  // Some data structures include interface storage -- do some index gymnastics
  static const int nq_p_i =
      static_cast<int>(weights.extent(0)) + 2; // size of nodes + interfaces
  const int nq_u = static_cast<int>(U.extent(1));
  const int offset = (nq_u == nq_p_i) ? 1 : 0;

  double avg = 0.0;

  for (int q = 0; q < nNodes; ++q) {
    const double w = weights(q);
    avg += w * U(i + extrapolate, q + offset, v);
  }
  return avg;
}

/**
 * Return the cell average of a field q on cell ix.
 * The parameter `int extrapolate` designates how the cell average is
 *computed.
 *  0  : Return standard cell average on ix
 * -1 : Extrapolate left, e.g.,  polynomial from ix+1 into ix
 * +1 : Extrapolate right, e.g.,  polynomial from ix-1 into ix
 **/
KOKKOS_INLINE_FUNCTION
auto cell_average(AthelasArray3D<double> U, AthelasArray2D<double> sqrt_gm,
                  AthelasArray1D<double> weights, const double dr, const int v,
                  const int i, const int extrapolate = 0) -> double {
  assert((extrapolate == -1 || extrapolate == 0 || extrapolate == 1) &&
         "cell_average:: extrapolate must be -1, 0, 1");
  static const int nNodes = static_cast<int>(weights.size());

  // Some data structures include interface storage -- do some index gymnastics
  static const int nq_p_i =
      static_cast<int>(sqrt_gm.extent(1)); // size of nodes + interfaces
  const int nq_u = static_cast<int>(U.extent(1));
  const int offset = (nq_u == nq_p_i) ? 1 : 0;

  double avg = 0.0;
  double vol = 0.0;

  for (int q = 0; q < nNodes; ++q) {
    const double w = weights(q);
    const auto dv = w * sqrt_gm(i, q + 1) * dr;
    vol += dv;
    avg += U(i + extrapolate, q + offset, v) * dv;
  }
  return avg / vol;
}

/**
 * Return the cell average of a field q on cell ix.
 * Takes a specific quantity (nx, nq) instead of a field (nx, nq, nv).
 **/
KOKKOS_INLINE_FUNCTION
auto cell_average(AthelasArray2D<double> U, AthelasArray2D<double> sqrt_gm,
                  AthelasArray1D<double> weights, const double dr, const int i,
                  const int extrapolate = 0) -> double {
  assert((extrapolate == -1 || extrapolate == 0 || extrapolate == 1) &&
         "cell_average:: extrapolate must be -1, 0, 1");
  static const int nNodes = static_cast<int>(weights.size());

  // Some data structures include interface storage -- do some index gymnastics
  static const int nq_p_i =
      static_cast<int>(sqrt_gm.extent(1)); // size of nodes + interfaces
  const int nq_u = static_cast<int>(U.extent(1));
  const int offset = (nq_u == nq_p_i) ? 0 : 1;

  double avg = 0.0;
  double vol = 0.0;

  for (int q = 0; q < nNodes; ++q) {
    const double w = weights(q);
    const auto dv = w * sqrt_gm(i, q + 1) * dr;
    vol += dv;
    avg += U(i + extrapolate, q + offset) * dv;
  }
  return avg / vol;
}

void modify_polynomial(AthelasArray3D<double> U,
                       AthelasArray2D<double> modified_polynomial,
                       double gamma_i, double gamma_l, double gamma_r, int ix,
                       int q);

auto smoothness_indicator(AthelasArray3D<double> U,
                          AthelasArray2D<double> modified_polynomial,
                          const GridStructure &grid,
                          const basis::NodalBasis &basis, int ix, int i,
                          int iCQ) -> double;

auto non_linear_weight(double gamma, double beta, double tau, double eps)
    -> double;

auto weno_tau(double beta_l, double beta_i, double beta_r, double weno_r)
    -> double;
} // namespace athelas
