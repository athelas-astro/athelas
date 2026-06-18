#pragma once

#include "Kokkos_Macros.hpp"
#include "geometry/mesh.hpp"
#include "limiters/slope_limiter.hpp"
#include "math/utils.hpp"

namespace athelas {

void conservative_correction(AthelasArray3D<double> u_k,
                             AthelasArray3D<double> evolved, const Mesh &mesh,
                             const IndexRange &vb);

auto initialize_slope_limiter(std::string field, const Mesh *mesh,
                              const ProblemIn *pin, IndexRange vars)
    -> SlopeLimiter;

// Standard MINMOD function
template <typename T>
constexpr auto MINMOD(T a, T b, T c) -> T {
  using math::utils::sgn;
  if (sgn(a) == sgn(b) && sgn(b) == sgn(c)) {
    return sgn(a) * std::min({std::abs(a), std::abs(b), std::abs(c)});
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

// TVD(B) minmod limit of a single scalar slope against the neighbour
// cell-average differences on a (possibly non-uniform) stencil. Returns the
// limited slope and reports via `changed` whether it moved beyond the relative
// threshold. Shared by the component-wise and characteristic paths of the slope
// limiters so the non-uniform-mesh slope arithmetic lives in one place.
KOKKOS_INLINE_FUNCTION
auto limit_slope_minmod(const double slope, const double avg_m,
                        const double avg_i, const double avg_p,
                        const double dr_i, const double dr_m, const double dr_p,
                        const double b_tvd, const double m_tvb,
                        const double threshold, bool &changed) -> double {
  const double s_p = (avg_p - avg_i) / (0.5 * (dr_i + dr_p));
  const double s_m = (avg_i - avg_m) / (0.5 * (dr_i + dr_m));
  const double scale = 0.5 * dr_i;
  const double limited =
      MINMOD_B(slope, b_tvd * scale * s_p, b_tvd * scale * s_m, dr_i, m_tvb);
  changed = std::abs(limited - slope) > threshold * std::abs(slope);
  return limited;
}

auto barth_jespersen(double U_v_L, double U_v_R, double U_c_L, double U_c_T,
                     double U_c_R, double alpha) -> double;

void detect_troubled_cells(const AthelasArray3D<double> U,
                           AthelasArray1D<double> D, const Mesh &mesh,
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
  const int nNodes = static_cast<int>(weights.size());

  // Some data structures include interface storage -- do some index gymnastics
  const int nq_p_i =
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
  const int nNodes = static_cast<int>(weights.size());

  // Some data structures include interface storage -- do some index gymnastics
  const int nq_p_i =
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

KOKKOS_INLINE_FUNCTION
auto cell_average_mass(AthelasArray3D<double> U, AthelasArray2D<double> dm_deta,
                       AthelasArray1D<double> weights, const int v, const int i,
                       const int extrapolate = 0) -> double {
  assert((extrapolate == -1 || extrapolate == 0 || extrapolate == 1) &&
         "cell_average_mass:: extrapolate must be -1, 0, 1");
  const int nNodes = static_cast<int>(weights.size());

  const int nq_m = static_cast<int>(dm_deta.extent(1));
  const int nq_u = static_cast<int>(U.extent(1));
  const int offset = (nq_u == nq_m) ? 0 : 1;

  double avg = 0.0;
  double mass = 0.0;
  for (int q = 0; q < nNodes; ++q) {
    const double dm = weights(q) * dm_deta(i, q);
    mass += dm;
    avg += U(i + extrapolate, q + offset, v) * dm;
  }
  return avg / mass;
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
  const int nNodes = static_cast<int>(weights.size());

  // Some data structures include interface storage -- do some index gymnastics
  const int nq_p_i =
      static_cast<int>(sqrt_gm.extent(1)); // size of nodes + interfaces
  const int nq_u = static_cast<int>(U.extent(1));
  const int offset = (nq_u == nq_p_i) ? 1 : 0;

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

KOKKOS_FUNCTION
void modify_polynomial(AthelasArray3D<double> U,
                       AthelasArray2D<double> modified_polynomial, double dr,
                       double dr_m, double dr_p, double gamma_i, double gamma_l,
                       double gamma_r, int ix, int v);

KOKKOS_FUNCTION
auto smoothness_indicator(AthelasArray2D<double> modified_polynomial,
                          const Mesh &mesh, int poly_idx, int v) -> double;

auto non_linear_weight(double gamma, double beta, double tau, double weno_p,
                       double eps) -> double;

auto weno_tau(double beta_l, double beta_i, double beta_r) -> double;
} // namespace athelas
