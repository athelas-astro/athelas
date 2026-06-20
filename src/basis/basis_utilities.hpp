#pragma once

/**
 * @file basis_utilities.hpp
 * --------------
 *
 * @brief Shared discontinuous-Galerkin operators built on a nodal basis.
 */

#include "basic_types.hpp"
#include "kokkos_abstraction.hpp"
#include "kokkos_types.hpp"
#include "loop_layout.hpp"

namespace athelas::basis {

/**
 * @brief DG surface (face) term of a flux divergence.
 *
 * Subtracts the face-flux contribution from the stage delta:
 *   delta(stage,i,q,v) -= dFlux_num(i+1,v) phi(i,nNodes+1,q)
 * sqrt_gm(i,nNodes+1)
 *                       - dFlux_num(i+0,v) phi(i,0,q)        sqrt_gm(i,0)
 * for every interior cell i, quadrature node q, and variable v in [vb.s, vb.e].
 *
 * Shared by the fluid and rad-hydro divergence kernels: a single nodal basis
 * applies to every evolved variable, so the same operator serves both.
 */
inline void
surface_term(AthelasArray4D<double> delta, AthelasArray2D<double> dFlux_num,
             AthelasArray3D<double> phi, AthelasArray2D<double> sqrt_gm,
             const int stage, const int nNodes, const IndexRange &ib,
             const IndexRange &qb, const IndexRange &vb, const char *label) {
  athelas::par_for(
      DEFAULT_LOOP_PATTERN, label, DevExecSpace(), ib.s, ib.e, qb.s, qb.e,
      KOKKOS_LAMBDA(const int i, const int q) {
        for (int v = vb.s; v <= vb.e; ++v) {
          delta(stage, i, q, v) -=
              (+dFlux_num(i + 1, v) * phi(i, nNodes + 1, q) *
                   sqrt_gm(i, nNodes + 1) -
               dFlux_num(i + 0, v) * phi(i, 0, q) * sqrt_gm(i, 0));
        }
      });
}

/**
 * @brief DG volume term of a flux divergence.
 *
 * Adds the quadrature-weighted volume flux to the stage delta:
 *   delta(stage,i,p,v) += sum_q weights(q) dphi(i,q+1,p) sqrt_gm(i,q+1)
 * f_v(i,q) for every interior cell i, basis dof p, and variable v in [0,
 * NVARS).
 *
 * The per-node flux vector is supplied by @p flux_at_node, a device callable
 * (i, q) -> Kokkos::Array<double, NVARS> returning the fluxes in variable order
 * (so f[v] pairs with delta(...,v)). This keeps the physics in the caller while
 * sharing the quadrature/accumulate skeleton across the fluid and rad-hydro
 * divergence kernels.
 */
template <int NVARS, class FluxFunction>
inline void
volume_term(AthelasArray4D<double> delta, AthelasArray3D<double> dphi,
            AthelasArray1D<double> weights, AthelasArray2D<double> sqrt_gm,
            const int stage, const int nNodes, const IndexRange &ib,
            const IndexRange &qb, const FluxFunction flux_func,
            const char *label) {
  athelas::par_for(
      DEFAULT_LOOP_PATTERN, label, DevExecSpace(), ib.s, ib.e, qb.s, qb.e,
      KOKKOS_LAMBDA(const int i, const int p) {
        Kokkos::Array<double, NVARS> local_sum{};
        for (int q = 0; q < nNodes; ++q) {
          const double w_dphi_sqrtgm =
              weights(q) * dphi(i, q + 1, p) * sqrt_gm(i, q + 1);
          const Kokkos::Array<double, NVARS> flux = flux_func(i, q);
          for (int v = 0; v < NVARS; ++v) {
            local_sum[v] += w_dphi_sqrtgm * flux[v];
          }
        }
        for (int v = 0; v < NVARS; ++v) {
          delta(stage, i, p, v) += local_sum[v];
        }
      });
}

} // namespace athelas::basis
