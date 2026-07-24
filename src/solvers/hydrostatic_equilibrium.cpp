#include "Kokkos_Core.hpp"

#include <algorithm>

#include "composition/composition.hpp"
#include "eos/eos.hpp"
#include "geometry/mesh.hpp"
#include "interface/state.hpp"
#include "kokkos_abstraction.hpp"
#include "kokkos_types.hpp"
#include "loop_layout.hpp"
#include "math/interp.hpp"
#include "solvers/hydrostatic_equilibrium.hpp"
#include "utils/constants.hpp"

namespace athelas {

using math::interp::find_closest_cell;
using math::interp::lagrange_interp;

auto HydrostaticEquilibrium::rhs(const double mass_enc, const double p,
                                 const double r) const -> double {
  // Valid over the whole physical range p > 0 (matching dm/dr).  The surface
  // pressure p_threshold_ is a loop event located by the caller, NOT a term in
  // the equation: cutting the RHS off at p_threshold_ would let an RK stage that
  // dips below it -- while still positive -- evaluate a different (zeroed)
  // equation than dm/dr, making the final step internally inconsistent.
  if (p <= 0.0) {
    return 0.0;
  }
  static constexpr double G = constants::G_GRAV;
  const double rho = std::pow(p / k_, n_ / (n_ + 1.0));
  return -G * mass_enc * rho / (r * r);
}

void HydrostaticEquilibrium::solve(MeshState &mesh_state, Mesh *mesh,
                                   ProblemIn *pin) {
  auto derived = mesh_state(0).get_field("derived");
  auto sd0 = mesh_state(0);

  const int ihi = mesh->get_ihi();
  const int nNodes = mesh->n_nodes();
  const double rmax = mesh->get_x_r();
  const double dr_bulk = rmax / ihi;
  const double vel = 0.0;
  const double energy = 0.0;
  double lambda[eos::EOS_LAMBDA_SIZE] = {};
  if (mesh_state.enabled("ionization")) {
    atom::paczynski_terms(sd0, 1, 0, lambda);
  }
  const auto &eos = mesh_state.eos();
  const double p_c =
      pressure_from_conserved(eos, 1.0 / rho_c_, vel, energy, lambda);

  // Start the outward integration very close to the centre (well inside the
  // first mesh node) from the regular-center expansion, then march on a refined
  // grid (set in the loop).  A uniform step from near the origin stalls RK4 at
  // 2nd order because of the 1/r^2 source; refining there -- with the Taylor
  // start p = P_c - (2 pi / 3) G rho_c^2 r^2 + O(r^4),
  // m = (4 pi / 3) rho_c r^3 + O(r^5) -- restores the design order.
  const double r0 = rmax * 1.0e-6;
  double m_enc = constants::FOUR_THIRDS_PI * rho_c_ * r0 * r0 * r0;

  // host data
  auto h_derived = Kokkos::create_mirror_view(derived);

  const int size = mesh->n_elements() * (nNodes + 2) + 2 * (nNodes + 2);
  AthelasArray1D<double> d_r("host radius", size);
  std::vector<double> pressure(1);
  std::vector<double> radius(1);
  auto r = mesh->nodal_grid();
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Solvers :: Hydrostatic :: copy grid",
      DevExecSpace(), 1, ihi, KOKKOS_LAMBDA(const int i) {
        for (int q = 0; q < nNodes + 2; ++q) {
          const double rq = r(i, q);
          d_r(i * (nNodes + 2) + q) = rq;
        }
      });
  auto h_r = Kokkos::create_mirror_view(d_r);
  Kokkos::deep_copy(h_r, d_r);
  radius[0] = r0;
  pressure[0] = p_c - (constants::FOURPI / 6.0) * constants::G_GRAV * rho_c_ *
                          rho_c_ * r0 * r0;

  // dm/dr = 4 pi rho r^2 for the coupled RK4 below (dp/dr is the member rhs()).
  const auto dm_dr = [&](const double pp, const double rr) -> double {
    const double rho = (pp > 0.0) ? std::pow(pp / k_, n_ / (n_ + 1.0)) : 0.0;
    return constants::FOURPI * rho * rr * rr;
  };

  int i = 0;
  while (pressure.back() > p_threshold_) {
    const double r = radius[i];
    const double p = pressure[i];
    // Coupled RK4 for the (pressure, enclosed-mass) system,
    //   dp/dr = -G m rho / r^2,   dm/dr = 4 pi rho r^2,
    // advancing both together 
    const double kp1 = rhs(m_enc, p, r);
    const double km1 = dm_dr(p, r);
    // Center-and-surface-refined step: bound dr by r (resolving the stiff
    // 1/r^2 source near the origin) and by the local pressure scale height
    // p / |dp/dr| (resolving the surface, where a dr ~ r log grid would
    // otherwise coarsen), capped at the uniform bulk step.
    const double hp = (kp1 != 0.0) ? p / std::abs(kp1) : r;
    const double dr = std::min(std::min(r, hp) / ihi, dr_bulk);
    const double kp2 =
        rhs(m_enc + 0.5 * dr * km1, p + 0.5 * dr * kp1, r + 0.5 * dr);
    const double km2 = dm_dr(p + 0.5 * dr * kp1, r + 0.5 * dr);
    const double kp3 =
        rhs(m_enc + 0.5 * dr * km2, p + 0.5 * dr * kp2, r + 0.5 * dr);
    const double km3 = dm_dr(p + 0.5 * dr * kp2, r + 0.5 * dr);
    const double kp4 = rhs(m_enc + dr * km3, p + dr * kp3, r + dr);
    const double km4 = dm_dr(p + dr * kp3, r + dr);

    const double dp = (dr / 6.0) * (kp1 + 2.0 * kp2 + 2.0 * kp3 + kp4);
    const double dm = (dr / 6.0) * (km1 + 2.0 * km2 + 2.0 * km3 + km4);
    m_enc += dm;
    const double new_p = p + dp;
    // safety first
    if (std::isnan(new_p)) {
      std::println("NaN pressure found in hydrostatic equilibrium solve!");
      break;
    }
    pressure.push_back(std::max(new_p, p_threshold_));
    radius.push_back(r + dr);

    if (new_p <= p_threshold_) {
      break;
    }
    i++;
  }

  // Precise truncation radius: the marching loop overshoots p_threshold_ by up
  // to one step, quantizing R to O(dr) and injecting grid noise at the surface
  // at every resolution.  Cubic-invert r(p) through the last four samples above
  // threshold (p is monotone in r) and terminate the profile exactly at
  // p = p_threshold_.
  {
    int hi = static_cast<int>(pressure.size()) - 1;
    while (hi > 0 && pressure[hi] <= p_threshold_) {
      --hi;
    }
    if (hi >= 3) {
      // Cubic-invert r(p) through the last four samples above threshold.
      const double r_surf =
          lagrange_interp(pressure, radius, hi - 3, 4, p_threshold_);
      radius.resize(hi + 2);
      pressure.resize(hi + 2);
      radius[hi + 1] = r_surf;
      pressure[hi + 1] = p_threshold_;
    }
  }

  std::println("# Hydrostatic Equilibrium Solver ::");
  std::println("# Integrated mass = {:.5e}\n", m_enc);
  std::println("# Radius = {:.5e}", radius.back());
  std::println("# Dynamical time = {:.5e}",
               std::sqrt((radius.back() * radius.back() * radius.back()) /
                         (constants::G_GRAV * m_enc)));
  std::println("# Free-fall time = {:.5e}",
               1.0 / std::sqrt(constants::G_GRAV * rho_c_));

  // update domain boundary and mesh
  auto &xr = pin->param()->get_mutable_ref<double>("mesh.xr");
  xr = radius.back();
  // this is awful
  // TODO(astrobarker): when cleaning up mesh, get this
  auto newgrid = Mesh(pin);
  *mesh = newgrid;

  // refill host radius array
  auto r_new = mesh->nodal_grid();
  auto node_r_h = Kokkos::create_mirror_view(r_new);
  Kokkos::deep_copy(node_r_h, r_new);
  for (int i = 1; i <= ihi; ++i) {
    for (int q = 0; q < nNodes + 2; ++q) {
      const double rq = node_r_h(i, q);
      h_r(i * (nNodes + 2) + q) = rq;
    }
  }

  // Interpolate the pressure profile onto the mesh with a 4-point cubic
  // Lagrange stencil.  Plain linear interpolation here is only 2nd-order
  // accurate.
  const int npts = static_cast<int>(radius.size());
  const int stencil = std::min(4, npts);
  for (int ix = 1; ix <= ihi; ++ix) {
    for (int q = 0; q < nNodes + 2; ++q) {
      const double rq = h_r(ix * (nNodes + 2) + q);
      const int idx = find_closest_cell(radius, rq, npts);
      const int j0 = std::clamp(idx - 1, 0, std::max(npts - stencil, 0));
      h_derived(ix, q, iP_) = lagrange_interp(radius, pressure, j0, stencil, rq);
    }
  }

  Kokkos::deep_copy(derived, h_derived);
}

} // namespace athelas
