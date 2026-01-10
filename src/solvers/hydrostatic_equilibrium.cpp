#include "Kokkos_Core.hpp"

#include "composition/composition.hpp"
#include "geometry/grid.hpp"
#include "kokkos_abstraction.hpp"
#include "kokkos_types.hpp"
#include "loop_layout.hpp"
#include "solvers/hydrostatic_equilibrium.hpp"
#include "state/state.hpp"
#include "utils/constants.hpp"
#include "utils/utilities.hpp"

namespace athelas {

using utilities::LINTERP;

auto HydrostaticEquilibrium::rhs(const double mass_enc, const double p,
                                 const double r) const -> double {
  static constexpr double G = constants::G_GRAV;
  const double rho = std::pow(p / k_, n_ / (n_ + 1.0));
  return -G * mass_enc * rho / (r * r);
}

void HydrostaticEquilibrium::solve(MeshState &mesh_state, GridStructure *grid,
                                   ProblemIn *pin) {
  auto uAF = mesh_state(0).get_field("u_af");
  auto sd0 = mesh_state(0);

  static constexpr int ilo = 1;
  const int ihi = grid->get_ihi();
  const int nNodes = grid->n_nodes();
  const double rmax = grid->get_x_r();
  const double dr = rmax / ihi;
  // Subtely: do we associate rho_c_ with the inner boundary or first nodal
  // point?
  const double vel = 0.0;
  const double energy = 0.0;
  double lambda[8];
  if (mesh_state.ionization_enabled()) {
    atom::paczynski_terms(sd0, 1, 0, lambda);
  }
  const auto &eos = mesh_state.eos();
  const double p_c = pressure_from_conserved(eos, rho_c_, vel, energy, lambda);

  const double r_c = grid->node_coordinate(ilo, 0);
  double m_enc = (constants::FOURPI / 3.0) * (r_c * r_c * r_c) * rho_c_;

  // host data
  auto h_uAF = Kokkos::create_mirror_view(uAF);

  const int size = grid->n_elements() * (nNodes + 2) + 2 * nNodes;
  AthelasArray1D<double> d_r("host radius", size);
  std::vector<double> pressure(1);
  std::vector<double> radius(1);
  auto x_l = grid->x_l();
  auto r = grid->nodal_grid();
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Solvers :: Hydrostatic :: copy grid",
      DevExecSpace(), 1, ihi, KOKKOS_LAMBDA(const int i) {
        for (int q = 0; q < nNodes + 2; ++q) {
          const double rq =
              (q == 0) ? x_l(i)
                       : ((q == nNodes + 1) ? x_l(i + 1) : r(i, q - 1));
          d_r(i * (nNodes + 2) + q) = rq;
        }
      });
  auto h_r = Kokkos::create_mirror_view(d_r);
  Kokkos::deep_copy(h_r, d_r);
  pressure[0] = p_c;
  radius[0] = h_r[nNodes + 3];

  int i = 0;
  while (pressure.back() > p_threshold_) {
    const double r = radius[i];
    const double p = pressure[i];
    const double rho = std::pow(p / k_, n_ / (n_ + 1.0));

    // RK4
    // NOTE: Currently holding m constant through the stages!
    const double k1 = dr * rhs(m_enc, p, r);
    const double k2 = dr * rhs(m_enc, p + 0.5 * k1, r + 0.5 * dr);
    const double k3 = dr * rhs(m_enc, p + 0.5 * k2, r + 0.5 * dr);
    const double k4 = dr * rhs(m_enc, p + k3, r + dr);

    m_enc += constants::FOURPI * rho * r * r * dr;
    const double dp = (1.0 / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    const double new_p = p + dp;
    // safety first
    if (std::isnan(new_p)) {
      std::println("NaN pressure found in hydrostatic equilibrium solve!");
      break;
    }
    pressure.push_back(new_p);
    radius.push_back(r + dr);

    i++;
  }
  std::println("# Hydrostatic Equilibrium Solver ::");
  std::println("# Integrated mass = {:.5e}\n", m_enc);
  std::println("# Radius = {:.5e}", radius.back());
  std::println("# Dynamical time = {:.5e}",
               std::sqrt((radius.back() * radius.back() * radius.back()) /
                         (constants::G_GRAV * m_enc)));
  std::println("# Free-fall time = {:.5e}",
               1.0 / std::sqrt(constants::G_GRAV * rho_c_));

  // update domain boundary and grid
  auto &xr = pin->param()->get_mutable_ref<double>("problem.xr");
  xr = radius.back();
  // this is awful
  // TODO(astrobarker): when cleaning up grid, get this
  auto newgrid = GridStructure(pin);
  *grid = newgrid;

  // refill host radius array
  auto r_new = grid->nodal_grid();
  auto x_l_new = grid->x_l();
  auto node_r_h = Kokkos::create_mirror_view(r_new);
  auto x_l_h = Kokkos::create_mirror_view(x_l_new);
  for (int i = 1; i <= ihi; ++i) {
    for (int q = 0; q < nNodes + 2; ++q) {
      const double rq =
          (q == 0) ? x_l_h(i)
                   : ((q == nNodes + 1) ? x_l_h(i + 1) : node_r_h(i, q - 1));
      h_r(i * (nNodes + 2) + q) = rq;
    }
  }

  // now we have to interpolate onto our grid
  for (int ix = 1; ix <= ihi; ++ix) {
    for (int q = 0; q < nNodes + 2; ++q) {
      const double rq = h_r(ix * (nNodes + 2) + q);
      const int idx = utilities::find_closest_cell(radius, rq, radius.size());
      const double y = LINTERP(radius[idx], radius[idx + 1], pressure[idx],
                               pressure[idx + 1], rq);
      h_uAF(ix, q, iP_) = y;
    }
  }

  Kokkos::deep_copy(uAF, h_uAF);
}

} // namespace athelas
