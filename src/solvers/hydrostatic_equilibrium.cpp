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

void HydrostaticEquilibrium::solve(State *state, GridStructure *grid,
                                   ProblemIn *pin) {
  auto uAF = state->u_af();

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
  if (state->ionization_enabled()) {
    atom::paczynski_terms(state, 1, 0, lambda);
  }
  const double p_c = pressure_from_conserved(eos_, rho_c_, vel, energy, lambda);

  const double r_c = grid->node_coordinate(ilo, 0);
  double m_enc = (constants::FOURPI / 3.0) * (r_c * r_c * r_c) * rho_c_;

  // host data
  auto h_uAF = Kokkos::create_mirror_view(uAF);

  const int size = grid->n_elements() * nNodes + 2 * nNodes;
  AthelasArray1D<double> d_r("host radius", size);
  std::vector<double> pressure(1);
  std::vector<double> radius(1);
  athelas::par_for(
      DEFAULT_FLAT_LOOP_PATTERN, "Solvers :: Hydrostatic :: copy grid",
      DevExecSpace(), 0, ihi, KOKKOS_LAMBDA(const int i) {
        for (int q = 0; q < nNodes; ++q) {
          const double r = grid->node_coordinate(i + 1, q);
          d_r(i * nNodes + q) = r;
        }
      });
  auto h_r = Kokkos::create_mirror_view(d_r);
  Kokkos::deep_copy(h_r, d_r);
  pressure[0] = p_c;
  radius[0] = r_c;

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
    const double new_p = p + (1.0 / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    // safety first
    if (std::isnan(new_p)) {
      std::println("NaN pressure found in hydrostatic equilibrium solve!");
      break;
    }
    pressure.push_back(new_p);
    ;
    radius.push_back(r + dr);

    i++;
  }
  std::println("# Hydrostatic Equilibrium Solver ::");
  std::println("# Integrated mass = {:.5e}\n", m_enc);
  std::println("# Radius = {:.5e}", radius.back());
  std::println("# Dynamical time = {:.5e}",
               std::sqrt((radius.back() * radius.back() * radius.back()) /
                         (constants::G_GRAV * m_enc)));

  // update domain boundary and grid
  auto &xr = pin->param()->get_mutable_ref<double>("problem.xr");
  xr = radius.back();
  // this is awful
  // TODO(astrobarker): when cleaning up grid, get this
  auto newgrid = GridStructure(pin);
  *grid = newgrid;

  // refill host radius array
  for (int ix = 0; ix <= ihi; ++ix) {
    for (int q = 0; q < nNodes; ++q) {
      h_r(ix * nNodes + q) = grid->node_coordinate(ix, q);
    }
  }

  // now we have to interpolate onto our grid
  // This is horrible but it's fine, only happens once.
  for (int ix = 0; ix <= ihi; ++ix) {
    for (int q = 0; q < nNodes; ++q) {
      const double r = h_r(ix * nNodes + q);
      for (size_t i = 0; i < pressure.size() - 2; ++i) {
        if (radius[i] <= r && radius[i + 1] >= r) { // search
          const double y = LINTERP(radius[i], radius[i + 1], pressure[i],
                                   pressure[i + 1], r);
          h_uAF(ix, q, iP_) = y;
          break;
        }
      }
    }
  }

  Kokkos::deep_copy(uAF, h_uAF);
}

} // namespace athelas
